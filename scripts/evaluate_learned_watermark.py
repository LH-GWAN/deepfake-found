"""Measure the learned watermark against the real, non-differentiable swapper.

Training optimises against a differentiable stand-in for a face swap. A model can
learn that stand-in rather than the thing it stands for, so the number that counts
comes from the graphics swapper in ``build_manipulation_set.py``, which never
appears in the training graph: 106 landmarks, a Delaunay warp, colour matching and
Poisson blending.

Three stages are reported, because knowing which one loses the signal is what
tells you where to work next.

crop
    Encode and decode inside the aligned frame. This is the model's own ceiling.
roundtrip
    Warp the residual back into the full photograph, re-detect the face and align
    it again. Isolates what the coordinate round trip costs.
swap
    Run the real swapper with the marked photograph as the source - the direction
    the product promise depends on - and decode from its output.

``--orientation transposed`` decodes with the image transposed. Checkpoints
trained before the displacement grid was corrected saw every image transposed,
because the grid was built in (row, column) order where ``grid_sample`` reads
(x, y); with square crops that transposes rather than fails. Such a checkpoint
decodes correctly only in the orientation it was trained on. The 9,000-step
``models/learned_watermark.pt`` is one and needs the flag; every later checkpoint
(v2, v3 and anything the training script writes now) is upright, which is the
default, and the training script records the orientation in the checkpoint so
this can be checked rather than remembered.

Usage:
    python scripts/evaluate_learned_watermark.py --checkpoint models/learned_watermark.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_manipulation_set as swaps
from learned_watermark import BITS, Decoder, Encoder, pick_device

from deepshield.face.backends import ARCFACE_TEMPLATE_112

SIZE = 128


def align_matrix(cv2: Any, app: Any, image: np.ndarray) -> np.ndarray | None:
    """Return the similarity transform onto the canonical template."""
    face = swaps.largest_face(app, image)
    if face is None:
        return None
    matrix, _ = cv2.estimateAffinePartial2D(
        np.asarray(face.kps, dtype=np.float32),
        ARCFACE_TEMPLATE_112 * (SIZE / 112.0),
        method=cv2.LMEDS,
    )
    return None if matrix is None else np.asarray(matrix)


def as_tensor(image: np.ndarray, device: str, transposed: bool) -> torch.Tensor:
    """Return the image as a batch of one, in the decoder's orientation."""
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
    tensor = tensor.float().div(127.5).sub(1).unsqueeze(0).to(device)
    return tensor.transpose(2, 3).contiguous() if transposed else tensor


def identity_pairs(faces: Path, limit: int) -> list[tuple[Path, Path]]:
    """Pair each identity with the next, so no pair shares a person."""
    by_identity: dict[str, list[Path]] = {}
    for path in sorted(faces.glob("*.jpg")):
        by_identity.setdefault(path.stem.rsplit("_", 1)[0], []).append(path)
    names = sorted(by_identity)
    pairs = [
        (by_identity[names[index]][0], by_identity[names[index + 1]][0])
        for index in range(len(names) - 1)
    ]
    return pairs[:limit]


def main(argv: list[str] | None = None) -> int:
    """Report bit accuracy at each stage of the source-direction pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--faces", type=Path, default=Path("data/test/manipulated/real"))
    parser.add_argument("--models", type=Path, default=Path("models/insightface"))
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--orientation", choices=("transposed", "upright"), default=None)
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        raise SystemExit(
            f"missing {args.checkpoint}; run scripts/train_learned_watermark.py first"
        )
    pairs = identity_pairs(args.faces, args.pairs)
    if not pairs:
        raise SystemExit(f"no identity pairs found in {args.faces}")

    import cv2

    from deepshield.media import load_image

    device = pick_device()
    state = torch.load(args.checkpoint, map_location=device)
    encoder, decoder = Encoder().to(device), Decoder().to(device)
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    encoder.eval()
    decoder.eval()
    scale = state["res_scale"]
    orientation = args.orientation or state.get("orientation", "upright")
    transposed = orientation == "transposed"
    app = swaps.build_analyzer(args.models)

    rng = np.random.default_rng(0)
    stages: dict[str, list[float]] = {"crop": [], "roundtrip": [], "swap": []}
    quality: list[float] = []
    for own_path, other_path in pairs:
        own, other = load_image(own_path), load_image(other_path)
        matrix = align_matrix(cv2, app, own)
        if matrix is None:
            continue
        crop = cv2.warpAffine(own, matrix, (SIZE, SIZE))
        message = torch.from_numpy(rng.integers(0, 2, (1, BITS)).astype(np.float32)).to(device)

        def read(image: np.ndarray, bound: torch.Tensor = message) -> float | None:
            found = align_matrix(cv2, app, image)
            if found is None:
                return None
            aligned = cv2.warpAffine(image, found, (SIZE, SIZE))
            with torch.no_grad():
                bits = (decoder(as_tensor(aligned, device, transposed)) > 0).float()
            return float((bits == bound).float().mean().item())

        with torch.no_grad():
            residual = encoder(as_tensor(crop, device, False), message) * scale
            marked_crop = (as_tensor(crop, device, False) + residual).clamp(-1, 1)
            if transposed:
                marked_crop = marked_crop.transpose(2, 3).contiguous()
            read_bits = (decoder(marked_crop) > 0).float()
        stages["crop"].append(float((read_bits == message).float().mean().item()))

        shift = residual.squeeze(0).permute(1, 2, 0).cpu().numpy() * 127.5
        spread = cv2.warpAffine(
            shift, cv2.invertAffineTransform(matrix), (own.shape[1], own.shape[0])
        )
        marked = np.clip(own.astype(np.float32) + spread, 0, 255).astype(np.uint8)
        error = ((marked.astype(float) - own.astype(float)) ** 2).mean()
        quality.append(float(10 * np.log10(255.0**2 / max(error, 1e-9))))

        recovered = read(marked)
        if recovered is not None:
            stages["roundtrip"].append(recovered)
        swapped = swaps.swap_face(cv2, marked, other, app)
        if swapped is not None:
            recovered = read(swapped)
            if recovered is not None:
                stages["swap"].append(recovered)

    report = {
        "pairs": len(pairs),
        "orientation": orientation,
        "psnr_db": round(float(np.mean(quality)), 2) if quality else None,
        "stages": {
            name: {
                "n": len(values),
                "bit_accuracy": round(float(np.mean(values)), 4) if values else None,
                "exact": int(sum(value == 1.0 for value in values)),
            }
            for name, values in stages.items()
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "learned_watermark_under_swap.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
