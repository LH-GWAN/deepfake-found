r"""Measure gradient-based identity cloaking against a differentiable surrogate.

The cloaking layer shipped in :mod:`deepshield.protection.adversarial` is
gradient-free (SPSA) because the production embedders are ONNX sessions. It was
measured to move an embedding less than JPEG compression does, which left one
question open: is that the ceiling of cloaking, or the ceiling of a gradient
estimate made from forward passes? This script answers it by attacking a model
with real gradients and then measuring the result on the models the pipeline
actually uses.

surrogate
    FaceNet (InceptionResnetV1 trained on VGGFace2, from ``facenet-pytorch``),
    a different architecture and training set from ArcFace and SFace, so
    transfer is measured across genuinely different models rather than between
    two copies of one.
attack
    Projected gradient ascent on cosine distance from the clean embedding, in
    the aligned 160-pixel face frame, with an expectation over differentiable
    resampling, blur and noise so the perturbation is not a point solution.
    The perturbation is warped back into the photograph exactly as the learned
    watermark's residual is, so every downstream measurement re-detects and
    re-aligns the face the way the pipeline would.
report
    Three displacements per budget, never conflated: on the surrogate itself,
    on ArcFace and SFace through the project's own detector and aligner, and
    after JPEG-70 and a 50% resize. The identity decision the matcher would
    make is reported next to them, because a displacement that leaves the
    similarity above the high-confidence threshold protects nobody.

Usage:
    python scripts/evaluate_cloaking_whitebox.py --images data/test/eval_faces --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learned_watermark import gaussian_blur, pick_device

from deepshield.config import FaceEmbedderConfig, load_config
from deepshield.face.aligner import build_aligner
from deepshield.face.backends import ARCFACE_TEMPLATE_112
from deepshield.face.detector import build_detector
from deepshield.face.embedder import build_embedder
from deepshield.media import IMAGE_SUFFIXES, load_image
from deepshield.quality import psnr, ssim
from deepshield.transforms import Transformation

SIZE = 160
STEPS = 300


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return one minus the cosine similarity of two vectors."""
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_surrogate(device: str) -> Any:
    """Return FaceNet in eval mode, frozen."""
    try:
        from facenet_pytorch import InceptionResnetV1
    except ImportError as exc:
        raise SystemExit("pip install --no-deps facenet-pytorch (its pins are stale)") from exc
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def eot(image: torch.Tensor) -> torch.Tensor:
    """Apply one random differentiable resampling, blur and noise."""
    scale = float(torch.empty(1).uniform_(0.7, 1.0).item())
    side = max(64, int(round(SIZE * scale)))
    small = F.interpolate(image, size=(side, side), mode="bilinear", align_corners=False)
    back = F.interpolate(small, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    blurred = gaussian_blur(back, float(torch.empty(1).uniform_(0.0, 0.8).item()))
    return (blurred + torch.randn_like(blurred) * 0.01).clamp(-1, 1)


def surrogate_embed(model: Any, crop: torch.Tensor) -> torch.Tensor:
    """Return FaceNet's embedding of a crop scaled to [-1, 1]."""
    return F.normalize(model(crop), dim=1)


def cloak_crop(model: Any, crop: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Return the perturbation, in [-1, 1] units, that maximises displacement."""
    with torch.no_grad():
        anchor = surrogate_embed(model, crop)
    budget = epsilon * 2.0
    step = budget / 10.0
    delta = torch.zeros_like(crop, requires_grad=True)
    for _ in range(STEPS):
        probe = eot((crop + delta).clamp(-1, 1))
        loss = -(1.0 - F.cosine_similarity(surrogate_embed(model, probe), anchor)).mean()
        (gradient,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta -= step * gradient.sign()
            delta.clamp_(-budget, budget)
            delta.copy_((crop + delta).clamp(-1, 1) - crop)
    return delta.detach()


class Pipeline:
    """The project's detector, aligner and embedders, as the analysis would use them."""

    def __init__(self) -> None:
        """Build the real face backends from the default configuration."""
        config = load_config()
        self.detector = build_detector(config.face.detector)
        self.aligner = build_aligner(config.face.aligner)
        self.embedders = {
            "arcface": build_embedder(config.face.embedder),
            "sface": build_embedder(
                FaceEmbedderConfig(backend="opencv_sface", model_name="sface", flip_tta=False)
            ),
        }
        self.thresholds = config.thresholds.face_similarity

    def embeddings(self, image: np.ndarray) -> dict[str, np.ndarray] | None:
        """Detect, align and embed the most confident face with every embedder."""
        faces = self.detector.detect(image)
        if not faces:
            return None
        aligned = self.aligner.align(image, faces[0]).image
        return {name: emb.embed(aligned).vector for name, emb in self.embedders.items()}

    def face_matrix(self, cv2: Any, image: np.ndarray) -> np.ndarray | None:
        """Return the similarity transform onto the 160-pixel canonical frame."""
        faces = self.detector.detect(image)
        if not faces or faces[0].landmarks is None:
            return None
        matrix, _ = cv2.estimateAffinePartial2D(
            np.asarray(faces[0].landmarks, dtype=np.float32),
            ARCFACE_TEMPLATE_112 * (SIZE / 112.0),
            method=cv2.LMEDS,
        )
        return None if matrix is None else np.asarray(matrix)


def main(argv: list[str] | None = None) -> int:
    """Cloak every image at each budget and measure the displacement three ways."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--epsilon", type=float, nargs="+", default=[0.03, 0.06])
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no images under {args.images}")

    import cv2

    device = pick_device()
    surrogate = load_surrogate(device)
    pipeline = Pipeline()
    after = {
        "jpeg_70": Transformation("jpeg_compression", "jpeg_compression", {"quality": 70}),
        "resize_50": Transformation("resize", "resize", {"scale": 0.5}),
    }

    rows: dict[float, list[dict[str, Any]]] = {epsilon: [] for epsilon in args.epsilon}
    baseline: list[dict[str, float]] = []
    for path in paths:
        image = load_image(path)
        matrix = pipeline.face_matrix(cv2, image)
        clean = pipeline.embeddings(image)
        if matrix is None or clean is None:
            continue
        for name, transform in after.items():
            plain = pipeline.embeddings(transform.apply(image, seed=1))
            if plain is not None:
                baseline.append(
                    {
                        f"{name}_{model}": cosine_distance(clean[model], plain[model])
                        for model in clean
                    }
                )
        crop = cv2.warpAffine(image, matrix, (SIZE, SIZE))
        tensor = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)
        tensor = tensor.float().div(127.5).sub(1).unsqueeze(0).to(device)
        with torch.no_grad():
            clean_surrogate = surrogate_embed(surrogate, tensor).cpu().numpy()

        for epsilon in args.epsilon:
            delta = cloak_crop(surrogate, tensor, epsilon)
            shift = delta.squeeze(0).permute(1, 2, 0).cpu().numpy() * 127.5
            spread = cv2.warpAffine(
                shift, cv2.invertAffineTransform(matrix), (image.shape[1], image.shape[0])
            )
            cloaked = np.clip(image.astype(np.float32) + spread, 0, 255).astype(np.uint8)

            row: dict[str, Any] = {
                "image": path.name,
                "psnr": round(psnr(image, cloaked), 2),
                "ssim": round(ssim(image, cloaked), 4),
            }
            recovered = pipeline.face_matrix(cv2, cloaked)
            if recovered is not None:
                recrop = cv2.warpAffine(cloaked, recovered, (SIZE, SIZE))
                retensor = torch.from_numpy(np.ascontiguousarray(recrop)).permute(2, 0, 1)
                retensor = retensor.float().div(127.5).sub(1).unsqueeze(0).to(device)
                with torch.no_grad():
                    moved = surrogate_embed(surrogate, retensor).cpu().numpy()
                row["whitebox_facenet"] = cosine_distance(clean_surrogate, moved)
            embedded = pipeline.embeddings(cloaked)
            if embedded is not None:
                for model in clean:
                    row[f"transfer_{model}"] = cosine_distance(clean[model], embedded[model])
                    row[f"similarity_{model}"] = 1.0 - row[f"transfer_{model}"]
            for name, transform in after.items():
                processed = pipeline.embeddings(transform.apply(cloaked, seed=1))
                if processed is not None:
                    for model in clean:
                        row[f"{name}_{model}"] = cosine_distance(clean[model], processed[model])
            rows[epsilon].append(row)
        print(f"{path.name}", flush=True)

    def mean(values: list[dict[str, Any]], key: str) -> float | None:
        picked = [float(v[key]) for v in values if key in v]
        return round(float(np.mean(picked)), 4) if picked else None

    high = pipeline.thresholds.high_confidence_threshold
    report: dict[str, Any] = {
        "surrogate": "facenet-pytorch InceptionResnetV1 (VGGFace2)",
        "attack": f"PGD {STEPS} steps with expectation over resampling, blur and noise",
        "images": len(next(iter(rows.values()))) if rows else 0,
        "high_confidence_threshold": high,
        "baseline_no_cloak": {
            key: mean(baseline, key)
            for key in sorted({k for row in baseline for k in row})
        },
        "budgets": {},
    }
    for epsilon, values in rows.items():
        keys = sorted({k for row in values for k in row if k != "image"})
        summary = {key: mean(values, key) for key in keys}
        for model in ("arcface", "sface"):
            sims = [float(v[f"similarity_{model}"]) for v in values if f"similarity_{model}" in v]
            summary[f"still_high_confidence_{model}"] = (
                round(float(np.mean([s >= high for s in sims])), 4) if sims else None
            )
        report["budgets"][f"{epsilon:g}"] = summary

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "cloaking_whitebox.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
