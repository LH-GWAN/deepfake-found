r"""Measure what an attacker who is trying to remove the watermark can do.

Every other watermark number in this repository assumes an attacker who does
not know the mark is there: they re-encode, crop, swap or regenerate for their
own reasons and the question is what survives. This script drops that
assumption and asks how much it costs, in image quality, to strip the mark on
purpose.

Two attacker strengths are measured, because they answer different questions.

uninformed
    Knows a watermark may exist, not how it works. Applies degradations chosen
    to destroy fine structure: heavy JPEG, blur, median filtering, downscaling,
    noise, sharpening, and regeneration through Stable Diffusion's VAE. Each is
    applied to the marked photograph and to the unmarked one, so the detection
    number is an AUC between the two under the same attack, and the attribution
    number is top-1 over the codebook.
informed
    Has the decoder. For the learned watermark this is a white-box projected
    gradient attack in the aligned face frame under an L-infinity budget of a
    few grey levels, warped back into the photograph exactly as the mark was.
    Two objectives: ``erase`` drives every logit towards zero so no codeword
    correlates; ``frame`` drives the logits towards another registrant's code.
    The second is the one to worry about, because a successful frame does not
    remove evidence - it manufactures it against an innocent person. The
    attack is computed as an expectation over the training distortion layer,
    because a perturbation crafted for one exact alignment is undone by the
    re-detection and re-alignment the decoder performs, and an attacker who
    has the decoder has the training code too.

The DCT watermark has no secret: its coefficient pair and tile layout are in
the source. An informed attacker can therefore equalise the two coefficients in
every block (removal) or embed a different code over the top (forgery) at a
cost measured here in PSNR. That is a property of any keyless scheme, and the
right conclusion is that a deployment needs a keyed layout, not a stronger mark.

Usage:
    python scripts/evaluate_watermark_removal.py \\
        --checkpoint models/learned_watermark_v2.pt models/learned_watermark_v3.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from deepshield.config import WatermarkConfig
from deepshield.media import load_image
from deepshield.protection.fingerprint import dct2, idct2
from deepshield.protection.watermark import (
    BLOCK_SIZE,
    CARRIER_PAIRS,
    CODE_BITS,
    DctWatermarker,
    derive_layout,
)
from deepshield.quality import psnr
from deepshield.transforms import Transformation
from deepshield.types import WatermarkPayload

Pair = tuple[tuple[int, int], tuple[int, int]]

PGD_STEPS = 40
PGD_EOT_STRENGTH = 0.5


def load_learned_stack() -> None:
    """Import the torch and insightface halves of this script.

    They are loaded on demand rather than at module scope so that ``--dct-only``
    can run on a machine with neither. The DCT half needs nothing but numpy and
    Pillow, and it is the half a keyed layout is measured on.
    """
    global torch, F, swaps, SIZE, align_matrix, as_tensor
    global SUBSETS, Model, closed_set_top1, identity_pairs, rank_auc, score, summarise
    global BITS, distort, pick_device, load_vae, vae_roundtrip

    import build_manipulation_set as swaps
    import torch
    import torch.nn.functional as F
    from evaluate_learned_watermark import SIZE, align_matrix, as_tensor
    from evaluate_learned_watermark_attribution import (
        SUBSETS,
        Model,
        closed_set_top1,
        identity_pairs,
        rank_auc,
        score,
        summarise,
    )
    from learned_watermark import BITS, distort, pick_device
    from train_learned_watermark import load_vae, vae_roundtrip


def dct_paths(faces: Path, limit: int) -> list[Path]:
    """Return the photographs ``identity_pairs`` would hand the DCT measurement.

    It reproduces that selection rather than calling it because ``identity_pairs``
    lives beside the learned-watermark evaluation, and importing that module pulls
    in torch - which is exactly what ``--dct-only`` exists to avoid.
    """
    by_identity: dict[str, list[Path]] = {}
    for path in sorted(faces.rglob("*.jpg")):
        identity = path.parent.name if path.parent != faces else path.stem.rsplit("_", 1)[0]
        by_identity.setdefault(identity, []).append(path)
    names = sorted(by_identity)
    return [by_identity[name][0] for name in names[0 : 2 * limit : 2]]


def uninformed_attacks() -> dict[str, Any]:
    """Return the degradations an attacker without the decoder would try."""

    def transform(kind: str, **params: Any) -> Any:
        def apply(image: np.ndarray) -> np.ndarray:
            return Transformation(kind, kind, params).apply(image, seed=1)

        return apply

    def median(image: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(image).filter(ImageFilter.MedianFilter(3)))

    def sharpen(image: np.ndarray) -> np.ndarray:
        filtered = Image.fromarray(image).filter(
            ImageFilter.UnsharpMask(radius=2, percent=150, threshold=0)
        )
        return np.asarray(filtered)

    def resize_down_up(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        small = Image.fromarray(image).resize(
            (width // 2, height // 2), Image.Resampling.LANCZOS
        )
        return np.asarray(small.resize((width, height), Image.Resampling.BICUBIC))

    def combined(image: np.ndarray) -> np.ndarray:
        out = transform("blur", sigma=1.0)(image)
        out = transform("noise", sigma=5.0)(out)
        return transform("jpeg_compression", quality=50)(out)

    return {
        "none": lambda image: image,
        "jpeg_50": transform("jpeg_compression", quality=50),
        "jpeg_30": transform("jpeg_compression", quality=30),
        "blur_1.5": transform("blur", sigma=1.5),
        "median_3": median,
        "resize_50_up": resize_down_up,
        "noise_8": transform("noise", sigma=8.0),
        "sharpen": sharpen,
        "blur_noise_jpeg": combined,
    }


class VaeAttack:
    """Regenerate the photograph through the Stable Diffusion VAE."""

    def __init__(self, device: str) -> None:
        """Load the VAE lazily on first use."""
        self.device = device
        self._vae: Any = None

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Return the photograph after one encode-decode round trip."""
        if self._vae is None:
            self._vae = load_vae(self.device)
        height, width = image.shape[:2]
        side = (max(height, width) + 7) // 8 * 8
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        canvas[:height, :width] = image
        tensor = torch.from_numpy(canvas).permute(2, 0, 1).float().div(127.5).sub(1)
        with torch.no_grad():
            out = vae_roundtrip(self._vae, tensor.unsqueeze(0).to(self.device))
        array = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        array = np.clip((array + 1) * 127.5, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array[:height, :width])


def pgd_in_crop(
    model: Model,
    crop: np.ndarray,
    levels: float,
    target: np.ndarray | None,
) -> np.ndarray:
    """Return an additive perturbation of the aligned crop, in 8-bit units.

    ``target`` of ``None`` erases (logits towards zero); otherwise the logits
    are pushed towards the given code.
    """
    clean = as_tensor(crop, model.device, False)
    epsilon = float(levels) * 2.0 / 255.0
    step = epsilon / 6.0
    delta = torch.zeros_like(clean, requires_grad=True)
    wanted = (
        None
        if target is None
        else torch.from_numpy(target[None].astype(np.float32)).to(model.device)
    )
    for _ in range(PGD_STEPS):
        probe = distort((clean + delta).clamp(-1, 1), PGD_EOT_STRENGTH)
        if model.transposed:
            probe = probe.transpose(2, 3).contiguous()
        logits = model.decoder(probe)
        loss = (logits**2).mean() if wanted is None else F.binary_cross_entropy_with_logits(
            logits, wanted
        )
        (gradient,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta -= step * gradient.sign()
            delta.clamp_(-epsilon, epsilon)
            delta.copy_((clean + delta).clamp(-1, 1) - clean)
    return delta.detach().squeeze(0).permute(1, 2, 0).cpu().numpy() * 127.5


def warp_back(cv2: Any, image: np.ndarray, matrix: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Add a crop-frame perturbation to the photograph it came from."""
    spread = cv2.warpAffine(
        shift, cv2.invertAffineTransform(matrix), (image.shape[1], image.shape[0])
    )
    return np.clip(image.astype(np.float32) + spread, 0, 255).astype(np.uint8)


def dct_equalise(image: np.ndarray, pairs: Sequence[Pair]) -> np.ndarray:
    """Remove the DCT mark by giving each named pair of coefficients their mean.

    The pairs are a parameter because that is the whole question a key asks. An
    attacker who guesses the wrong carrier equalises coefficients the mark was
    never in and pays the quality cost for nothing; one who refuses to guess has
    to flatten the whole published family in a single pass, which is the attack
    a coefficient key this short actually has to answer.
    """
    ycbcr = np.asarray(Image.fromarray(image).convert("YCbCr"), dtype=np.float64)
    luminance = ycbcr[:, :, 0]
    rows, cols = luminance.shape[0] // BLOCK_SIZE, luminance.shape[1] // BLOCK_SIZE
    for row in range(rows):
        for col in range(cols):
            y0, x0 = row * BLOCK_SIZE, col * BLOCK_SIZE
            block = dct2(luminance[y0 : y0 + BLOCK_SIZE, x0 : x0 + BLOCK_SIZE])
            for first, second in pairs:
                mean = (block[first] + block[second]) / 2.0
                block[first] = mean
                block[second] = mean
            luminance[y0 : y0 + BLOCK_SIZE, x0 : x0 + BLOCK_SIZE] = idct2(block)
    ycbcr[:, :, 0] = np.clip(luminance, 0, 255)
    return np.asarray(Image.fromarray(ycbcr.astype(np.uint8), mode="YCbCr").convert("RGB"))


def evaluate_dct(paths: list[Path], owner_key: str | None, forger_key: str) -> dict[str, Any]:
    """Measure informed removal and forgery against the DCT mark, with and without the key.

    Both attacks are run twice: once by an attacker holding the owner's key and
    once by one holding a different key. The first is the control. Without it a
    key that quietly broke the attack would look exactly like a key that
    defeated it, and the numbers would say nothing.

    Removal is run a third time against the whole published family of carrier
    pairs at once. Three pairs is about a bit and a half, so guessing is not the
    attack a coefficient key has to survive - sweeping is, and its cost belongs
    in the same table as the guesses.
    """
    watermarker = DctWatermarker(WatermarkConfig(key=owner_key))
    attacker = DctWatermarker(WatermarkConfig(key=forger_key))
    right, wrong = derive_layout(owner_key), derive_layout(forger_key)
    right_pair = (right.coefficient_a, right.coefficient_b)
    wrong_pair = (wrong.coefficient_a, wrong.coefficient_b)
    owner = WatermarkPayload(version=1, user_token="owner", asset_id="photo", distribution_id="a")
    forger = WatermarkPayload(version=1, user_token="forger", asset_id="photo", distribution_id="b")
    owner_code = f"{owner.code(CODE_BITS):08x}"
    forger_code = f"{forger.code(CODE_BITS):08x}"
    outcomes: dict[str, dict[str, list[float]]] = {
        name: {"owner": [], "forger": [], "detected": [], "psnr": []}
        for name in (
            "none",
            "equalise_right_key",
            "overwrite_right_key",
            "equalise_wrong_key",
            "overwrite_wrong_key",
            "equalise_family",
        )
    }
    for path in paths:
        image = load_image(path)
        try:
            marked = watermarker.embed(image, owner)
        except Exception:
            continue
        variants = {
            "none": marked,
            "equalise_right_key": dct_equalise(marked, [right_pair]),
            "overwrite_right_key": watermarker.embed(marked, forger),
            "equalise_wrong_key": dct_equalise(marked, [wrong_pair]),
            "overwrite_wrong_key": attacker.embed(marked, forger),
            "equalise_family": dct_equalise(marked, CARRIER_PAIRS),
        }
        for name, variant in variants.items():
            result = watermarker.detect(variant)
            outcomes[name]["detected"].append(float(result.detected))
            outcomes[name]["owner"].append(float(result.watermark_code == owner_code))
            outcomes[name]["forger"].append(float(result.watermark_code == forger_code))
            outcomes[name]["psnr"].append(psnr(marked, variant))
    return {
        name: {
            "n": len(values["detected"]),
            "detected": round(float(np.mean(values["detected"])), 4),
            "owner_code_read": round(float(np.mean(values["owner"])), 4),
            "forger_code_read": round(float(np.mean(values["forger"])), 4),
            "psnr_vs_marked_db": (
                None
                if not np.isfinite(np.mean(values["psnr"]))
                else round(float(np.mean(values["psnr"])), 2)
            ),
        }
        for name, values in outcomes.items()
        if values["detected"]
    }


def evaluate_learned(
    cv2: Any,
    app: Any,
    model: Model,
    pairs: list[tuple[Path, Path]],
    codebook: np.ndarray,
    limit: int,
    seed: int,
    levels: list[float],
    device: str,
) -> dict[str, Any]:
    """Run every attack on one checkpoint and summarise each."""
    attacks = uninformed_attacks()
    attacks["vae_roundtrip"] = VaeAttack(device)
    informed = [("pgd_erase", level, None) for level in levels]
    informed += [("pgd_frame", level, "frame") for level in levels]
    cells: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {"marked": [], "unmarked": []} for name in attacks
    }
    for name, level, _ in informed:
        cells[f"{name}_{level:g}"] = {"marked": [], "unmarked": []}
    quality: dict[str, list[float]] = {name: [] for name in cells}
    framed: dict[str, list[float]] = {}
    samples = 0

    def record(
        name: str, label: str, image: np.ndarray, truth: int, index: int
    ) -> np.ndarray | None:
        logits = model.logits(cv2, app, image)
        if logits is None:
            return None
        scores = score(logits, codebook)
        subset_rng = np.random.default_rng(seed * 1000 + index)
        cells[name][label].append(
            {
                "detection_score": float(scores.max()),
                "top1": float(int(np.argmax(scores)) == truth),
                "closed_set": {
                    str(size): closed_set_top1(scores, truth, size, subset_rng) for size in SUBSETS
                },
            }
        )
        return scores

    for index, (own_path, _) in enumerate(pairs):
        if samples >= limit:
            break
        own = load_image(own_path)
        truth = index % codebook.shape[0]
        prepared = model.mark(cv2, app, own, codebook[truth])
        if prepared is None:
            continue
        marked, _ = prepared
        samples += 1

        for name, attack in attacks.items():
            for label, probe in (("marked", marked), ("unmarked", own)):
                attacked = attack(probe)
                if label == "marked":
                    quality[name].append(psnr(marked, attacked))
                record(name, label, attacked, truth, index)

        matrix = align_matrix(cv2, app, marked)
        if matrix is None:
            continue
        crop = cv2.warpAffine(marked, matrix, (SIZE, SIZE))
        clean_matrix = align_matrix(cv2, app, own)
        clean_crop = (
            None if clean_matrix is None else cv2.warpAffine(own, clean_matrix, (SIZE, SIZE))
        )
        for name, level, objective in informed:
            key = f"{name}_{level:g}"
            target = None
            if objective == "frame":
                target = codebook[(truth + 1) % codebook.shape[0]]
            attacked = warp_back(cv2, marked, matrix, pgd_in_crop(model, crop, level, target))
            quality[key].append(psnr(marked, attacked))
            scores = record(key, "marked", attacked, truth, index)
            if objective == "frame" and scores is not None:
                hit = int(np.argmax(scores)) == (truth + 1) % codebook.shape[0]
                framed.setdefault(key, []).append(float(hit))
            if clean_crop is not None:
                control = warp_back(
                    cv2, own, clean_matrix, pgd_in_crop(model, clean_crop, level, target)
                )
                record(key, "unmarked", control, truth, index)
        if samples % 20 == 0:
            print(f"  {samples} samples", flush=True)

    report: dict[str, Any] = {}
    for name, cell in cells.items():
        if not cell["marked"] or not cell["unmarked"]:
            continue
        summary = summarise(cell["marked"], cell["unmarked"], seed)
        marked_scores = np.array([row["detection_score"] for row in cell["marked"]])
        unmarked_scores = np.array([row["detection_score"] for row in cell["unmarked"]])
        report[name] = {
            "n": len(cell["marked"]),
            "psnr_vs_marked_db": (
                None
                if not np.isfinite(np.mean(quality[name]))
                else round(float(np.mean(quality[name])), 2)
            ),
            "detection_auc": round(rank_auc(marked_scores, unmarked_scores), 4),
            "attribution_top1_of_codebook": summary["attribution_all"]["top1_of_codebook"],
            "marked_score_mean": round(float(marked_scores.mean()), 2),
            "unmarked_score_mean": round(float(unmarked_scores.mean()), 2),
        }
    for key, hits in framed.items():
        if key in report:
            report[key]["attributed_to_framed_registrant"] = round(float(np.mean(hits)), 4)
    return report


def main(argv: list[str] | None = None) -> int:
    """Measure removal attacks against the learned and the DCT watermarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", default=[])
    parser.add_argument(
        "--dct-only",
        action="store_true",
        help="measure only the DCT mark, which needs neither torch nor a GPU",
    )
    parser.add_argument("--owner-key", default=None)
    parser.add_argument("--forger-key", default="forger-key")
    parser.add_argument("--faces", type=Path, default=Path("data/sklearn/lfw_home/lfw_funneled"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    parser.add_argument("--pairs", type=int, default=60)
    parser.add_argument("--codebook", type=int, default=72)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--levels", type=float, nargs="+", default=[4.0, 8.0, 16.0])
    parser.add_argument("--orientation", choices=("transposed", "upright"), default=None)
    args = parser.parse_args(argv)
    if not args.dct_only and not args.checkpoint:
        parser.error("--checkpoint is required unless --dct-only is given")

    if args.dct_only:
        paths = dct_paths(args.faces, args.pairs)
        if not paths:
            raise SystemExit(f"no photographs found in {args.faces}")
        report = {
            "question": "does a key make informed removal and forgery need the key",
            "faces": str(args.faces),
            "pairs": len(paths),
            "dct": {
                "keyless": evaluate_dct(paths, None, args.forger_key),
                "keyed": evaluate_dct(paths, args.owner_key or "owner-key", args.forger_key),
            },
        }
        args.output.mkdir(parents=True, exist_ok=True)
        destination = args.output / "watermark_removal_dct.json"
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"\nwrote {destination}")
        return 0

    load_learned_stack()
    pairs = identity_pairs(args.faces)
    if not pairs:
        raise SystemExit(f"no identity pairs found in {args.faces}")

    import cv2

    device = pick_device()
    app = swaps.build_analyzer(args.models)
    rng = np.random.default_rng(args.seed)
    codebook = rng.integers(0, 2, (args.codebook, BITS)).astype(np.uint8)

    report: dict[str, Any] = {
        "question": "how much image quality does it cost to strip or forge the mark on purpose",
        "faces": str(args.faces),
        "pairs": args.pairs,
        "codebook": args.codebook,
        "pgd": {
            "steps": PGD_STEPS,
            "budget_grey_levels": args.levels,
            "domain": "aligned 128px face frame, warped back",
            "expectation_over_distortion_strength": PGD_EOT_STRENGTH,
        },
        "learned": {},
    }
    for checkpoint in args.checkpoint:
        print(f"== {checkpoint}", flush=True)
        model = Model(checkpoint, device, args.orientation)
        report["learned"][checkpoint.stem] = evaluate_learned(
            cv2, app, model, pairs, codebook, args.pairs, args.seed, args.levels, device
        )
    print("== dct", flush=True)
    report["dct"] = evaluate_dct(
        [own for own, _ in pairs[: args.pairs]], args.owner_key, args.forger_key
    )

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "watermark_removal.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
