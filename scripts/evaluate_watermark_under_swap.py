"""Measure whether the watermark survives a face swap, separately per direction.

The protection story this project was built around is that a mark embedded in a
user's photograph should still be recoverable from a deepfake made with it. That
claim hides two different events, and they do not have the same answer.

target
    Someone swaps a different face onto the user's watermarked photograph. The
    swap repaints the facial hull and leaves everything else alone, so most of
    the carrier survives untouched.
source
    The user's watermarked photograph supplies the face, which is warped onto
    somebody else's picture. Only the pixels inside the hull travel to the
    output, and they arrive resampled triangle by triangle onto different facial
    geometry.

The distinction matters because the second direction is the one the product
promise rests on, and it is the one that destroys the mark: the decoder needs the
8x8 block grid, and a piecewise-affine warp does not preserve it. Reporting a
single averaged number over both directions would hide that.

Framing is swept as well, because the fraction of the frame a face occupies
decides how much untouched carrier the target direction keeps. The evaluation
faces are tight crops, so wider framings are simulated by insetting each crop in
a larger canvas.

Bit accuracy is reported next to the detection count because it separates a mark
that is degraded from one that is gone. Chance is 0.5.

``--swapper inswapper`` runs the same protocol through the GAN swapper
``inswapper_128`` instead of the graphics one. Its source input is an identity
embedding rather than pixels, so the source direction has no carrier by
construction; the target direction is the one worth measuring, because the GAN
repaints a 128-pixel aligned crop and pastes it back, which is a different
footprint from the graphics swap's landmark hull.

Usage:
    python scripts/evaluate_watermark_under_swap.py
    python scripts/evaluate_watermark_under_swap.py --pairs 40
    python scripts/evaluate_watermark_under_swap.py --swapper inswapper
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_manipulation_set as swaps
from evaluate_learned_watermark_attribution import Attacks

from deepshield.config import WatermarkConfig
from deepshield.protection.watermark import DctWatermarker
from deepshield.types import WatermarkPayload

FRAMINGS = (1.0, 0.5, 0.25)


def inset(cv2: Any, image: np.ndarray, fraction: float) -> np.ndarray:
    """Return the crop centred in a canvas where it covers ``fraction`` of the area."""
    if fraction >= 1.0:
        return image
    height, width = image.shape[:2]
    side = int(round(width / (fraction**0.5)))
    canvas = cv2.resize(image, (side, side), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.GaussianBlur(canvas, (0, 0), side / 60.0)
    offset = (side - width) // 2
    canvas[offset : offset + height, offset : offset + width] = image
    return np.asarray(canvas)


def hull_fraction(cv2: Any, app: Any, image: np.ndarray) -> float | None:
    """Return the share of the frame covered by the largest face's landmark hull."""
    face = swaps.largest_face(app, image)
    if face is None:
        return None
    landmarks = np.asarray(face.landmark_2d_106, dtype=np.float32)
    mask = swaps.hull_mask(cv2, image.shape[:2], landmarks)
    return float((mask > 0).sum()) / float(image.shape[0] * image.shape[1])


def identity_pairs(faces: Path, limit: int) -> list[tuple[Path, Path]]:
    """Pair each identity with the next one, so no pair shares a person."""
    by_identity: dict[str, list[Path]] = {}
    for path in sorted(faces.glob("*.jpg")) + sorted(faces.glob("*.png")):
        by_identity.setdefault(path.stem.rsplit("_", 1)[0], []).append(path)
    names = sorted(by_identity)
    pairs = [
        (by_identity[names[index]][0], by_identity[names[index + 1]][0])
        for index in range(len(names) - 1)
    ]
    return pairs[:limit]


def measure(
    cv2: Any,
    app: Any,
    watermarker: DctWatermarker,
    payload: WatermarkPayload,
    pairs: list[tuple[Path, Path]],
    fraction: float,
    swap: Any,
) -> dict[str, Any]:
    """Run both swap directions over every pair at one framing."""
    from deepshield.media import load_image

    hulls: list[float] = []
    stats: dict[str, dict[str, list[float]]] = {
        "target": {"detected": [], "bit_accuracy": []},
        "source": {"detected": [], "bit_accuracy": []},
    }
    for own_path, other_path in pairs:
        own = inset(cv2, load_image(own_path), fraction)
        other = inset(cv2, load_image(other_path), fraction)
        marked = watermarker.embed(own, payload)
        hull = hull_fraction(cv2, app, marked)
        if hull is not None:
            hulls.append(hull)
        outputs = {
            "target": swap(other, marked),
            "source": swap(marked, other),
        }
        for direction, output in outputs.items():
            if output is None:
                continue
            stats[direction]["detected"].append(
                float(watermarker.detect(output).detected)
            )
            stats[direction]["bit_accuracy"].append(
                watermarker.bit_accuracy(output, payload)
            )
    return {
        "framing": fraction,
        "hull_fraction": round(float(np.mean(hulls)), 4) if hulls else None,
        "directions": {
            direction: {
                "swaps": len(values["detected"]),
                "detected": int(sum(values["detected"])),
                "bit_accuracy": (
                    round(float(np.mean(values["bit_accuracy"])), 4)
                    if values["bit_accuracy"]
                    else None
                ),
            }
            for direction, values in stats.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Measure watermark recovery through face swaps in both directions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", type=Path, default=Path("data/test/manipulated/real"))
    parser.add_argument("--models", type=Path, default=Path("models/insightface"))
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    parser.add_argument("--pairs", type=int, default=24)
    parser.add_argument("--swapper", choices=("graphics", "inswapper"), default="graphics")
    parser.add_argument(
        "--inswapper", type=Path, default=Path("models/inswapper/inswapper_128.onnx")
    )
    args = parser.parse_args(argv)

    if not args.faces.is_dir():
        raise SystemExit(
            f"missing {args.faces}; run scripts/build_manipulation_set.py first"
        )
    pairs = identity_pairs(args.faces, args.pairs)
    if not pairs:
        raise SystemExit(f"no identity pairs found in {args.faces}")

    import cv2

    app = swaps.build_analyzer(args.models)
    watermarker = DctWatermarker(WatermarkConfig())
    payload = WatermarkPayload(version=1, user_token="evaluation", asset_id="swap")

    if args.swapper == "inswapper":
        attacks = Attacks(args.models.parent, args.inswapper, "cpu")

        def swap(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
            return attacks.inswap(source, target)

    else:

        def swap(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
            return swaps.swap_face(cv2, source, target, app)

    report = {
        "pairs": len(pairs),
        "swapper": args.swapper,
        "framings": [
            measure(cv2, app, watermarker, payload, pairs, fraction, swap)
            for fraction in FRAMINGS
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.swapper == "graphics" else f"_{args.swapper}"
    destination = args.output / f"watermark_under_swap{suffix}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
