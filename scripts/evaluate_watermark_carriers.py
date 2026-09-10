"""Screen candidate carrier coefficient pairs for the keyed DCT watermark.

A key chooses the two coefficients the mark rides on, so the list it chooses
from is part of the design and not an implementation detail. Widening it buys
key space; widening it carelessly buys key space by handing some owners a mark
that dies to a JPEG. This script is the measurement that decides membership,
and it is kept so that the list can be widened later without guessing.

Only mid-band transposes (u,v)/(v,u) are candidates. The two coefficients have
to share a radial frequency for their difference to survive quantisation: the
blind decoder reads a sign, not a magnitude, and a pair that JPEG quantises
unevenly loses that sign before anything else happens.

A pair is adopted only if it matches or beats the published (3,4) pair on every
attack measured here, so that choosing a key can never cost robustness.

Usage:
    python scripts/evaluate_watermark_carriers.py
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

from deepshield.config import WatermarkConfig
from deepshield.media import load_image
from deepshield.protection.watermark import (
    CARRIER_PAIRS,
    CODE_BITS,
    MESSAGE_BITS,
    DctWatermarker,
    KeyedLayout,
)
from deepshield.quality import psnr, ssim
from deepshield.transforms import Transformation
from deepshield.types import WatermarkPayload

Pair = tuple[tuple[int, int], tuple[int, int]]

CANDIDATES: tuple[Pair, ...] = (
    ((3, 4), (4, 3)),
    ((2, 3), (3, 2)),
    ((1, 4), (4, 1)),
    ((2, 4), (4, 2)),
    ((1, 5), (5, 1)),
    ((2, 5), (5, 2)),
    ((1, 6), (6, 1)),
    ((3, 5), (5, 3)),
    ((2, 6), (6, 2)),
    ((4, 5), (5, 4)),
    ((3, 6), (6, 3)),
)

ATTACKS: dict[str, dict[str, Any] | None] = {
    "clean": None,
    "jpeg_q70": {"quality": 70},
    "resize_50": {"scale": 0.5},
    "crop_20": {"ratio": 0.2},
    "rotate_5": {"degrees": 5.0},
}
KINDS = {
    "jpeg_q70": "jpeg_compression",
    "resize_50": "resize",
    "crop_20": "crop",
    "rotate_5": "rotation",
}


def screening_layout(pair: Pair) -> KeyedLayout:
    """Return a layout that puts the mark on ``pair`` with the tile left in order.

    Screening reaches past the key on purpose: a candidate that is not in
    ``CARRIER_PAIRS`` yet cannot be selected by any key, which is exactly why it
    is being screened. The permutation is held at identity so that the only
    thing varying across rows is the carrier.
    """
    identity = np.arange(MESSAGE_BITS, dtype=np.intp)
    identity.flags.writeable = False
    return KeyedLayout(pair[0], pair[1], identity, identity)


def main(argv: list[str] | None = None) -> int:
    """Measure quality and post-attack recovery for every candidate pair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/sklearn/lfw_home/lfw_funneled"))
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--strength", type=float, default=0.16)
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)

    paths = sorted(args.images.rglob("*.jpg"))[: args.limit]
    if not paths:
        raise SystemExit(f"no photographs under {args.images}")
    owner = WatermarkPayload(version=1, user_token="owner", asset_id="photo", distribution_id="a")
    code = f"{owner.code(CODE_BITS):08x}"

    rows: dict[str, dict[str, Any]] = {}
    header = f"{'carrier':10} {'PSNR':>6} {'SSIM':>6} " + " ".join(f"{a:>9}" for a in ATTACKS)
    print(header, flush=True)
    for pair in CANDIDATES:
        marker = DctWatermarker(WatermarkConfig(strength=args.strength))
        marker.layout = screening_layout(pair)
        quality: list[float] = []
        structure: list[float] = []
        hits: dict[str, list[float]] = {name: [] for name in ATTACKS}
        for path in paths:
            image = load_image(path)
            marked = marker.embed(image, owner)
            quality.append(psnr(image, marked))
            structure.append(ssim(image, marked))
            for name, params in ATTACKS.items():
                probe = marked
                if params is not None:
                    kind = KINDS[name]
                    probe = Transformation(kind, kind, params).apply(marked, seed=1)
                hits[name].append(float(marker.detect(probe).watermark_code == code))
        row = {
            "psnr_db": round(float(np.mean(quality)), 2),
            "ssim": round(float(np.mean(structure)), 3),
            "recovered": {name: round(float(np.mean(hits[name])), 3) for name in ATTACKS},
            "adopted": pair in CARRIER_PAIRS,
        }
        rows[str(pair[0])] = row
        print(
            f"{str(pair[0]):10} {row['psnr_db']:6.2f} {row['ssim']:6.3f} "
            + " ".join(f"{row['recovered'][name]:9.2f}" for name in ATTACKS),
            flush=True,
        )

    report = {"images": len(paths), "strength": args.strength, "carriers": rows}
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "watermark_carriers.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
