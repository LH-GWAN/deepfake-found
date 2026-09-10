"""Measure what a key costs the rotation search.

A key permutes which tile slot carries which message bit, and a permuted
message is not the same picture. Neighbouring blocks stop sharing a sign, so
the mark's own spatial pattern gets finer, and resampling through a rotation
attenuates fine patterns first. There is a reason to expect a key to cost the
angle search something, which is why it is measured rather than assumed.

The cost is small, key-dependent, and it only appears at the larger size. On
twenty LFW photographs at 250 pixels every key recovers five and ten degrees
exactly as the keyless layout does. At 512 the median key still recovers both,
while the worst of eight gives up one photograph in twenty at five degrees and
two at ten. That is the spread this reports rather than one number, because a
key is drawn and not chosen.

The carrier pair is not what does this: held at the identity permutation, every
adopted pair recovers rotation as well as the published one
(``evaluate_watermark_carriers.py``).

This script exists because a keyed unit test failed five degrees on the
synthetic texture ``tests/conftest.py`` builds, and the question was whether
that was the design or the fixture. It is mostly the fixture - that image loses
five keys in twenty-four at 250 pixels and three at 512, far thinner than any
photograph here - but the 512 row above is the part that is real, and it is
written down rather than tuned away.

Usage:
    python scripts/evaluate_watermark_key_rotation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepshield.config import WatermarkConfig
from deepshield.media import load_image
from deepshield.protection.watermark import CODE_BITS, DctWatermarker
from deepshield.transforms import Transformation
from deepshield.types import WatermarkPayload


def recovered(marker: DctWatermarker, images: list[np.ndarray], degrees: float) -> float:
    """Return the share of images whose code still reads back after a rotation."""
    owner = WatermarkPayload(version=1, user_token="owner", asset_id="photo", distribution_id="a")
    code = f"{owner.code(CODE_BITS):08x}"
    hits = []
    for image in images:
        marked = marker.embed(image, owner)
        probe = Transformation("rotation", "rotation", {"degrees": degrees}).apply(marked, seed=1)
        hits.append(float(marker.detect(probe).watermark_code == code))
    return float(np.mean(hits))


def main(argv: list[str] | None = None) -> int:
    """Compare keyless and keyed rotation recovery at two image sizes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("data/sklearn/lfw_home/lfw_funneled"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--keys", type=int, default=8)
    parser.add_argument("--sizes", type=int, nargs="+", default=[250, 512])
    parser.add_argument("--degrees", type=float, nargs="+", default=[5.0, 10.0])
    parser.add_argument("--strength", type=float, default=0.16)
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)

    paths = sorted(args.images.rglob("*.jpg"))[: args.limit]
    if not paths:
        raise SystemExit(f"no photographs under {args.images}")
    originals = [load_image(path) for path in paths]
    keys = [f"key-{index}" for index in range(args.keys)]

    report: dict[str, Any] = {
        "images": len(paths),
        "strength": args.strength,
        "keys": keys,
        "sizes": {},
    }
    for size in args.sizes:
        images = [
            np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.LANCZOS))
            for image in originals
        ]
        rows: dict[str, dict[str, float]] = {}
        for name, key in [("keyless", None), *[(key, key) for key in keys]]:
            marker = DctWatermarker(WatermarkConfig(strength=args.strength, key=key))
            rows[name] = {
                f"{degrees:g}": round(recovered(marker, images, degrees), 3)
                for degrees in args.degrees
            }
            read = " ".join(f"{d}deg={v:.2f}" for d, v in rows[name].items())
            print(f"{size:4} {name:10} {read}", flush=True)
        keyed = [rows[key] for key in keys]
        report["sizes"][str(size)] = {
            "keyless": rows["keyless"],
            "keyed": {key: rows[key] for key in keys},
            "keyed_median": {
                f"{degrees:g}": round(float(np.median([row[f"{degrees:g}"] for row in keyed])), 3)
                for degrees in args.degrees
            },
            "keyed_worst": {
                f"{degrees:g}": round(float(np.min([row[f"{degrees:g}"] for row in keyed])), 3)
                for degrees in args.degrees
            },
        }

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "watermark_key_rotation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
