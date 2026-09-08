"""Count watermark detections on images that carry no watermark.

Every search stage the decoder gains - crop grids, tile phases, and now
rotation angles - is another chance for a checksum to pass on noise, and a
detector that names a distribution channel for an unmarked photograph is worse
than one that reports nothing. This is the measurement that has to be repeated
whenever the search space grows, and it is kept as a script so that it is.

Each image is tested as it is and after a rotation, because the rotation search
only runs on images the crop search could not decode, which is exactly the
case an unmarked image presents.

Usage:
    python scripts/evaluate_watermark_false_positives.py data/test/eval_faces
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepshield.config import load_config
from deepshield.media import IMAGE_SUFFIXES, load_image
from deepshield.protection.watermark import build_watermarker
from deepshield.transforms import Transformation


def main(argv: list[str] | None = None) -> int:
    """Run the detector over unmarked images and report every positive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("--rotations", type=float, nargs="+", default=[0.0, 5.0])
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"no images under {args.images}")
    config = load_config()
    watermarker = build_watermarker(config.protection.watermark)

    positives: list[dict[str, object]] = []
    seconds: list[float] = []
    for path in paths:
        image = load_image(path)
        for degrees in args.rotations:
            probe = image
            if degrees:
                probe = Transformation("rotation", "rotation", {"degrees": degrees}).apply(
                    image, seed=1
                )
            started = time.perf_counter()
            result = watermarker.detect(probe)
            seconds.append(time.perf_counter() - started)
            if result.detected:
                positives.append(
                    {"image": path.name, "rotation": degrees, "code": result.watermark_code}
                )

    report = {
        "images": len(paths),
        "rotations": args.rotations,
        "probes": len(paths) * len(args.rotations),
        "false_positives": len(positives),
        "positives": positives,
        "mean_seconds_per_probe": round(sum(seconds) / len(seconds), 3),
        "search": {
            "resync_enabled": config.protection.watermark.resync_enabled,
            "rotation_enabled": config.protection.watermark.resync_rotation_enabled,
            "rotation_max_degrees": config.protection.watermark.resync_rotation_max_degrees,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "watermark_false_positives.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
