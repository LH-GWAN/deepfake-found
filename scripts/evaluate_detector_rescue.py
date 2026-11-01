"""Count face detections with and without the rescue passes, per degradation.

The rescue passes in :class:`deepshield.face.detector.FaceDetector` exist for
two measured failures: a 250-pixel photograph downscaled to a quarter, and a
face cropped until it fills the frame. This script is the before-and-after
for those cases, kept so the "before" number has a file behind it rather than
a sentence.

Usage:
    python scripts/evaluate_detector_rescue.py data/test/eval_faces
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepshield.config import load_config
from deepshield.face.detector import build_detector
from deepshield.media import IMAGE_SUFFIXES, load_image
from deepshield.transforms import Transformation

DEGRADATIONS = {
    "clean": ("identity", {}),
    "downscale_25": ("downscale", {"scale": 0.25}),
    "downscale_50": ("downscale", {"scale": 0.5}),
    "crop_30": ("crop", {"ratio": 0.3}),
}


def main(argv: list[str] | None = None) -> int:
    """Detect every image under every degradation, rescue off and on."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/results/detector_rescue"))
    args = parser.parse_args(argv)

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise SystemExit(f"no images under {args.images}")
    config = load_config()
    detectors = {
        "without_rescue": build_detector(
            config.face.detector.model_copy(update={"rescue_enabled": False})
        ),
        "with_rescue": build_detector(config.face.detector),
    }

    counts: dict[str, dict[str, int]] = {}
    shapes: dict[str, list[int]] = {}
    for name, (kind, params) in DEGRADATIONS.items():
        counts[name] = {label: 0 for label in detectors}
        for path in paths:
            probe = Transformation(kind, kind, params).apply(load_image(path), seed=1)
            shapes.setdefault(name, list(probe.shape[:2]))
            for label, detector in detectors.items():
                counts[name][label] += int(bool(detector.detect(probe)))
        print(name, counts[name], flush=True)

    report = {
        "images": len(paths),
        "detector": config.face.detector.backend,
        "rescue": {
            "min_side": config.face.detector.rescue_min_side,
            "pad_fraction": config.face.detector.rescue_pad_fraction,
        },
        "probe_shape": shapes,
        "detected": counts,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "detection_counts.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
