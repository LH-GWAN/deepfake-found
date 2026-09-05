"""Cache landmark-aligned face crops for training the learned watermark.

Training needs faces in the frame the detector will hand the decoder at analysis
time, so the crop is produced the same way the pipeline produces one: five
landmarks mapped onto the canonical ArcFace template by a similarity transform.
Doing this once and caching the result keeps it out of the training loop, where
it would otherwise dominate the step time.

The corpus is LFW, which this project already has locally because scikit-learn
downloads it for the face-matching evaluation. No dataset agreement is involved,
which is the reason the deepfake detector work is blocked and this is not.

Usage:
    python scripts/prepare_face_crops.py --output data/results/face_crops_128.npy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_manipulation_set as swaps

from deepshield.face.backends import ARCFACE_TEMPLATE_112


def aligned_crop(cv2: Any, app: Any, image: np.ndarray, size: int) -> np.ndarray | None:
    """Return the face warped onto the canonical template, or None if none is found."""
    face = swaps.largest_face(app, image)
    if face is None:
        return None
    landmarks = np.asarray(face.kps, dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks, ARCFACE_TEMPLATE_112 * (size / 112.0), method=cv2.LMEDS
    )
    if matrix is None:
        return None
    return np.asarray(cv2.warpAffine(image, matrix, (size, size)))


def main(argv: list[str] | None = None) -> int:
    """Align every corpus face once and save the crops as one array."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--faces", type=Path, default=Path("data/sklearn/lfw_home/lfw_funneled")
    )
    parser.add_argument("--models", type=Path, default=Path("models/insightface"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args(argv)

    if not args.faces.is_dir():
        raise SystemExit(
            f"missing {args.faces}; run scripts/evaluate_face_pipeline.py first to fetch it"
        )

    import cv2

    from deepshield.media import load_image

    app = swaps.build_analyzer(args.models)
    paths = sorted(args.faces.glob("*/*.jpg"))
    crops = []
    for index, path in enumerate(paths, start=1):
        crop = aligned_crop(cv2, app, load_image(path), args.size)
        if crop is not None:
            crops.append(crop)
        if index % 2000 == 0:
            print(f"{index}/{len(paths)} kept={len(crops)}", flush=True)
    if not crops:
        raise SystemExit(f"no faces found under {args.faces}")

    stacked = np.stack(crops).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, stacked)
    print(f"wrote {args.output} {stacked.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
