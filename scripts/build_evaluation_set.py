"""Materialise a labelled face set for evaluation and threshold calibration.

Identity thresholds can only be defended by the data they were fitted on. The
handful of public-domain portraits used during early development produced ten
independent photo pairs, which is far too thin to place a decision boundary: on
that set every configuration scores a perfect AUC and nothing can be compared.

This script writes a subset of Labeled Faces in the Wild to disk with a manifest.
LFW is the standard face-verification benchmark, is distributed for research, and
has many photographs per identity taken under uncontrolled conditions, which is
what makes a genuine-versus-impostor comparison meaningful.

Its bias is well documented and matters here: LFW is drawn from news photography
and is heavily skewed by demographic and by pose. Numbers measured on it describe
this pipeline's behaviour on that population and should not be read as accuracy
for any particular user. Nothing this script writes is committed: the cache and
the crops it produces are named people, and ``data/sklearn`` and
``data/test/eval_faces`` are git-ignored so they are regenerated rather than
redistributed.

Usage:
    python scripts/build_evaluation_set.py --identities 30 --per-identity 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepshield.config import load_config
from deepshield.face.detector import build_detector
from deepshield.media import save_image

MIN_CONFIDENCE = 0.7
MIN_FACE_PIXELS = 60


def main(argv: list[str] | None = None) -> int:
    """Download LFW, keep photos with one clear face, and write a manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument("--cache", type=Path, default=Path("data/sklearn"))
    parser.add_argument("--identities", type=int, default=30)
    parser.add_argument("--per-identity", type=int, default=8)
    parser.add_argument("--min-photos", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    try:
        from sklearn.datasets import fetch_lfw_people
    except ImportError:
        raise SystemExit(
            "scikit-learn is required; install the 'experiments' extra"
        ) from None

    print("loading Labeled Faces in the Wild", flush=True)
    dataset = fetch_lfw_people(
        min_faces_per_person=args.min_photos,
        resize=1.0,
        color=True,
        slice_=None,
        data_home=str(args.cache.resolve()),
        download_if_missing=True,
    )
    images = (dataset.images * 255).clip(0, 255).astype(np.uint8)
    targets = dataset.target
    names = dataset.target_names

    counts = np.bincount(targets)
    eligible = [index for index in np.argsort(-counts) if counts[index] >= args.min_photos]
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.permutation(eligible)[: args.identities].tolist())

    config = load_config()
    detector = build_detector(config.face.detector)

    args.output.mkdir(parents=True, exist_ok=True)
    for existing in args.output.glob("*"):
        if existing.is_file():
            existing.unlink()

    manifest: list[tuple[str, str]] = []
    rejected = 0

    for target in chosen:
        identity = str(names[target]).lower().replace(" ", "_")
        indices = np.where(targets == target)[0]
        kept = 0
        for index in indices:
            if kept >= args.per_identity:
                break
            image = images[index]
            faces = detector.detect(image)
            usable = [
                face
                for face in faces
                if face.detection_confidence >= MIN_CONFIDENCE
                and min(face.bbox.width, face.bbox.height) >= MIN_FACE_PIXELS
            ]
            if len(usable) != 1:
                rejected += 1
                continue
            filename = f"{identity}_{kept}.png"
            save_image(image, args.output / filename)
            manifest.append((filename, identity))
            kept += 1
        print(f"  {identity:28s} {kept} photos", flush=True)

    manifest_path = args.output / "identities.txt"
    manifest_path.write_text(
        "\n".join(f"{name}\t{identity}" for name, identity in manifest) + "\n",
        encoding="utf-8",
    )

    identities = len({identity for _, identity in manifest})
    print(
        f"\nwrote {len(manifest)} photos across {identities} identities to {args.output}\n"
        f"rejected {rejected} photos without exactly one clear face\n"
        f"manifest: {manifest_path}"
    )
    print(
        "\nLFW is skewed by demographic and pose. These numbers describe this "
        "pipeline on that population, not accuracy for any particular user."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
