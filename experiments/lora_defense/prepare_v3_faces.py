"""Pick more people for the anti-LoRA test and write their photographs.

The first two runs used three people. This adds ``--count`` more from Labeled Faces in
the Wild, chosen at random (fixed seed) among the people with at least eight usable
photographs who are not in the evaluation gallery yet, with the same filter as
``scripts/build_evaluation_set.py``: exactly one detected face, confident and at least
60 pixels. For each it writes the first eight usable photographs:

- ``<output>/faces_v3.zip`` with ``clean/<identity>/<n>.png``, for Kaggle;
- ``<output>/gallery/<identity>_<n>.png``, the evaluation gallery: the existing one
  plus these people, the same photographs as the training ones, as for the first three.

Nothing here is committed: they are photographs of named people.

    python experiments/lora_defense/prepare_v3_faces.py --count 17 \
        --output experiments/lora_defense/v3
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_evaluation_set import MIN_CONFIDENCE, MIN_FACE_PIXELS, read_like_sklearn

from deepshield.config import load_config
from deepshield.face.detector import build_detector
from deepshield.media import save_image

PER_IDENTITY = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lfw", type=Path, default=ROOT / "data/sklearn/lfw_home/lfw_funneled")
    parser.add_argument("--gallery", type=Path, default=ROOT / "data/test/eval_faces")
    parser.add_argument("--count", type=int, default=17)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    taken = {p.stem.rsplit("_", 1)[0] for p in args.gallery.glob("*.png")}
    folders = [f for f in sorted(args.lfw.iterdir())
               if f.is_dir() and f.name.lower() not in taken
               and len(list(f.glob("*.jpg"))) >= PER_IDENTITY]
    order = np.random.default_rng(args.seed).permutation(len(folders))
    detector = build_detector(load_config().face.detector)

    gallery = args.output / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    for path in args.gallery.glob("*.png"):
        shutil.copy(path, gallery / path.name)
    chosen: list[str] = []
    with zipfile.ZipFile(args.output / "faces_v3.zip", "w") as bundle:
        for index in order:
            if len(chosen) == args.count:
                break
            folder = folders[index]
            usable = []
            for path in sorted(folder.glob("*.jpg")):
                image = read_like_sklearn(path)
                faces = [f for f in detector.detect(image)
                         if f.detection_confidence >= MIN_CONFIDENCE
                         and min(f.bbox.width, f.bbox.height) >= MIN_FACE_PIXELS]
                if len(faces) == 1:
                    usable.append(image)
                if len(usable) == PER_IDENTITY:
                    break
            if len(usable) < PER_IDENTITY:
                continue
            identity = folder.name.lower()
            for n, image in enumerate(usable):
                target = gallery / f"{identity}_{n}.png"
                save_image(image, target)
                bundle.write(target, f"clean/{identity}/{n}.png")
            chosen.append(identity)
            print(identity, flush=True)
    (args.output / "identities.txt").write_text("\n".join(chosen) + "\n")
    print(f"{len(chosen)} people, gallery of {len({p.stem.rsplit('_', 1)[0] for p in gallery.glob('*.png')})}")


if __name__ == "__main__":
    main()
