r"""Make the inputs of the joint swap-and-LoRA test, locally.

Mist v2 enlarges each 250-pixel photograph to 512 pixels with bilinear resampling
before perturbing it. The joint test starts from that same 512-pixel photograph, marked
the way the protection pipeline marks it, in two forms:

- ``wm512``: the watermark only (``protect`` in trace mode). The joint optimisation on
  Kaggle starts here and has to fit the shield and Mist into one budget.
- ``shield512``: the watermark and the shipped swap shield (``protect --mode shield``).
  Mist is then run on it unchanged: the shield first, Mist after.

It also writes the five SCRFD landmarks of every face in each ``wm512`` photograph, the
frame the shield optimises in, which the joint optimisation needs on Kaggle, and a
manifest of the watermark code each photograph carries, so that the watermark can be
checked again after the noise is added. Photographs land in
``<work>/protected/<variant>/<identity>/<n>.png`` and the rest in
``<work>/joint_inputs/``; a rerun skips what exists.

    python experiments/lora_defense/prepare_joint_inputs.py \
        --clean experiments/lora_defense/v3/clean --work experiments/lora_defense/v3
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_verdicts import isolated

from deepshield.config import FaceDetectorConfig, load_config
from deepshield.face.backends import InsightFaceDetector
from deepshield.media import load_image, save_image
from deepshield.pipeline.protection_pipeline import DefaultProtectionPipeline

SIZE = 512
# The people whose Mist-alone and Mist-then-shield LoRAs differed most (README, "LoRA 방어
# 3차"): 8 of 56 generations ranked them first after Mist alone, 43 of 63 after the stack.
PILOT = ["hugo_chavez", "jeb_bush", "jennifer_lopez", "nicole_kidman"]


def enlarge(path: Path) -> Image.Image:
    """Return the photograph as Mist sees it: torchvision's bilinear Resize of a PIL image."""
    return Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)


def write_json(path: Path, value: Any) -> None:
    """Write ``value`` as JSON through a temporary file, so a stop never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".part")
    partial.write_text(json.dumps(value, indent=1), "utf-8")
    partial.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", type=Path, required=True, help="<identity>/<n>.png")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--shield-people", default=",".join(PILOT),
                        help="comma-separated identities that also get shield512")
    args = parser.parse_args()

    people = sorted(p.name for p in args.clean.iterdir() if len(list(p.glob("*.png"))) >= 8)
    shield_people = set(args.shield_people.split(",")) if args.shield_people else set()
    missing = shield_people - set(people)
    if missing:
        raise SystemExit(f"not among the people: {sorted(missing)}")
    inputs = args.work / "joint_inputs"
    manifest_path, landmarks_path = inputs / "manifest.json", inputs / "landmarks.json"
    manifest: dict[str, Any] = (
        json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {})
    landmarks: dict[str, Any] = (
        json.loads(landmarks_path.read_text("utf-8")) if landmarks_path.exists() else {})

    with tempfile.TemporaryDirectory() as raw:
        workspace = Path(raw)
        config = isolated(load_config(), workspace)
        if config.protection.shield.landmark_detector != "insightface":
            raise SystemExit("the shield is configured with another landmark detector")
        protection = DefaultProtectionPipeline(config)
        # The detector the shield places its frame with (``_shield_detector``).
        detector = InsightFaceDetector(
            FaceDetectorConfig(backend="insightface"), Path(config.runtime.model_dir))

        for index, person in enumerate(people):
            variants = [("wm512", "trace")]
            if person in shield_people:
                variants.append(("shield512", "shield"))
            for n in range(8):
                name = f"{n}.png"
                source = workspace / "clean512" / person / name
                source.parent.mkdir(parents=True, exist_ok=True)
                enlarge(args.clean / person / name).save(source)
                for variant, mode in variants:
                    target = args.work / "protected" / variant / person / name
                    entry = manifest.get(variant, {}).get(person, {}).get(name)
                    if target.exists() and entry is not None:
                        continue
                    started = time.perf_counter()
                    report = protection.protect(source, person, "evaluation", mode=mode)
                    if mode == "shield" and not report["shielded"]:
                        raise SystemExit(f"no face to shield in {person}/{name}")
                    if not report["watermark"]["verified_after_save"]:
                        raise SystemExit(f"watermark not read back from {person}/{name}")
                    save_image(load_image(Path(report["protected_path"])), target)
                    manifest.setdefault(variant, {}).setdefault(person, {})[name] = {
                        "watermark_code": report["watermark"]["code"],
                        "psnr_vs_enlarged_clean": report["quality"]["psnr"],
                        "ssim_vs_enlarged_clean": report["quality"]["ssim"],
                        "shield_similarity_to_clean_face":
                            report["shield"].get("similarity_to_clean_face"),
                        "seconds": round(time.perf_counter() - started, 2),
                    }
                if name not in landmarks.get(person, {}):
                    marked = load_image(args.work / "protected" / "wm512" / person / name)
                    faces = [
                        np.asarray(face.landmarks, dtype=np.float64)[:5].round(3).tolist()
                        for face in detector.detect(marked)
                        if face.landmarks is not None and len(face.landmarks) >= 5
                    ]
                    if not faces:
                        raise SystemExit(f"no face with landmarks in wm512 {person}/{name}")
                    landmarks.setdefault(person, {})[name] = faces
            write_json(manifest_path, manifest)
            write_json(landmarks_path, landmarks)
            print(f"{index + 1}/{len(people)} {person}", flush=True)

    counts = {v: sum(len(p) for p in manifest.get(v, {}).values()) for v in ("wm512", "shield512")}
    faces = Counter(len(f) for person in landmarks.values() for f in person.values())
    print(json.dumps({"photos": counts, "faces_per_photo": dict(sorted(faces.items()))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
