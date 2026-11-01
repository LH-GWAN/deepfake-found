r"""Measure the one-perturbation pilot's photographs: swaps, watermark, shield and cost.

The Kaggle pilot (kaggle_run_v5.sh) made two kinds of photograph meant to stop both
inswapper and a LoRA:

- ``shield_mist16``: Mist v2 run on the shielded photograph (watermark, then the shield,
  then Mist);
- ``joint8``: Mist v2 with the shield's loss inside its own PGD, from the watermarked
  photograph, in one budget of 8/255.

For every photograph this measures:

- the face swap as evaluate_mist_shield.py does it: inswapper onto another person's
  photograph, as saved and after JPEG 85, JPEG 70 and halving, scored against the whole
  gallery with the pipeline's ArcFace and an unattacked SFace;
- whether the watermark it started with (prepare_joint_inputs.py's manifest) still
  reads, as saved and after JPEG 85;
- the ArcFace similarity to its clean face in the aligned frame, which the shield pushes
  down and the joint optimisation stops pushing at ``tau``;
- PSNR and SSIM against the clean photograph enlarged to 512 pixels with bicubic
  resampling, as evaluate_mist_shield.py measures them.

Mist alone (``mist16``) and Mist with the shield stacked on it (``mist16_shield``) are
measured alongside for the same people. The LoRA half is scored by
scripts/evaluate_lora_defense.py. Run it from the checkout that holds ./models.

    python <this file> --work experiments/lora_defense/v3 \
        --clean experiments/lora_defense/v3/clean --gallery experiments/lora_defense/v3/gallery \
        --output data/results/lora_defense_v3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joint_arcface as joint
from check_joint_arcface import require_local_models, to_mist
from evaluate_source_protection import RESAVES, Recognisers, jpeg, score
from evaluate_verdicts import GanSwapper, photographs
from prepare_joint_inputs import PILOT

from deepshield.config import load_config
from deepshield.media import load_image
from deepshield.protection.shield import pick_device
from deepshield.protection.watermark import build_watermarker
from deepshield.quality import psnr, ssim

# The photographs each variant started from, whose watermark code the manifest holds.
MARKED_FROM = {"shield_mist16": "shield512", "joint8": "wm512", "joint12": "wm512"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True, help="<identity>/<n>.png")
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--inswapper", type=Path,
                        default=Path("models/inswapper/inswapper_128.onnx"))
    parser.add_argument("--people", default=",".join(PILOT))
    parser.add_argument("--variants", default="shield_mist16,joint8,mist16,mist16_shield")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", default="joint_pilot_photos.json",
                        help="file name of the report under --output")
    args = parser.parse_args()

    require_local_models()
    config = load_config()
    model_dir = Path(config.runtime.model_dir)
    device = pick_device()
    encoder = joint.OnnxGraph(str(model_dir / config.protection.shield.encoder_model), device)
    watermarker = build_watermarker(config.protection.watermark)
    inputs = args.work / "joint_inputs"
    manifest = json.loads((inputs / "manifest.json").read_text("utf-8"))
    landmarks = json.loads((inputs / "landmarks.json").read_text("utf-8"))

    people = args.people.split(",")
    variants = args.variants.split(",")
    everyone = sorted(p.name for p in args.clean.iterdir() if len(list(p.glob("*.png"))) >= 8)
    grouped = photographs(args.gallery)
    recognise = Recognisers()
    galleries = {
        identity: {p.name: v for p in paths if (v := recognise(load_image(p))) is not None}
        for identity, paths in grouped.items()
    }
    swapper = GanSwapper(model_dir / "insightface", args.inswapper)

    swaps: dict[str, list[dict[str, Any] | None]] = defaultdict(list)
    first_by_person: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    marks: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    frame: dict[str, list[float]] = defaultdict(list)
    quality: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for person in people:
        other = everyone[(everyone.index(person) + 1) % len(everyone)]
        picture = load_image(grouped[other][0])
        for n in range(8):
            name = f"{n}.png"
            clean = load_image(args.clean / person / name)
            marked = load_image(args.work / "protected" / "wm512" / person / name)
            term = joint.ArcFaceTerm(encoder, [landmarks[person][name]], to_mist(marked),
                                     device, tau=0.0, step=0.0)
            for variant in variants:
                photo = load_image(args.work / "protected" / variant / person / name)
                reference = np.asarray(Image.fromarray(clean).resize(
                    photo.shape[1::-1], Image.BICUBIC))
                quality[variant].append((psnr(reference, photo), ssim(reference, photo)))
                frame[variant].append(max(term.similarities(to_mist(photo), 0)))
                if variant in MARKED_FROM:
                    expected = manifest[MARKED_FROM[variant]][person][name]["watermark_code"]
                    for condition, image in (("as_saved", photo), ("jpeg85", jpeg(photo, 85))):
                        found = watermarker.detect(image)
                        marks[variant][condition] += int(
                            found.detected and found.watermark_code == expected)
                conditions = {variant: photo}
                conditions.update({f"{variant}_{k}": f(photo) for k, f in RESAVES.items()})
                for condition, image in conditions.items():
                    swapped = swapper(image, picture)
                    scored = score(None if swapped is None else recognise(swapped),
                                   galleries, person, f"{person}_{name}")
                    swaps[condition].append(scored)
                    if scored is not None and scored["arcface"]["donor_first"]:
                        first_by_person[condition][person] += 1
        print(f"{person} done", flush=True)

    def summarise(entries: list[dict[str, Any] | None]) -> dict[str, Any]:
        scored = [e for e in entries if e is not None]
        sims = [e["arcface"]["donor_similarity"] for e in scored]
        return {
            "swaps": len(entries), "with_a_face": len(scored),
            "arcface_owner_first": int(sum(e["arcface"]["donor_first"] for e in scored)),
            "sface_owner_first": int(sum(e["sface"]["donor_first"] for e in scored)),
            "median_owner_similarity": round(float(np.median(sims)), 4) if sims else None,
            "above_high_confidence": int(sum(v >= recognise.high for v in sims)),
        }

    report = {
        "question": "does one photograph stop both inswapper and (scored separately) a LoRA?",
        "people": people, "gallery_identities": len(galleries),
        "swaps": {c: summarise(e) for c, e in swaps.items()},
        "swap_owner_first_by_person": {c: dict(v) for c, v in first_by_person.items()},
        "watermark_read": {
            v: {**dict(c), "photos": 8 * len(people)} for v, c in marks.items()},
        "frame_similarity_to_clean_face": {
            v: {"median": round(float(np.median(s)), 4), "max": round(float(np.max(s)), 4)}
            for v, s in frame.items()},
        "quality_vs_clean_at_512_bicubic": {
            v: {"psnr": round(float(np.mean([q[0] for q in qs])), 2),
                "ssim": round(float(np.mean([q[1] for q in qs])), 4)} for v, qs in quality.items()},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / args.report).write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps({k: report[k] for k in (
        "watermark_read", "frame_similarity_to_clean_face",
        "quality_vs_clean_at_512_bicubic")}, indent=1))
    print(json.dumps({c: (s["arcface_owner_first"], s["with_a_face"])
                      for c, s in report["swaps"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
