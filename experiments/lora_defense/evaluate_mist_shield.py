r"""Stack the swap shield on Mist photographs and measure whether face swaps are still stopped.

The shield (``protect --mode shield``) stops inswapper from carrying a person; Mist v2
at 8/255 stopped LoRA fine-tuning. One photograph would have to carry both. This runs
the shipped shield, through the protection pipeline with its watermark, on each Mist
photograph (``mist16_shield``), and on each clean one for comparison (``shield``), then
swaps every photograph's face onto another person's photograph with inswapper and asks
whether the result still shows the owner, among the whole gallery.

Conditions: ``clean``, ``mist16``, ``shield``, ``mist16_shield`` and the last after
JPEG 85, JPEG 70 and halving. Shielded photographs are written to
``<work>/protected/<variant>/<identity>/<n>.png`` as they are made, and a rerun skips
them; ``mist16_shield`` is what the LoRA half of the test trains on.

    python experiments/lora_defense/evaluate_mist_shield.py \
        --mist experiments/lora_defense/v3/protected/mist16 \
        --clean experiments/lora_defense/v3/clean --gallery experiments/lora_defense/v3/gallery \
        --work experiments/lora_defense/v3 --output data/results/lora_defense_v3
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_source_protection import RESAVES, Recognisers, score
from evaluate_verdicts import GanSwapper, enroll, isolated, photographs

from deepshield.config import load_config
from deepshield.media import load_image, save_image
from deepshield.pipeline.analysis_pipeline import DefaultAnalysisPipeline
from deepshield.pipeline.protection_pipeline import DefaultProtectionPipeline
from deepshield.quality import psnr, ssim


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mist", type=Path, required=True, help="<identity>/<n>.png")
    parser.add_argument("--clean", type=Path, required=True, help="<identity>/<n>.png")
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=ROOT / "models/insightface")
    parser.add_argument("--inswapper", type=Path,
                        default=ROOT / "models/inswapper/inswapper_128.onnx")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    people = sorted(p.name for p in args.mist.iterdir() if len(list(p.glob("*.png"))) >= 8)
    grouped = photographs(args.gallery)
    recognise = Recognisers()
    galleries = {
        identity: {p.name: v for p in paths if (v := recognise(load_image(p))) is not None}
        for identity, paths in grouped.items()
    }
    swapper = GanSwapper(args.models, args.inswapper)

    with tempfile.TemporaryDirectory() as raw:
        workspace = Path(raw)
        config = isolated(load_config(), workspace)
        protection = DefaultProtectionPipeline(config)
        analysis = DefaultAnalysisPipeline(config)

        def shielded(source: Path, variant: str, owner: str, n: int) -> Path | None:
            target = args.work / "protected" / variant / owner / f"{n}.png"
            if not target.exists():
                report = protection.protect(source, owner, "evaluation", mode="shield")
                if not report["shielded"]:
                    return None
                target.parent.mkdir(parents=True, exist_ok=True)
                save_image(load_image(Path(report["protected_path"])), target)
            return target

        swaps: dict[str, list[dict[str, Any] | None]] = defaultdict(list)
        quality: dict[str, list[tuple[float, float]]] = defaultdict(list)
        unshielded: dict[str, int] = defaultdict(int)
        for index, owner in enumerate(people):
            enroll(analysis, owner, grouped[owner])
            other = people[(index + 1) % len(people)]
            picture = load_image(grouped[other][0])
            for n in range(8):
                clean_path = args.clean / owner / f"{n}.png"
                mist_path = args.mist / owner / f"{n}.png"
                clean = load_image(clean_path)
                mist = load_image(mist_path)
                photos = {"clean": clean, "mist16": mist}
                for variant, source in (("shield", clean_path), ("mist16_shield", mist_path)):
                    path = shielded(source, variant, owner, n)
                    if path is None:
                        unshielded[variant] += 1
                    else:
                        photos[variant] = load_image(path)
                if "mist16_shield" in photos:
                    both = photos["mist16_shield"]
                    photos.update({f"mist16_shield_{k}": f(both) for k, f in RESAVES.items()})
                    # Quality against the clean photograph at Mist's 512 pixels.
                    reference = np.asarray(Image.fromarray(clean).resize(
                        both.shape[1::-1], Image.BICUBIC))
                    quality["mist16_shield"].append((psnr(reference, both), ssim(reference, both)))
                    quality["mist16"].append((psnr(reference, mist), ssim(reference, mist)))
                for condition, photo in photos.items():
                    swapped = swapper(photo, picture)
                    swaps[condition].append(score(
                        None if swapped is None else recognise(swapped),
                        galleries, owner, f"{owner}_{n}.png"))
            print(f"{index + 1}/{len(people)} {owner}", flush=True)

    def summarise(entries: list[dict[str, Any] | None]) -> dict[str, Any]:
        scored = [e for e in entries if e is not None]
        out: dict[str, Any] = {"swaps": len(entries), "with_a_face": len(scored)}
        for model in ("arcface", "sface"):
            sims = [e[model]["donor_similarity"] for e in scored]
            out[model] = {
                "owner_first": int(sum(e[model]["donor_first"] for e in scored)),
                "median_owner_similarity": round(float(np.median(sims)), 4) if sims else None,
            }
        sims = [e["arcface"]["donor_similarity"] for e in scored]
        out["arcface"]["above_high_confidence"] = int(sum(v >= recognise.high for v in sims))
        return out

    report = {
        "question": "does the swap shield still stop inswapper when it is stacked on Mist v2?",
        "people": people,
        "gallery_identities": len(galleries),
        "arcface_high_confidence": recognise.high,
        "unshielded": dict(unshielded),
        "swaps": {c: summarise(e) for c, e in swaps.items()},
        "photo_quality_vs_clean_at_512": {
            v: {"psnr": round(float(np.mean([q[0] for q in qs])), 2),
                "ssim": round(float(np.mean([q[1] for q in qs])), 4)}
            for v, qs in quality.items()
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "mist_shield_swap.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
