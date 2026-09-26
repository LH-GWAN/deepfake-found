r"""Measure ``protect --mode shield`` end to end, as it ships.

``evaluate_source_protection.py`` measured the attack on its own: on clean
photographs, framed by the swapper's own detector, one face per photo. The
feature differs from that in ways that could matter: the shield runs on the
watermarked image, inside the protection pipeline, with its own seeded code.
This script runs that code and nothing else in its place.

For each of 30 identities, one photograph is protected in shield mode in a
temporary workspace, with the owner enrolled from their other photographs.
Then:

swap
    ``inswapper_128`` takes the identity from the shielded file (as saved, and
    after JPEG 85, JPEG 70 and halving) and puts it on another identity's
    photograph. Does the result still show the owner? Measured by the owner's
    rank among the 30 identities with ArcFace (the attacked model) and SFace
    (a recogniser nobody attacked), and by ArcFace's high-confidence
    threshold. The clean photograph is the baseline.
the file
    Does the shielded file still match its owner by face? What did it cost
    (PSNR, SSIM, seconds), and does its watermark still read after saving?
verdicts
    The shielded file re-saved as JPEG 85 should be ``own_copy``; another
    identity's face swapped onto it should be ``own_altered``.

Usage:
    python scripts/evaluate_shield_mode.py
    python scripts/evaluate_shield_mode.py --identities 5
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_source_protection import RESAVES, Recognisers, jpeg, score
from evaluate_verdicts import GanSwapper, enroll, isolated, photographs

from deepshield.config import load_config
from deepshield.experiments import environment
from deepshield.media import load_image, save_image
from deepshield.pipeline.analysis_pipeline import DefaultAnalysisPipeline
from deepshield.pipeline.protection_pipeline import DefaultProtectionPipeline
from deepshield.quality import psnr, ssim


def main(argv: list[str] | None = None) -> int:
    """Shield one photograph per identity through the pipeline and measure what it stops."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument("--identities", type=int, default=30)
    parser.add_argument("--models", type=Path, default=Path("models/insightface"))
    parser.add_argument(
        "--inswapper", type=Path, default=Path("models/inswapper/inswapper_128.onnx")
    )
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)

    grouped = photographs(args.faces)
    identities = sorted(grouped)[: args.identities]
    recognise = Recognisers()
    swapper = GanSwapper(args.models, args.inswapper)
    galleries: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for identity in identities:
        galleries[identity] = {}
        for path in grouped[identity]:
            vectors = recognise(load_image(path))
            if vectors is not None:
                galleries[identity][path.name] = vectors

    conditions = ["clean", "shielded", *(f"shielded_{name}" for name in RESAVES)]
    swaps: dict[str, list[dict[str, Any] | None]] = {c: [] for c in conditions}
    itself: list[dict[str, Any] | None] = []
    files: list[dict[str, Any]] = []
    verdicts: dict[str, Counter[str]] = {"repost_jpeg85": Counter(), "gan_target": Counter()}
    unshielded = 0

    with tempfile.TemporaryDirectory() as raw:
        workspace = Path(raw)
        config = isolated(load_config(), workspace)
        protection = DefaultProtectionPipeline(config)
        analysis = DefaultAnalysisPipeline(config)
        for index, owner in enumerate(identities):
            source_path = grouped[owner][0]
            enroll(analysis, owner, grouped[owner][1:])
            report = protection.protect(source_path, owner, "evaluation", mode="shield")
            if not report["shielded"]:
                unshielded += 1
                print(f"{index + 1}/{len(identities)} {owner}: no face shielded", flush=True)
                continue
            shielded = load_image(Path(report["protected_path"]))
            source = load_image(source_path)
            shield = report["shield"]
            files.append(
                {
                    "seconds": shield["seconds"],
                    "device": shield["device"],
                    "faces": shield["faces"],
                    "similarity_to_clean_face": shield["similarity_to_clean_face"],
                    "psnr": psnr(source, shielded),
                    "ssim": ssim(source, shielded),
                    "watermark_verified": report["watermark"]["verified_after_save"],
                }
            )
            itself.append(score(recognise(shielded), galleries, owner, source_path.name))

            other = identities[(index + 1) % len(identities)]
            picture = load_image(grouped[other][0])
            variants = {"clean": source, "shielded": shielded}
            variants.update({f"shielded_{n}": resave(shielded) for n, resave in RESAVES.items()})
            for condition, photo in variants.items():
                swapped = swapper(photo, picture)
                swaps[condition].append(
                    score(
                        None if swapped is None else recognise(swapped),
                        galleries, owner, source_path.name,
                    )
                )

            repost = save_image(jpeg(shielded, 85), workspace / f"{owner}_repost.png")
            verdicts["repost_jpeg85"][
                analysis.analyze_image(repost, owner).risk.verdict.value
            ] += 1
            target = swapper(picture, shielded)
            if target is not None:
                path = save_image(target, workspace / f"{owner}_target.png")
                verdicts["gan_target"][analysis.analyze_image(path, owner).risk.verdict.value] += 1
            print(f"{index + 1}/{len(identities)} {owner}", flush=True)

    def summarise(entries: list[dict[str, Any] | None]) -> dict[str, Any]:
        scored = [e for e in entries if e is not None]
        out: dict[str, Any] = {"scored": len(scored), "no_face": len(entries) - len(scored)}
        for model in ("arcface", "sface"):
            sims = [e[model]["donor_similarity"] for e in scored]
            out[model] = {
                "owner_first": int(sum(e[model]["donor_first"] for e in scored)),
                "median_owner_similarity": round(float(np.median(sims)), 4) if sims else None,
            }
        sims = [e["arcface"]["donor_similarity"] for e in scored]
        out["arcface"]["above_high_confidence"] = int(sum(v >= recognise.high for v in sims))
        return out

    def median(key: str) -> float | None:
        values = [f[key] for f in files if f[key] is not None]
        return round(float(np.median(values)), 4) if values else None

    config = load_config()
    moved = [value for f in files for value in f["similarity_to_clean_face"]]
    report_out: dict[str, Any] = {
        "question": "does protect --mode shield, as shipped, stop inswapper from carrying the "
        "owner, and what does the shielded file cost and keep?",
        "shield": config.protection.shield.model_dump(mode="json"),
        "identities": len(identities),
        "unshielded": unshielded,
        "arcface_high_confidence": recognise.high,
        "swaps": {condition: summarise(entries) for condition, entries in swaps.items()},
        "shielded_file_itself": summarise(itself),
        "shielded_file": {
            "median_seconds": median("seconds"),
            "devices": sorted({f["device"] for f in files}),
            "median_psnr": median("psnr"),
            "median_ssim": median("ssim"),
            "median_similarity_to_clean_face": (
                round(float(np.median(moved)), 4) if moved else None
            ),
            "watermark_verified": int(sum(bool(f["watermark_verified"]) for f in files)),
        },
        "verdicts": {name: dict(counts) for name, counts in verdicts.items()},
        "environment": environment(config),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "shield_mode.json"
    destination.write_text(json.dumps(report_out, indent=2), encoding="utf-8")
    print(json.dumps(report_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
