r"""Score the LoRA-defence run: whether LoRAs trained on perturbed photos still draw the person.

The GPU half (``experiments/lora_defense/lora_defense.ipynb``) trains one
Stable Diffusion 1.5 LoRA per identity and per variant of that identity's
photographs, and generates sixteen images from each. This script unpacks its
``results.zip`` and measures every generated image with the tracking side's
own tools: the pipeline's ArcFace and an independent SFace, against galleries
of all 30 evaluation identities.

Per variant it reports how often a face is generated at all, the similarity
to the person the LoRA was trained on, how often that person ranks first,
and how often ArcFace clears the high-confidence threshold. The ``base``
control (no LoRA) gives the similarity a generic face reaches by chance. It
also reports the image quality each perturbation cost.

Usage:
    python scripts/evaluate_lora_defense.py \
        --results experiments/lora_defense/results.zip \
        --faces experiments/lora_defense/faces.zip
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_source_protection import Recognisers, score

from deepshield.media import load_image
from deepshield.quality import psnr, ssim


def read_png(bundle: zipfile.ZipFile, name: str) -> np.ndarray:
    """Return one PNG from a zip as an RGB array."""
    return np.asarray(Image.open(io.BytesIO(bundle.read(name))).convert("RGB"))


def main(argv: list[str] | None = None) -> int:
    """Score every generated image and write ``lora_defense.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--faces", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)

    recognise = Recognisers()
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(args.gallery.glob("*.png")):
        grouped[path.stem.rsplit("_", 1)[0]].append(path)
    galleries: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for identity, paths in grouped.items():
        if len(paths) < 3:
            continue
        galleries[identity] = {}
        for path in paths:
            vectors = recognise(load_image(path))
            if vectors is not None:
                galleries[identity][path.name] = vectors

    generated: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with zipfile.ZipFile(args.results) as results, zipfile.ZipFile(args.faces) as faces:
        for name in results.namelist():
            parts = name.split("/")
            if parts[0] == "generated" and name.endswith(".png"):
                generated[parts[1]][parts[2]].append(name)
        identities = sorted({i for v, by in generated.items() if v != "base" for i in by})

        rows: dict[str, list[dict[str, Any] | None]] = defaultdict(list)
        for variant, by_identity in generated.items():
            for identity, names in by_identity.items():
                for name in names:
                    probe = recognise(read_png(results, name))
                    if variant == "base":
                        for person in identities:
                            rows[f"base_vs_{person}"].append(score(probe, galleries, person, ""))
                    else:
                        rows[variant].append(score(probe, galleries, identity, ""))

        quality: dict[str, list[tuple[float, float]]] = defaultdict(list)
        clean = {n: n for n in faces.namelist() if n.startswith("clean/") and n.endswith(".png")}
        for name in clean:
            original = read_png(faces, name)
            twin = "arcface8/" + name.split("/", 1)[1]
            if twin in faces.namelist():
                changed = read_png(faces, twin)
                quality["arcface8"].append((psnr(original, changed), ssim(original, changed)))
            for variant in ("encoder8", "encoder16", "fsmg8", "fsmg16"):
                protected = f"protected/{variant}/" + name.split("/", 1)[1]
                if protected in results.namelist():
                    changed = read_png(results, protected)
                    quality[variant].append((psnr(original, changed), ssim(original, changed)))

    def summarise(entries: list[dict[str, Any] | None]) -> dict[str, Any]:
        scored = [e for e in entries if e is not None]
        out: dict[str, Any] = {"images": len(entries), "with_a_face": len(scored)}
        for model in ("arcface", "sface"):
            sims = [e[model]["donor_similarity"] for e in scored]
            out[model] = {
                "person_first": int(sum(e[model]["donor_first"] for e in scored)),
                "median_similarity": round(float(np.median(sims)), 4) if sims else None,
            }
        sims = [e["arcface"]["donor_similarity"] for e in scored]
        out["arcface"]["above_high_confidence"] = int(sum(v >= recognise.high for v in sims))
        return out

    report: dict[str, Any] = {
        "question": "does a LoRA trained on perturbed photographs still generate the person?",
        "identities": identities,
        "gallery_identities": len(galleries),
        "arcface_high_confidence": recognise.high,
        "variants": {variant: summarise(entries) for variant, entries in sorted(rows.items())},
        "image_quality": {
            variant: {
                "psnr": round(float(np.mean([q[0] for q in values])), 2),
                "ssim": round(float(np.mean([q[1] for q in values])), 4),
            }
            for variant, values in sorted(quality.items())
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "lora_defense.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
