r"""Find what undid Mist when the swap shield was stacked on it.

A LoRA sees its training photographs only through the Stable Diffusion VAE, as latents,
so whatever Mist v2 does to a LoRA it does by moving each photograph's latent. Stacking
``protect --mode shield`` on the Mist photographs brought the LoRA back from 32% to 63% of
generations ranking the person first (README, "LoRA 방어 3차"). The stack adds two
things, the watermark and the shield's 8/255 noise. This makes Mist photographs with each
one alone and measures how much of Mist's latent shift survives:

- ``mist16_wm``: the watermark only (``protect`` in trace mode);
- ``mist16_shieldonly``: the shield's noise only, no watermark, on the first
  ``--shield-photos`` photographs of each person (the shield takes seconds a photograph);
- ``mist16_shield``: both, as shipped (made by evaluate_mist_shield.py);
- ``mist16_random8``: uniform random +-8/255 noise over the whole photograph, a control
  that is at least as large as the shield's noise and aimed at nothing.

With z_c the latent of the 512-pixel bilinear enlargement Mist starts from, z_m the
latent of the Mist photograph and z_v a variant's, Mist's shift is d = z_m - z_c, and

- kept = <z_v - z_c, d> / |d|^2 is 1 when all of Mist's shift is still there, 0 when none is;
- moved = |z_v - z_m| / |d| is how far the variant's latent moved off Mist's, in units
  of Mist's own shift.

The watermark alone and watermark plus shield on a clean photograph (``wm512``,
``shield512`` from prepare_joint_inputs.py) are measured against the clean latent, for
scale. Per person, the stack's kept is set against how much that person's LoRA
recovered. The VAE runs forward only, from the local cache.

    python experiments/lora_defense/diagnose_mist_overwrite.py \
        --work experiments/lora_defense/v3 --clean experiments/lora_defense/v3/clean \
        --lora-results data/results/lora_defense_v3/lora_defense.json \
        --output data/results/lora_defense_v3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the cached SD 1.5 VAE; never download

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_verdicts import isolated
from prepare_joint_inputs import enlarge

from deepshield.config import FaceDetectorConfig, load_config
from deepshield.face.backends import InsightFaceDetector
from deepshield.media import load_image, save_image
from deepshield.pipeline.protection_pipeline import DefaultProtectionPipeline
from deepshield.protection.shield import SwapShield, pick_device

MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
FROM_MIST = ("mist16_wm", "mist16_shieldonly", "mist16_shield", "mist16_random8")
FROM_CLEAN = ("wm512", "shield512")


def make_variants(work: Path, people: list[str], shield_photos: int) -> None:
    """Write ``mist16_wm`` and ``mist16_shieldonly`` next to the other variants."""
    with tempfile.TemporaryDirectory() as raw:
        config = isolated(load_config(), Path(raw))
        protection = DefaultProtectionPipeline(config)
        detector = InsightFaceDetector(
            FaceDetectorConfig(backend="insightface"), Path(config.runtime.model_dir))
        shield = SwapShield(config.protection.shield, Path(config.runtime.model_dir))
        for index, person in enumerate(people):
            for n in range(8):
                source = work / "protected" / "mist16" / person / f"{n}.png"
                target = work / "protected" / "mist16_wm" / person / f"{n}.png"
                if not target.exists():
                    report = protection.protect(source, person, "evaluation", mode="trace")
                    save_image(load_image(Path(report["protected_path"])), target)
                target = work / "protected" / "mist16_shieldonly" / person / f"{n}.png"
                if n < shield_photos and not target.exists():
                    image = load_image(source)
                    shielded, report = shield.shield(image, detector.detect(image))
                    if not report["applied"]:
                        raise SystemExit(f"no face to shield in mist16 {person}/{n}")
                    save_image(shielded, target)
            print(f"variants {index + 1}/{len(people)} {person}", flush=True)


def correlation(x: list[float], y: list[float]) -> dict[str, float]:
    """Return Pearson's and Spearman's coefficients."""
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ranks = lambda v: np.argsort(np.argsort(v)).astype(np.float64)  # noqa: E731
    return {"pearson": round(float(np.corrcoef(a, b)[0, 1]), 3),
            "spearman": round(float(np.corrcoef(ranks(a), ranks(b))[0, 1]), 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True, help="<identity>/<n>.png")
    parser.add_argument("--lora-results", type=Path, required=True)
    parser.add_argument("--shield-photos", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from diffusers import AutoencoderKL

    mist = args.work / "protected" / "mist16"
    people = sorted(p.name for p in mist.iterdir() if len(list(p.glob("*.png"))) >= 8)
    make_variants(args.work, people, args.shield_photos)

    device = pick_device()
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae").to(device).eval()

    def latent(image: np.ndarray) -> torch.Tensor:
        batch = torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].float().div(127.5).sub(1)
        with torch.no_grad():
            encoded = vae.encode(batch.to(device)).latent_dist.mean
        return (encoded * vae.config.scaling_factor).flatten().cpu().double()

    rows: list[dict[str, Any]] = []
    for index, person in enumerate(people):
        for n in range(8):
            name = f"{n}.png"
            z_c = latent(np.asarray(enlarge(args.clean / person / name)))
            mist_image = load_image(mist / person / name)
            z_m = latent(mist_image)
            d = z_m - z_c
            row: dict[str, Any] = {"person": person, "n": n, "mist_shift": float(d.norm())}
            noise = np.random.default_rng(index * 8 + n).choice([-8, 8], size=mist_image.shape)
            images = {"mist16_random8": np.clip(mist_image.astype(np.int16) + noise, 0, 255)
                      .astype(np.uint8)}
            for variant in FROM_MIST[:-1] + FROM_CLEAN:
                path = args.work / "protected" / variant / person / name
                if path.exists():
                    images[variant] = load_image(path)
            for variant, image in images.items():
                z_v = latent(image)
                if variant in FROM_MIST:
                    row[variant] = {
                        "kept": float((z_v - z_c) @ d / (d @ d)),
                        "moved": float((z_v - z_m).norm() / d.norm()),
                    }
                else:
                    shift = z_v - z_c
                    row[variant] = {
                        "size": float(shift.norm() / d.norm()),
                        "cosine_with_mist": float(shift @ d / (shift.norm() * d.norm())),
                    }
            rows.append(row)
        print(f"latents {index + 1}/{len(people)} {person}", flush=True)

    def summary(variant: str, key: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
        values = [r[variant][key] for r in subset if variant in r]
        if not values:
            return {}
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        return {"photos": len(values), "median": round(float(median), 3),
                "iqr": [round(float(q1), 3), round(float(q3), 3)],
                "mean": round(float(np.mean(values)), 3)}

    # The variants compared on the same photographs: those that also have shieldonly.
    both = [r for r in rows if "mist16_shieldonly" in r]
    lora = json.loads(args.lora_results.read_text("utf-8"))["per_identity"]
    first = {
        variant: {p: e["arcface_person_first"] / e["with_a_face"] for p, e in lora[variant].items()}
        for variant in ("mist16", "mist16_shield")
    }
    per_person = {}
    for person in people:
        mine = [r for r in rows if r["person"] == person]
        per_person[person] = {
            "kept": {v: round(float(np.mean([r[v]["kept"] for r in mine if v in r])), 3)
                     for v in FROM_MIST if any(v in r for r in mine)},
            "lora_first_mist16": round(first["mist16"][person], 3),
            "lora_first_mist16_shield": round(first["mist16_shield"][person], 3),
        }
    recovery = [first["mist16_shield"][p] - first["mist16"][p] for p in people]
    report = {
        "question": "which part of the stacked swap shield undoes Mist v2 in the VAE latent?",
        "people": len(people),
        "vae": f"{MODEL} (forward only, latent_dist.mean x scaling factor)",
        "from_mist": {
            v: {"all_photos": {k: summary(v, k, rows) for k in ("kept", "moved")},
                "same_photos_as_shieldonly": {k: summary(v, k, both) for k in ("kept", "moved")}}
            for v in FROM_MIST
        },
        "from_clean": {v: {k: summary(v, k, rows) for k in ("size", "cosine_with_mist")}
                       for v in FROM_CLEAN},
        "stack_kept_vs_lora_recovery": {
            "note": "per person: mean kept of mist16_shield against the rise in the share "
                    "of generations ranking the person first, mist16_shield minus mist16",
            **correlation([per_person[p]["kept"]["mist16_shield"] for p in people], recovery),
        },
        "per_person": per_person,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "mist_overwrite.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps({k: report[k] for k in ("from_mist", "from_clean",
                                              "stack_kept_vs_lora_recovery")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
