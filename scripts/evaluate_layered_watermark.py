r"""Measure what layering the keyed DCT mark and the learned mark buys against removal.

Each mark alone has a measured weak point. The keyed DCT grid is removed by
zeroing its eleven mid-band coefficients at about the cost of embedding it; the
learned face mark survives compression and regeneration but an attacker holding
its public decoder erases it with PGD and frames an innocent enrollee with less.
The idea tested here is to carry both and ask how much image quality an
attacker must give up to remove both, against a single mark of the same
embedding cost.

Configurations, per photograph: the original (the control every threshold is
fitted on), each mark alone, three layered versions (DCT then v3_39db, v3_39db
then DCT, DCT then v3), and three single-mark baselines at the layered image's
cost: the DCT mark at the layered PSNR, the v3 mark at the layered PSNR, and the
v3 mark at the layered SSIM. Matching PSNR alone flatters the DCT mark, which
spends its budget on structure that SSIM sees, while the learned residual is
nearly free in SSIM.

Attackers climb a ladder:

blind
    JPEG q50 and one Stable Diffusion VAE round trip.
knows the algorithms, not the key
    Zeroing the whole mid band, which removes any keyed DCT layout.
also holds the learned decoder
    40-step PGD in the aligned face to erase (4, 8, 16/255) or to frame another
    enrollee (4/255), alone and after the band is zeroed.
holds everything
    Removal on the key's own carriers, and forging the DCT with the key, alone
    and together with PGD framing of the learned mark.

A DCT mark is read two ways: by its CRC-gated decoder, which only reports a code
whose checksum holds, and by the soft correlation the learned mark is read with,
so the two marks can be compared under the same rule. A learned mark survives
when its best score clears a threshold fitted on unmarked originals of the
calibration half and names the owner. Every rate is reported on the test half
as a count out of ``n``, next to the union's false positives on unmarked photos.

Checkpointed per photograph under ``--work`` so an interrupted run resumes. Set
the DCT key in ``DEEPSHIELD_WATERMARK_KEY`` for this process only; it is never
written to the report. Needs torch; a GPU is strongly advised.

Usage:
    DEEPSHIELD_WATERMARK_KEY=... python scripts/evaluate_layered_watermark.py --n 150
    DEEPSHIELD_WATERMARK_KEY=... python scripts/evaluate_layered_watermark.py --report-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from deepshield.config import WatermarkConfig, load_config
from deepshield.experiments import environment
from deepshield.protection.fingerprint import dct2, idct2
from deepshield.protection.watermark import (
    BLOCK_SIZE,
    CODE_BITS,
    TILE_COLS,
    TILE_ROWS,
    DctWatermarker,
    build_message,
)
from deepshield.quality import psnr, ssim
from deepshield.types import WatermarkPayload

CODEBOOK_SIZE = 72
DCT_CARRIERS = 6
DCT_STRENGTH = 0.28
PGD_LEVELS = (4.0, 8.0, 16.0)
FRAME_LEVEL = 4.0
CALIBRATION_SHARE = 0.5
RUNGS = {
    "blind": ("jpeg50", "vae"),
    "algorithm": ("band",),
    "decoder": ("pgd_e4", "pgd_e8", "pgd_e16", "band_pgd_e4", "band_pgd_e8", "band_pgd_e16"),
    "key": ("keyrm", "keyrm_pgd_e8"),
}
FRAMING = ("pgd_f4", "band_pgd_f4", "forge_dct", "forge_all")


def codebook() -> np.ndarray:
    """Return the 72-code table the learned-mark benchmarks draw (seed 0)."""
    from learned_watermark import BITS

    return np.random.default_rng(0).integers(0, 2, (CODEBOOK_SIZE, BITS)).astype(np.uint8)


def payload_for(registrant: int) -> WatermarkPayload:
    """Return the DCT payload registrant ``registrant`` publishes with."""
    return WatermarkPayload(
        version=1, user_token=f"user{registrant}", asset_id="photo", distribution_id="a",
        timestamp="2026-09-25T00:00:00Z",
    )


class Marks:
    """Both marks, their decoders and every attack, built once for the whole run."""

    def __init__(self, key: str, checkpoints: dict[str, Path], device: str, models: Path) -> None:
        """Load the face stack, the learned models, the VAE and the keyed DCT watermarker."""
        import cv2
        import insightface
        from evaluate_learned_watermark_attribution import Model
        from evaluate_watermark_removal import VaeAttack, uninformed_attacks

        self.cv2 = cv2
        context = 0 if device == "cuda" else -1
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l", root=str(models / "insightface"),
            allowed_modules=["detection", "landmark_2d_106"],
        )
        self.app.prepare(ctx_id=context, det_size=(640, 640))
        self.recogniser = insightface.app.FaceAnalysis(
            name="buffalo_l", root=str(models / "insightface"),
            allowed_modules=["detection", "recognition"],
        )
        self.recogniser.prepare(ctx_id=context, det_size=(640, 640))
        self.device = device
        self.models = {name: Model(path, device, None) for name, path in checkpoints.items()}
        self.book = codebook()
        self.key = key
        self.wm = self.watermarker(DCT_STRENGTH)
        self.reader = DctWatermarker(
            WatermarkConfig(key=key, carrier_coefficients=DCT_CARRIERS, strength=DCT_STRENGTH,
                            resync_enabled=False)
        )
        self.jpeg50 = uninformed_attacks()["jpeg_50"]
        self.vae = VaeAttack(device)
        messages = np.stack(
            [build_message(payload_for(j).code(CODE_BITS)) for j in range(CODEBOOK_SIZE)]
        )
        self.dct_signs = 2.0 * messages.astype(np.float64) - 1.0
        self.dct_codes = {f"{payload_for(j).code(CODE_BITS):08x}": j for j in range(CODEBOOK_SIZE)}
        self.lpips = self._lpips()

    def _lpips(self) -> Any:
        try:
            import lpips
            import torch

            model = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()

            def distance(first: np.ndarray, second: np.ndarray) -> float:
                def tensor(image: np.ndarray) -> Any:
                    return (torch.from_numpy(image).permute(2, 0, 1)[None].float()
                            .div(127.5).sub(1).to(self.device))

                with torch.no_grad():
                    return float(model(tensor(first), tensor(second)).item())

            return distance
        except ImportError:
            return None

    def watermarker(self, strength: float) -> DctWatermarker:
        """Return the keyed spread DCT watermarker at ``strength``."""
        return DctWatermarker(
            WatermarkConfig(key=self.key, carrier_coefficients=DCT_CARRIERS, strength=strength)
        )

    def align(self, image: np.ndarray) -> np.ndarray | None:
        """Return the similarity transform onto the learned mark's 128-pixel face frame."""
        from evaluate_learned_watermark import align_matrix

        return align_matrix(self.cv2, self.app, image)

    def pgd(self, decoder: str, image: np.ndarray, level: float, target: np.ndarray | None,
            seed: int) -> np.ndarray | None:
        """Run the repository's EOT-PGD in the aligned face and paste it back."""
        import torch
        from evaluate_learned_watermark import SIZE
        from evaluate_watermark_removal import pgd_in_crop, warp_back

        matrix = self.align(image)
        if matrix is None:
            return None
        crop = self.cv2.warpAffine(image, matrix, (SIZE, SIZE))
        torch.manual_seed(seed)
        shift = pgd_in_crop(self.models[decoder], crop, level, target)
        return warp_back(self.cv2, image, matrix, shift)

    def remove_with_key(self, watermarker: DctWatermarker, image: np.ndarray) -> np.ndarray:
        """Move every block's reading on its own keyed carrier to zero."""
        from PIL import Image

        ycbcr = np.asarray(Image.fromarray(image).convert("YCbCr"), dtype=np.float64)
        luminance = ycbcr[:, :, 0]
        schedule = watermarker.schedule
        norm = float(schedule.carrier_size)
        for row in range(luminance.shape[0] // BLOCK_SIZE):
            for col in range(luminance.shape[1] // BLOCK_SIZE):
                y0, x0 = row * BLOCK_SIZE, col * BLOCK_SIZE
                block = dct2(luminance[y0 : y0 + BLOCK_SIZE, x0 : x0 + BLOCK_SIZE])
                slot = (row % TILE_ROWS) * TILE_COLS + (col % TILE_COLS)
                carrier = schedule.carriers[schedule.carrier_of_slot[slot]]
                block -= float(np.sum(carrier * block)) / norm * carrier
                luminance[y0 : y0 + BLOCK_SIZE, x0 : x0 + BLOCK_SIZE] = idct2(block)
        ycbcr[:, :, 0] = np.clip(luminance, 0, 255)
        return np.asarray(Image.fromarray(ycbcr.astype(np.uint8), mode="YCbCr").convert("RGB"))

    def identity(self, image: np.ndarray) -> np.ndarray | None:
        """Return the unit ArcFace embedding of the largest face."""
        faces = self.recogniser.get(np.ascontiguousarray(image[:, :, ::-1]))
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        vector = np.asarray(face.embedding, dtype=np.float64)
        return vector / np.linalg.norm(vector)

    def read(self, image: np.ndarray, decoders: list[str]) -> dict[str, Any]:
        """Read the DCT mark both ways and every requested learned decoder."""
        from evaluate_learned_watermark_attribution import score

        result = self.reader.detect(image)
        votes, counts = self.reader._extract_votes(image)
        ratios = np.divide(votes, counts, out=np.full(votes.shape, 0.5), where=counts > 0)
        soft = self.dct_signs @ (2.0 * ratios - 1.0)
        row: dict[str, Any] = {
            "dct_code": self.dct_codes.get(result.watermark_code, -1)
            if result.detected else None,
            "dct_soft": [round(float(value), 4) for value in soft],
        }
        for name in decoders:
            logits = self.models[name].logits(self.cv2, self.app, image)
            row[name] = None if logits is None else [
                round(float(value), 4) for value in score(logits, self.book)
            ]
        return row


def build_configs(marks: Marks, own: np.ndarray, truth: int, matched: dict[str, float]
                  ) -> dict[str, tuple[np.ndarray, str | None, bool]]:
    """Return every configuration's marked photograph, its decoder and whether it has a DCT mark."""
    code, payload = marks.book[truth], payload_for(truth)
    v39, v3 = marks.models["v3_39db"], marks.models["v3"]
    dct = marks.wm.embed(own, payload)
    alone39 = v39.mark(marks.cv2, marks.app, own, code)[0]
    configs = {
        "original": (own, None, False),
        "dct": (dct, None, True),
        "v3_39db": (alone39, "v3_39db", False),
        "v3": (v3.mark(marks.cv2, marks.app, own, code)[0], "v3", False),
        "dct_then_v3_39db": (v39.mark(marks.cv2, marks.app, dct, code)[0], "v3_39db", True),
        "v3_39db_then_dct": (marks.wm.embed(alone39, payload), "v3_39db", True),
        "dct_then_v3": (v3.mark(marks.cv2, marks.app, dct, code)[0], "v3", True),
        "dct_at_layered_psnr": (
            marks.watermarker(matched["dct_strength"]).embed(own, payload), None, True
        ),
    }
    native = v3.scale
    for name, scale in (("v3_at_layered_psnr", matched["v3_psnr_scale"]),
                        ("v3_at_layered_ssim", matched["v3_ssim_scale"])):
        v3.scale = scale
        configs[name] = (v3.mark(marks.cv2, marks.app, own, code)[0], "v3", False)
    v3.scale = native
    return configs


def matching(marks: Marks, photos: list[np.ndarray]) -> dict[str, float]:
    """Find the single-mark settings whose cost equals the layered image's, on a few photos."""
    sample = photos[:12]
    layered = []
    for index, own in enumerate(sample):
        truth = index % CODEBOOK_SIZE
        dct = marks.wm.embed(own, payload_for(truth))
        both = marks.models["v3_39db"].mark(marks.cv2, marks.app, dct, marks.book[truth])[0]
        layered.append((psnr(own, both), ssim(own, both)))
    target_psnr = float(np.mean([p for p, _ in layered]))
    target_ssim = float(np.mean([s for _, s in layered]))

    def mean_cost(strength: float) -> float:
        probe = marks.watermarker(strength)
        return float(np.mean([psnr(own, probe.embed(own, payload_for(i))) for i, own in
                              enumerate(sample)]))

    strengths = [0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44]
    dct_strength = min(strengths, key=lambda s: abs(mean_cost(s) - target_psnr))

    v3 = marks.models["v3"]
    native = v3.scale

    def v3_cost(scale: float) -> tuple[float, float]:
        v3.scale = scale
        costs = [
            v3.mark(marks.cv2, marks.app, own, marks.book[i])[0] for i, own in enumerate(sample)
        ]
        v3.scale = native
        return (float(np.mean([psnr(o, c) for o, c in zip(sample, costs, strict=True)])),
                float(np.mean([ssim(o, c) for o, c in zip(sample, costs, strict=True)])))

    native_psnr, _ = v3_cost(native)
    psnr_scale = native * 10 ** ((native_psnr - target_psnr) / 20.0)
    low, high = native, native * 20.0
    for _ in range(20):
        middle = (low + high) / 2.0
        if v3_cost(middle)[1] > target_ssim:
            low = middle
        else:
            high = middle
    return {
        "layered_psnr": round(target_psnr, 3),
        "layered_ssim": round(target_ssim, 4),
        "dct_strength": dct_strength,
        "v3_psnr_scale": round(float(psnr_scale), 5),
        "v3_ssim_scale": round(float((low + high) / 2.0), 5),
    }


def attack_photo(marks: Marks, index: int, own: np.ndarray, matched: dict[str, float]
                 ) -> list[dict[str, Any]]:
    """Build every configuration of one photograph, attack it and read every result."""
    truth = index % CODEBOOK_SIZE
    victim = (truth + 1) % CODEBOOK_SIZE
    victim_code, victim_payload = marks.book[victim], payload_for(victim)
    configs = build_configs(marks, own, truth, matched)
    reference = marks.identity(own)
    rows: list[dict[str, Any]] = []

    def record(config: str, attack: str, image: np.ndarray | None, decoders: list[str],
               marked: np.ndarray) -> None:
        if image is None:
            return
        embedding = marks.identity(image)
        row = {
            "index": index, "truth": truth, "victim": victim, "config": config, "attack": attack,
            "psnr_vs_original": round(psnr(own, image), 3) if image is not own else None,
            "psnr_vs_marked": round(psnr(marked, image), 3) if image is not marked else None,
            "ssim_vs_original": round(ssim(own, image), 4),
            "lpips_vs_original": None if marks.lpips is None else round(marks.lpips(own, image), 4),
            "identity_to_original": None if embedding is None or reference is None
            else round(float(embedding @ reference), 4),
        }
        row.update(marks.read(image, decoders))
        rows.append(row)

    for config, (marked, decoder, has_dct) in configs.items():
        control = config == "original"
        decoders = list(marks.models) if control else ([decoder] if decoder else [])
        seed = index * 100
        record(config, "none", marked, decoders, marked)
        record(config, "jpeg50", marks.jpeg50(marked), decoders, marked)
        record(config, "vae", marks.vae(marked), decoders, marked)
        banded = _blank(marked)
        record(config, "band", banded, decoders, marked)
        for name in decoders:
            tag = f"@{name}" if control else ""
            for level in PGD_LEVELS:
                record(config, f"pgd_e{level:g}{tag}",
                       marks.pgd(name, marked, level, None, seed + int(level)), [name], marked)
                if has_dct or control:
                    record(config, f"band_pgd_e{level:g}{tag}",
                           marks.pgd(name, banded, level, None, seed + int(level)), [name], marked)
            record(config, f"pgd_f4{tag}",
                   marks.pgd(name, marked, FRAME_LEVEL, victim_code, seed + 54), [name], marked)
            if has_dct or control:
                record(config, f"band_pgd_f4{tag}",
                       marks.pgd(name, banded, FRAME_LEVEL, victim_code, seed + 54), [name], marked)
        if has_dct or control:
            owner_wm = (marks.watermarker(matched["dct_strength"])
                        if config == "dct_at_layered_psnr" else marks.wm)
            removed = marks.remove_with_key(owner_wm, marked)
            record(config, "keyrm", removed, decoders, marked)
            forged = owner_wm.embed(marked, victim_payload)
            record(config, "forge_dct", forged, decoders, marked)
            for name in decoders:
                tag = f"@{name}" if control else ""
                record(config, f"keyrm_pgd_e8{tag}",
                       marks.pgd(name, removed, 8.0, None, seed + 70), [name], marked)
                record(config, f"forge_all{tag}",
                       marks.pgd(name, forged, FRAME_LEVEL, victim_code, seed + 74), [name], marked)
    return rows


def _blank(image: np.ndarray) -> np.ndarray:
    from evaluate_watermark_removal import dct_blank_band

    return dct_blank_band(image)


def fit_thresholds(
    rows: list[dict[str, Any]], calibration: set[int]
) -> dict[str, dict[str, float]]:
    """Fit zero-false-positive thresholds on unmarked calibration originals.

    ``deployed`` is fitted on untouched originals, the only operating point a
    deployment can have; ``per_attack`` is fitted on originals after the same
    attack, which a real reader cannot know but which shows the best case.
    """
    fitted: dict[str, dict[str, float]] = {}
    for name in ("v3_39db", "v3", "dct_soft"):
        per_attack: dict[str, list[float]] = {}
        for row in rows:
            if row["config"] != "original" or row["index"] not in calibration:
                continue
            values = row.get("dct_soft") if name == "dct_soft" else row.get(name)
            if values is None:
                continue
            attack = row["attack"].split("@")[0]
            per_attack.setdefault(attack, []).append(max(values))
        fitted[name] = {attack: float(max(values)) for attack, values in per_attack.items()}
        fitted[name]["deployed"] = fitted[name].get("none", float("inf"))
    return fitted


def judge(row: dict[str, Any], thresholds: dict[str, dict[str, float]], per_attack: bool
          ) -> dict[str, Any]:
    """Return which marks survive in one attacked image and whom each names."""
    attack = row["attack"].split("@")[0]
    verdict: dict[str, Any] = {"dct_crc": row["dct_code"]}

    def reading(name: str, values: list[float] | None) -> int | None:
        if values is None:
            return None
        table = thresholds[name]
        threshold = table.get(attack, table["deployed"]) if per_attack else table["deployed"]
        best = int(np.argmax(values))
        return best if values[best] > threshold else None

    verdict["dct_soft"] = reading("dct_soft", row["dct_soft"])
    for name in ("v3_39db", "v3"):
        if name in row:
            verdict[name] = reading(name, row[name])
    return verdict


def summarise(rows: list[dict[str, Any]], calibration: set[int]) -> dict[str, Any]:
    """Tabulate survival, wrong names, framing and removal cost on the test half."""
    thresholds = fit_thresholds(rows, calibration)
    decoder_of = {
        "dct": None, "dct_at_layered_psnr": None, "original": None,
        "v3_39db": "v3_39db", "dct_then_v3_39db": "v3_39db", "v3_39db_then_dct": "v3_39db",
        "v3": "v3", "dct_then_v3": "v3", "v3_at_layered_psnr": "v3", "v3_at_layered_ssim": "v3",
    }
    has_dct = {"dct", "dct_at_layered_psnr", "dct_then_v3_39db", "v3_39db_then_dct", "dct_then_v3"}
    test = [row for row in rows if row["index"] not in calibration]
    table: dict[str, dict[str, Any]] = {}
    for rule in ("deployed", "per_attack"):
        for row in test:
            if row["config"] == "original":
                continue
            seen = judge(row, thresholds, rule == "per_attack")
            decoder = decoder_of[row["config"]]
            owner, victim = row["truth"], row["victim"]
            learned = seen.get(decoder) if decoder else None
            dct_owner = row["config"] in has_dct and seen["dct_crc"] == owner
            union = dct_owner or learned == owner
            cell = table.setdefault(f"{row['config']}|{row['attack']}", {"n": 0})
            cell["n"] += 1 if rule == "deployed" else 0
            outcomes = (
                ("learned_owner", learned == owner),
                ("learned_victim", learned == victim),
                ("union_survives", union),
            ) if rule == "per_attack" else (
                ("dct_owner", dct_owner),
                ("dct_wrong_crc", seen["dct_crc"] is not None and seen["dct_crc"] != owner),
                ("dct_wrong_soft", seen["dct_soft"] is not None and seen["dct_soft"] != owner),
                ("learned_owner", learned == owner),
                ("learned_wrong", learned is not None and learned != owner),
                ("learned_victim", learned == victim),
                ("union_survives", union),
                ("conflict_caught", learned == victim and dct_owner),
                ("inconclusive", learned == victim and seen["dct_crc"] is None),
            )
            prefix = "" if rule == "deployed" else "per_attack_"
            for key, hit in outcomes:
                cell[prefix + key] = cell.get(prefix + key, 0) + int(bool(hit))
            if rule == "deployed":
                for metric in ("psnr_vs_original", "psnr_vs_marked", "ssim_vs_original",
                               "lpips_vs_original", "identity_to_original"):
                    if row.get(metric) is not None:
                        cell.setdefault(f"_{metric}", []).append(row[metric])
    for cell in table.values():
        for key in [k for k in cell if k.startswith("_")]:
            cell[key[1:] + "_mean"] = round(float(np.mean(cell.pop(key))), 3)

    false_positives: dict[str, dict[str, Any]] = {}
    for row in test:
        if row["config"] != "original":
            continue
        seen = judge(row, thresholds, per_attack=False)
        attack = row["attack"]
        for decoder in ("v3_39db", "v3"):
            if decoder not in row:
                continue
            cell = false_positives.setdefault(
                f"{decoder}|{attack}", {"n": 0, "union": 0, "dct_crc": 0, "learned": 0}
            )
            cell["n"] += 1
            cell["dct_crc"] += int(seen["dct_crc"] is not None)
            cell["learned"] += int(seen.get(decoder) is not None)
            cell["union"] += int(seen["dct_crc"] is not None or seen.get(decoder) is not None)

    costs: dict[str, dict[str, Any]] = {}
    by_image: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in test:
        by_image.setdefault((row["config"], row["index"]), {})[row["attack"]] = row
    for (config, _), attacks in by_image.items():
        if config == "original":
            continue
        allowed: list[str] = []
        for rung, names in RUNGS.items():
            allowed += list(names)
            removals = []
            for name in allowed:
                row = attacks.get(name)
                if row is None or row["psnr_vs_marked"] is None:
                    continue
                seen = judge(row, thresholds, per_attack=False)
                decoder = decoder_of[config]
                survives = (config in has_dct and seen["dct_crc"] == row["truth"]) or (
                    decoder is not None and seen.get(decoder) == row["truth"]
                )
                if not survives:
                    removals.append(row["psnr_vs_marked"])
            entry = costs.setdefault(config, {}).setdefault(rung, {"removable": 0, "psnr": []})
            if removals:
                entry["removable"] += 1
                entry["psnr"].append(max(removals))
    for rungs in costs.values():
        for entry in rungs.values():
            values = entry.pop("psnr")
            entry["median_cheapest_removal_psnr_vs_marked"] = (
                round(float(np.median(values)), 3) if values else None
            )
    return {
        "thresholds": thresholds,
        "cells": table,
        "false_positives_on_unmarked": false_positives,
        "removal_cost": costs,
    }


def main(argv: list[str] | None = None) -> int:
    """Build, attack and read every photograph, then tabulate the test half."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", type=Path, default=Path("data/sklearn/lfw_home/lfw_funneled"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--device", default=None)
    parser.add_argument("--work", type=Path, default=Path("data/results/layered_watermark_parts"))
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)

    key = os.environ.get("DEEPSHIELD_WATERMARK_KEY")
    if not key:
        raise SystemExit("set DEEPSHIELD_WATERMARK_KEY for this process; it is not stored")
    import torch
    from evaluate_learned_watermark_attribution import identity_pairs

    from deepshield.media import load_image

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")
    args.work.mkdir(parents=True, exist_ok=True)
    pairs = identity_pairs(args.faces)

    if not args.report_only:
        marks = Marks(key, {"v3_39db": args.models / "learned_watermark_v3_39db.pt",
                            "v3": args.models / "learned_watermark_v3.pt"}, device, args.models)
        kept: list[tuple[int, np.ndarray]] = []
        for index, (own_path, _) in enumerate(pairs):
            if len(kept) >= args.n:
                break
            own = load_image(own_path)
            if marks.align(own) is not None:
                kept.append((index, own))
        settings = args.work / "matched.json"
        if settings.is_file():
            matched = json.loads(settings.read_text(encoding="utf-8"))
        else:
            matched = matching(marks, [own for _, own in kept])
            settings.write_text(json.dumps(matched, indent=2), encoding="utf-8")
        print(f"{len(kept)} photographs; matched settings {matched}", flush=True)
        started = time.time()
        for position, (index, own) in enumerate(kept):
            if position % args.shards != args.shard:
                continue
            part = args.work / f"photo_{index:05d}.json"
            if part.is_file():
                continue
            rows = attack_photo(marks, index, own, matched)
            part.write_text(json.dumps(rows) + "\n", encoding="utf-8")
            print(f"{position + 1}/{len(kept)} photo {index}: {len(rows)} readings, "
                  f"{time.time() - started:.0f}s", flush=True)

    rows = []
    for part in sorted(args.work.glob("photo_*.json")):
        rows += json.loads(part.read_text(encoding="utf-8"))
    indices = sorted({row["index"] for row in rows})
    calibration = set(indices[: int(round(len(indices) * CALIBRATION_SHARE))])
    report = {
        "question": (
            "does layering a keyed DCT mark and a learned mark raise the cost of removing both?"
        ),
        "photographs": len(indices),
        "calibration_photographs": len(calibration),
        "matched": json.loads((args.work / "matched.json").read_text(encoding="utf-8"))
        if (args.work / "matched.json").is_file() else None,
        "lpips": any(row.get("lpips_vs_original") is not None for row in rows),
        **summarise(rows, calibration),
        "environment": environment(load_config()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "layered_watermark.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
