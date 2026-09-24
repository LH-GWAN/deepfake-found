"""Measure whether identity matching finds the person whose face a swap used.

A face swap has two people in it. The donor supplies the face; the target
supplies the picture it is pasted into. The question this project started from,
"was my face used in a deepfake", is about the donor, and every watermark in the
repository measured at chance in that direction: a GAN swapper receives the
donor only as an identity embedding, so no donor pixel reaches the output.

That same fact means the identity itself is what a swap is built to carry. This
script measures it directly on the manipulation sets built by
``build_manipulation_set.py``, whose manifests record both people for every
fake. For each fake, every enrolled identity is represented by its photographs
in the evaluation set, with the one photograph used to make the fake left out,
so a match cannot come from comparing a file with itself.

Reported per swapper and per embedder:

* how often the donor is the top-ranked identity among all enrolled ones;
* how often the donor clears the calibrated candidate and confidence
  thresholds, for the embedder those thresholds were fitted on;
* the same for the target, and the best unrelated identity, as controls;
* genuine photographs under the same leave-one-out protocol, and the ROC-AUC
  of telling them from swaps by donor similarity alone. That number is what
  decides whether identity evidence could ever say "genuine" or "synthetic";
  the verdict engine assumes it cannot.

A second embedder that the swapper was not built around (``opencv_sface``) is
included because inswapper conditions on an ArcFace embedding, and a recogniser
from the same family could agree with it for reasons that would not transfer.

Usage:
    python scripts/evaluate_identity_under_swap.py
    python scripts/evaluate_identity_under_swap.py --embedders insightface
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepshield.config import DeepShieldConfig, load_config
from deepshield.experiments import environment
from deepshield.face.aligner import build_aligner
from deepshield.face.detector import build_detector
from deepshield.face.embedder import build_embedder
from deepshield.media import load_image
from deepshield.risk.calibration import roc_curve

SETS = {
    "graphics": Path("data/test/manipulated"),
    "inswapper": Path("data/test/manipulated_inswapper"),
}


def identity_of(filename: str) -> str:
    """Return the identity encoded in an evaluation-set filename."""
    return Path(filename).stem.rsplit("_", 1)[0]


class Embedder:
    """Detect, align and embed the largest face with one configured backend."""

    def __init__(self, config: DeepShieldConfig) -> None:
        """Build the face stack for ``config``."""
        self.detector = build_detector(config.face.detector)
        self.aligner = build_aligner(config.face.aligner)
        self.embedder = build_embedder(config.face.embedder)

    def __call__(self, path: Path) -> np.ndarray | None:
        """Return the embedding of the most confident face, or ``None``."""
        image = load_image(path)
        faces = self.detector.detect(image)
        if not faces:
            return None
        face = max(faces, key=lambda f: f.detection_confidence)
        return self.embedder.embed(self.aligner.align(image, face).image).vector


def best_similarity(
    probe: np.ndarray, gallery: dict[str, np.ndarray], exclude: str | None
) -> float | None:
    """Return the max cosine similarity to a gallery, leaving one photo out."""
    scores = [float(probe @ vector) for name, vector in gallery.items() if name != exclude]
    return max(scores) if scores else None


def rate(values: list[bool]) -> float | None:
    """Return the share of true values, or ``None`` for an empty list."""
    return round(float(np.mean(values)), 4) if values else None


def summary(values: list[float]) -> dict[str, float] | None:
    """Return mean, median and 5th percentile of a list of similarities."""
    if not values:
        return None
    array = np.asarray(values)
    return {
        "mean": round(float(array.mean()), 4),
        "median": round(float(np.median(array)), 4),
        "p05": round(float(np.percentile(array, 5)), 4),
    }


def evaluate(
    embed: Embedder,
    galleries: dict[str, dict[str, np.ndarray]],
    manifest: dict[str, Any],
    thresholds: tuple[float, float] | None,
    genuine: list[float],
) -> dict[str, Any]:
    """Score every fake against the donor, the target and everyone else."""
    rows: dict[str, list[float]] = defaultdict(list)
    ranks: list[str] = []
    undetected = 0
    for record in manifest["records"]:
        if record["label"] != "fake":
            continue
        probe = embed(ROOT / record["path"])
        if probe is None:
            undetected += 1
            continue
        donor, target = identity_of(record["donor"]), record["identity"]
        scores = {
            identity: best_similarity(
                probe,
                gallery,
                record["donor"] if identity == donor
                else record["source"] if identity == target
                else None,
            )
            for identity, gallery in galleries.items()
        }
        valid = {k: v for k, v in scores.items() if v is not None}
        if donor not in valid or target not in valid:
            continue
        rows["donor"].append(valid[donor])
        rows["target"].append(valid[target])
        rows["other"].append(max(v for k, v in valid.items() if k not in (donor, target)))
        top = max(valid, key=lambda k: valid[k])
        ranks.append("donor" if top == donor else "target" if top == target else "other")

    result: dict[str, Any] = {
        "fakes_scored": len(ranks),
        "fakes_without_a_face": undetected,
        "top1": {role: rate([r == role for r in ranks]) for role in ("donor", "target", "other")},
        "similarity": {role: summary(values) for role, values in rows.items()},
    }
    if genuine and rows["donor"]:
        pooled = np.asarray(genuine + rows["donor"])
        labels = np.asarray([1] * len(genuine) + [0] * len(rows["donor"]))
        result["genuine_vs_swap_auc"] = round(float(roc_curve(pooled, labels).auc), 4)
    if thresholds is not None:
        candidate, high = thresholds
        result["thresholds"] = {"candidate": candidate, "high_confidence": high}
        result["above_candidate"] = {
            role: rate([v >= candidate for v in values]) for role, values in rows.items()
        }
        result["above_high_confidence"] = {
            role: rate([v >= high for v in values]) for role, values in rows.items()
        }
    return result


def genuine_scores(galleries: dict[str, dict[str, np.ndarray]]) -> list[float]:
    """Score each genuine photo against its own identity, leaving itself out."""
    own: list[float] = []
    for gallery in galleries.values():
        for name, vector in gallery.items():
            score = best_similarity(vector, gallery, name)
            if score is not None:
                own.append(score)
    return own


def genuine_control(own: list[float], thresholds: tuple[float, float] | None) -> dict[str, Any]:
    """Summarise the genuine scores the swaps are compared against."""
    control: dict[str, Any] = {"photos": len(own), "similarity": summary(own)}
    if thresholds is not None:
        control["above_high_confidence"] = rate([v >= thresholds[1] for v in own])
    return control


def main(argv: list[str] | None = None) -> int:
    """Run the donor-identity measurement for every swapper and embedder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument(
        "--embedders", nargs="+", default=["insightface", "opencv_sface"]
    )
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)

    if not args.faces.is_dir():
        raise SystemExit(f"missing {args.faces}; run scripts/build_evaluation_set.py first")
    manifests = {
        name: json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for name, directory in SETS.items()
        if (directory / "manifest.json").is_file()
    }
    if not manifests:
        raise SystemExit("no manipulation set found; run scripts/build_manipulation_set.py first")

    base = load_config()
    calibrated = base.thresholds.face_similarity
    report: dict[str, Any] = {
        "question": "does identity matching find the donor whose face a swap used?",
        "evaluation_faces": str(args.faces),
        "embedders": {},
        "environment": environment(base),
    }
    for backend in args.embedders:
        config = base.model_copy(
            update={
                "face": base.face.model_copy(
                    update={
                        "embedder": base.face.embedder.model_copy(update={"backend": backend})
                    }
                )
            }
        )
        embed = Embedder(config)
        galleries: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
        for path in sorted(args.faces.glob("*.png")):
            vector = embed(path)
            if vector is not None:
                galleries[identity_of(path.name)][path.name] = vector
        thresholds = (
            (calibrated.candidate_threshold, calibrated.high_confidence_threshold)
            if backend == base.face.embedder.backend and calibrated.calibrated
            else None
        )
        genuine = genuine_scores(galleries)
        report["embedders"][backend] = {
            "identities": len(galleries),
            "thresholds_apply": thresholds is not None,
            "genuine_control": genuine_control(genuine, thresholds),
            "swappers": {
                name: evaluate(embed, galleries, manifest, thresholds, genuine)
                for name, manifest in manifests.items()
            },
        }
        for name, result in report["embedders"][backend]["swappers"].items():
            print(
                f"{backend:14s} {name:10s} donor top-1 {result['top1']['donor']}  "
                f"target top-1 {result['top1']['target']}  "
                f"donor >= high {result.get('above_high_confidence', {}).get('donor')}  "
                f"genuine-vs-swap AUC {result.get('genuine_vs_swap_auc')}"
            )

    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "identity_under_swap.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
