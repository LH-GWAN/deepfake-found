r"""Split a deepfake detector's false alarms into face size and compression.

GenD cleared the adoption gate's false-alarm bar on clean and JPEG photographs
and failed on one condition, ``video_small``: the photograph shrunk to 0.55 of
its size (a face of about 55 pixels) and re-encoded as H.264 at crf 35, where it
called nearly half the genuine photographs synthetic. That condition moves two
things at once. This script scores the gate's own items over a grid of the two,
downscale 1.0, 0.75 and 0.55 against no re-encode, crf 23 and crf 35, at the
threshold the gate fits: the 99.5th percentile of the calibration half's clean
genuine scores.

It then pools every condition's test photographs by the width of the face that
was scored. If the false alarms follow face width, a detector can be adopted for
faces above a size and abstain below it, a rule the pipeline can apply because
it knows every face's size. If they follow compression at any size, no rule on
the face alone can scope the detector.

Items, halves and the threshold come from ``evaluate_deepfake_detectors.py``
itself, including its fix that counts the genuine photographs the manipulation
sets share only once. Scores are cached per condition under ``--work``, so an
interrupted run resumes.

Usage:
    python scripts/evaluate_detector_conditions.py \
        --manifest data/test/manipulated/manifest.json \
                   data/test/manipulated_inswapper/manifest.json \
        --genuine data/sklearn/lfw_home/lfw_funneled --only gend --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_deepfake_detectors as gate

from deepshield.config import DeepfakeDetectorConfig, load_config
from deepshield.detection.deepfake_backends import OnnxDeepfakeDetector
from deepshield.experiments import environment
from deepshield.face.detector import build_detector
from deepshield.media import load_image
from deepshield.pipeline.analysis_pipeline import DEEPFAKE_CROP_MARGIN, crop_with_margin
from deepshield.risk.calibration import clopper_pearson_upper, roc_curve
from deepshield.transforms import Transformation

GRID: dict[str, tuple[str, dict[str, Any]]] = {
    "clean": ("identity", {}),
    "down75": ("downscale", {"scale": 0.75}),
    "down55": ("downscale", {"scale": 0.55}),
    "crf23": ("video_compression", {"scale": 1.0, "crf": 23}),
    "crf35": ("video_compression", {"scale": 1.0, "crf": 35}),
    "down75_crf35": ("video_compression", {"scale": 0.75, "crf": 35}),
    "down55_crf23": ("video_compression", {"scale": 0.55, "crf": 23}),
    "down55_crf35": ("video_compression", {"scale": 0.55, "crf": 35}),
}
WIDTH_EDGES = (0, 48, 64, 80, 96, 10_000)
SIZE_RULES = (64, 80, 96)


def rate(alarms: int, trials: int) -> dict[str, Any]:
    """Return a false-positive count with its rate and 95% Clopper-Pearson upper bound."""
    return {
        "n": trials,
        "false_positives": alarms,
        "rate": round(alarms / trials, 6) if trials else None,
        "upper_95": round(clopper_pearson_upper(alarms, trials), 6) if trials else None,
    }


def signature(items: list[dict[str, Any]]) -> str:
    """Identify the item list so a cache written for other items is never reused."""
    return hashlib.sha256("\n".join(item["path"] for item in items).encode()).hexdigest()[:16]


def score_condition(
    detector: OnnxDeepfakeDetector, face_detector: Any, items: list[dict[str, Any]], name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return each item's score and scored face width under one condition.

    The same steps as the gate's ``score_items``: the transformation with seed
    1, the largest detected face, the pipeline's crop margin. ``nan`` where no
    face is found.
    """
    kind, params = GRID[name]
    transformation = Transformation(name, kind, params)
    scores = np.full(len(items), np.nan)
    widths = np.full(len(items), np.nan)
    for index, item in enumerate(items):
        path = Path(item["path"])
        if not path.is_file():
            continue
        image = load_image(path)
        if name != "clean":
            image = transformation.apply(image, seed=1)
        faces = face_detector.detect(image)
        if not faces:
            continue
        face = max(faces, key=lambda f: f.bbox.width * f.bbox.height)
        widths[index] = float(min(face.bbox.width, face.bbox.height))
        scores[index] = detector.predict_image(
            crop_with_margin(image, face, DEEPFAKE_CROP_MARGIN)
        ).score
        if (index + 1) % 500 == 0:
            print(f"    {name}: {index + 1}/{len(items)}", flush=True)
    return scores, widths


def summarise(
    items: list[dict[str, Any]], scores: dict[str, np.ndarray], widths: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Judge every condition and every face-width band at the gate's threshold."""
    labels = np.asarray([item["label"] == "fake" for item in items])
    halves = np.asarray([item["half"] for item in items])
    family = np.asarray([item["family"] for item in items])
    shared_real = np.asarray(
        [item["label"] != "fake" and item["manifest"] != "genuine" for item in items]
    )
    families = sorted({item["family"] for item in items if item["label"] == "fake"})
    clean = scores["clean"]
    fitting = clean[(halves == "calibration") & ~labels & np.isfinite(clean)]
    threshold = float(np.quantile(fitting, gate.GENUINE_QUANTILE))
    test = halves == "test"

    report: dict[str, Any] = {
        "threshold": round(threshold, 6),
        "calibration_genuine": int(fitting.size),
        "conditions": {},
        "by_face_width": {},
        "size_rules": {},
    }
    for name, values in scores.items():
        found = np.isfinite(values)
        genuine = test & ~labels & found
        entry: dict[str, Any] = {
            "median_genuine_face_width": round(float(np.nanmedian(widths[name][genuine])), 1),
            "no_face": int((test & ~found).sum()),
            "genuine": rate(int((values[genuine] >= threshold).sum()), int(genuine.sum())),
            "families": {},
        }
        for fam in families:
            fakes = test & labels & (family == fam) & found
            pooled = ((labels & (family == fam)) | shared_real) & found
            auc = roc_curve(values[pooled], labels[pooled].astype(int)).auc
            entry["families"][fam] = {
                "test_fakes": int(fakes.sum()),
                "recall": round(float((values[fakes] >= threshold).mean()), 4)
                if fakes.any() else None,
                "separation_auc": round(float(auc), 4),
            }
        report["conditions"][name] = entry

    pooled_scores = np.concatenate([scores[name] for name in scores])
    pooled_widths = np.concatenate([widths[name] for name in scores])
    pooled_labels = np.tile(labels, len(scores))
    pooled_test = np.tile(test, len(scores))
    pooled_family = np.tile(family, len(scores))
    found = np.isfinite(pooled_scores)

    def judged(mask: np.ndarray) -> dict[str, Any]:
        genuine = mask & pooled_test & ~pooled_labels & found
        entry: dict[str, Any] = {
            "genuine": rate(int((pooled_scores[genuine] >= threshold).sum()), int(genuine.sum()))
        }
        for fam in families:
            fakes = mask & pooled_test & pooled_labels & (pooled_family == fam) & found
            entry[f"{fam}_recall"] = (
                round(float((pooled_scores[fakes] >= threshold).mean()), 4) if fakes.any() else None
            )
            entry[f"{fam}_fakes"] = int(fakes.sum())
        return entry

    for low, high in zip(WIDTH_EDGES[:-1], WIDTH_EDGES[1:], strict=True):
        band = (pooled_widths >= low) & (pooled_widths < high)
        report["by_face_width"][f"{low}-{high if high < 10_000 else ''}"] = judged(band)
    for minimum in SIZE_RULES:
        kept = pooled_widths >= minimum
        entry = judged(kept)
        scored = pooled_test & found
        entry["abstains_on"] = round(float((~kept & scored).sum() / max(int(scored.sum()), 1)), 4)
        report["size_rules"][f"face_at_least_{minimum}px"] = entry
    return report


def main(argv: list[str] | None = None) -> int:
    """Score the grid, cache each condition, and write the split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--genuine", type=Path, required=True)
    parser.add_argument("--genuine-limit", type=int, default=3000)
    parser.add_argument("--faces", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--only", default="gend")
    parser.add_argument("--conditions", nargs="+", choices=sorted(GRID), default=list(GRID))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--work", type=Path, default=Path("data/results/detector_conditions_cache"))
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)

    if "clean" not in args.conditions:
        args.conditions.insert(0, "clean")
    records = gate.load_records(args.manifest)
    excluded = {person for record in records for person in record["people"]}
    listing = args.faces / "identities.txt"
    if listing.is_file():
        excluded |= {
            line.split("\t")[1].strip()
            for line in listing.read_text(encoding="utf-8").splitlines() if "\t" in line
        }
    items = list(records) + gate.genuine_photos(args.genuine, excluded, args.genuine_limit,
                                                args.seed)
    gate.assign_halves(items, gate.CALIBRATION_SHARE, args.seed)
    key = signature(items)

    models = [model for model in gate.discover_models(args.models) if model[0] == args.only]
    if not models:
        raise SystemExit(f"no exported detector '{args.only}' in {args.models}")
    alias, onnx_path, metadata = models[0]
    config = load_config()
    detector = OnnxDeepfakeDetector(
        DeepfakeDetectorConfig(
            backend="onnx", model_path=onnx_path, model_name=metadata["repo"],
            input_size=metadata["input_size"], positive_index=metadata["positive_index"],
            training_dataset=metadata["repo"],
        ),
        device=args.device or config.runtime.device,
    )
    face_detector = build_detector(config.face.detector)
    print(f"{len(items)} images, {alias} on {detector.execution_provider}", flush=True)

    scores: dict[str, np.ndarray] = {}
    widths: dict[str, np.ndarray] = {}
    args.work.mkdir(parents=True, exist_ok=True)
    for name in args.conditions:
        cache = args.work / f"{alias}_{name}_{key}.npz"
        if cache.is_file():
            saved = np.load(cache)
            scores[name], widths[name] = saved["scores"], saved["widths"]
            print(f"  {name}: cached", flush=True)
            continue
        print(f"  {name}: scoring", flush=True)
        scores[name], widths[name] = score_condition(detector, face_detector, items, name)
        np.savez(cache, scores=scores[name], widths=widths[name])

    report = {
        "question": "do a detector's false alarms follow face size or compression?",
        "detector": alias,
        "repo": metadata["repo"],
        "execution_provider": detector.execution_provider,
        "images": len(items),
        "grid": {name: {"kind": GRID[name][0], **GRID[name][1]} for name in args.conditions},
        "fp_bound": gate.MAX_FPR_BOUND,
        "min_family_recall": gate.MIN_FAMILY_RECALL,
        **summarise(items, scores, widths),
        "environment": environment(config),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"deepfake_detector_conditions_{alias}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: report["conditions"][name]["genuine"] for name in report["conditions"]},
                     indent=1))
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
