"""Measure published deepfake detectors on this project's own data.

A checkpoint's headline accuracy describes the dataset it was trained on. This
script asks the only question that matters here: does it separate real
photographs from manipulated ones on the material this system will actually see.

Two numbers are reported for every detector, and the second one is the one that
usually disqualifies a model:

separation
    ROC-AUC over real versus manipulated faces, with the manipulated set built
    by ``build_manipulation_set.py``.
false positive rate on genuine photographs
    What fraction of untouched photographs the detector calls synthetic at its
    own default operating point. A detector that flags a third of real photos is
    unusable in a system whose whole purpose is to avoid false accusations,
    whatever its recall.

Detectors are run through the project's own ONNX adapter on the padded face crop
the analysis pipeline produces, so the measurement matches deployment rather
than a notebook.

With ``--write`` the best qualifying detector is wired in: it becomes the
configured backend and its thresholds are marked calibrated, which is what lets
the risk engine start scoring the signal. Nothing is wired in unless a detector
clears both bars, because a detector that fails them is worse than no detector -
the risk engine already handles a missing signal correctly.

Usage:
    python scripts/evaluate_deepfake_detectors.py
    python scripts/evaluate_deepfake_detectors.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepshield.config import DeepfakeDetectorConfig, load_config
from deepshield.detection.deepfake_backends import OnnxDeepfakeDetector
from deepshield.experiments import environment
from deepshield.face.detector import build_detector
from deepshield.media import load_image
from deepshield.pipeline.analysis_pipeline import DEEPFAKE_CROP_MARGIN, crop_with_margin
from deepshield.risk.calibration import precision_recall_at, roc_curve, threshold_for_precision
from deepshield.transforms import Transformation

DEGRADATIONS = {
    "clean": ("identity", {}),
    "jpeg50": ("jpeg_compression", {"quality": 50}),
}
DEFAULT_OPERATING_POINT = 0.5
MIN_USEFUL_AUC = 0.75
MAX_TOLERABLE_FPR = 0.10


def discover_models(model_dir: Path) -> list[tuple[str, Path, dict]]:
    """Return every exported detector and its metadata."""
    found = []
    for meta_path in sorted(model_dir.glob("deepfake_*.json")):
        onnx_path = meta_path.with_suffix(".onnx")
        if onnx_path.is_file():
            found.append(
                (meta_path.stem.replace("deepfake_", ""), onnx_path,
                 json.loads(meta_path.read_text(encoding="utf-8")))
            )
    return found


def score_set(
    detector: OnnxDeepfakeDetector,
    face_detector: object,
    records: list[dict[str, str]],
    degradation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return scores and labels over the manipulation set for one condition."""
    kind, params = DEGRADATIONS[degradation]
    transformation = Transformation(degradation, kind, params)

    scores: list[float] = []
    labels: list[int] = []
    for record in records:
        path = Path(record["path"])
        if not path.is_file():
            continue
        image = load_image(path)
        if degradation != "clean":
            image = transformation.apply(image, seed=1)
        faces = face_detector.detect(image)
        if not faces:
            continue
        crop = crop_with_margin(image, faces[0], DEEPFAKE_CROP_MARGIN)
        scores.append(detector.predict_image(crop).score)
        labels.append(1 if record["label"] == "fake" else 0)
    return np.asarray(scores), np.asarray(labels)


def main(argv: list[str] | None = None) -> int:
    """Evaluate every exported detector and report whether any is usable."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/test/manipulated/manifest.json")
    )
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument(
        "--write",
        action="store_true",
        help="adopt the best qualifying detector as the configured backend",
    )
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        raise SystemExit(
            f"missing {args.manifest}; run scripts/build_manipulation_set.py first"
        )
    records = json.loads(args.manifest.read_text(encoding="utf-8"))["records"]
    models = discover_models(args.models)
    if not models:
        raise SystemExit(
            f"no exported detectors in {args.models}; run "
            "scripts/fetch_deepfake_detector.py first"
        )

    config = load_config()
    face_detector = build_detector(config.face.detector)
    print(f"manipulation set: {len(records)} images, {len(models)} detectors\n")

    results: dict[str, dict] = {}
    for alias, onnx_path, metadata in models:
        print(f"{alias}  ({metadata['repo']})", flush=True)
        detector = OnnxDeepfakeDetector(
            DeepfakeDetectorConfig(
                backend="onnx",
                model_path=onnx_path,
                model_name=metadata["repo"],
                input_size=metadata["input_size"],
                positive_index=metadata["positive_index"],
                training_dataset=metadata["repo"],
            )
        )
        per_condition: dict[str, dict] = {}
        for degradation in DEGRADATIONS:
            scores, labels = score_set(detector, face_detector, records, degradation)
            if labels.sum() == 0 or (labels == 0).sum() == 0:
                continue
            curve = roc_curve(scores, labels)
            default_point = precision_recall_at(scores, labels, DEFAULT_OPERATING_POINT)
            threshold, point = threshold_for_precision(scores, labels, args.target_precision)
            genuine_flagged = float((scores[labels == 0] >= DEFAULT_OPERATING_POINT).mean())
            per_condition[degradation] = {
                "n": int(labels.size),
                "auc": round(curve.auc, 6),
                "eer": round(curve.eer, 6),
                "fpr_at_default": round(genuine_flagged, 6),
                "recall_at_default": round(default_point["recall"], 6),
                "tuned_threshold": round(threshold, 6),
                "tuned_precision": round(point["precision"], 6),
                "tuned_recall": round(point["recall"], 6),
            }
            print(
                f"  {degradation:7s} n={labels.size:3d} AUC={curve.auc:.4f} "
                f"EER={curve.eer:.4f}  at its own 0.5 point: "
                f"recall={default_point['recall']:.3f} "
                f"false-positive rate on real photos={genuine_flagged:.3f}"
            )

        clean = per_condition.get("clean", {})
        usable = (
            float(clean.get("auc", 0.0)) >= MIN_USEFUL_AUC
            and float(clean.get("fpr_at_default", 1.0)) <= MAX_TOLERABLE_FPR
        )
        results[alias] = {
            "repo": metadata["repo"],
            "labels": metadata["labels"],
            "per_condition": per_condition,
            "usable": usable,
        }
        print(f"  verdict: {'usable' if usable else 'NOT usable'}\n")

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "deepfake_detector_survey.json"
    report_path.write_text(
        json.dumps(
            {
                "environment": environment(config),
                "manifest": str(args.manifest),
                "criteria": {
                    "min_auc": MIN_USEFUL_AUC,
                    "max_false_positive_rate": MAX_TOLERABLE_FPR,
                },
                "detectors": results,
                "caveat": (
                    "Measured on graphics-based face swaps generated by this "
                    "repository. A detector may do better or worse on GAN and "
                    "diffusion output, which this set does not contain."
                ),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    best_alias, best_result = max(
        results.items(),
        key=lambda item: item[1]["per_condition"].get("clean", {}).get("auc", 0.0),
        default=(None, None),
    )
    print(f"wrote {report_path}")
    if best_alias is None:
        return 0

    clean = best_result["per_condition"].get("clean", {})
    print(
        f"best separation: {best_alias} at AUC {clean.get('auc', 0.0):.4f} "
        f"({'usable' if best_result['usable'] else 'still NOT usable'})"
    )

    if not args.write:
        return 0
    if not best_result["usable"]:
        print(
            "\nnot adopting any detector: none reached "
            f"AUC {MIN_USEFUL_AUC} with a false positive rate under "
            f"{MAX_TOLERABLE_FPR:.0%}. The signal stays out of the risk score."
        )
        return 0

    metadata = next(meta for alias, _, meta in models if alias == best_alias)
    onnx_path = next(path for alias, path, _ in models if alias == best_alias)
    suspicious = clean["tuned_threshold"]
    high = max(suspicious, min(0.99, suspicious + (1.0 - suspicious) / 2.0))

    default_path = ROOT / "configs" / "default.yaml"
    text = default_path.read_text(encoding="utf-8")
    block = (
        "  deepfake:\n"
        "    backend: onnx\n"
        f"    model_name: {metadata['repo']}\n"
        '    model_version: "onnx-export"\n'
        f"    model_path: {onnx_path.as_posix()}\n"
        f"    training_dataset: {metadata['repo']}\n"
        f"    positive_index: {metadata['positive_index']}\n"
        f"    input_size: {metadata['input_size']}\n"
        "    batch_size: 8\n"
        "    frame_aggregation: trimmed_mean\n"
    )
    head, _, rest = text.partition("  deepfake:\n")
    _, _, tail = rest.partition("  watermark:\n")
    default_path.write_text(head + block + "  watermark:\n" + tail, encoding="utf-8")
    print(f"updated {default_path}")

    thresholds_path = ROOT / "configs" / "thresholds.yaml"
    text = thresholds_path.read_text(encoding="utf-8")
    block = (
        "deepfake:\n"
        f"  suspicious_threshold: {suspicious:.4f}\n"
        f"  high_confidence_threshold: {high:.4f}\n"
        "  calibrated: true\n"
        f"  calibration_source: {report_path.as_posix()}\n"
    )
    head, _, rest = text.partition("deepfake:")
    _, _, tail = rest.partition("\n\n")
    thresholds_path.write_text(head + block + "\n" + tail, encoding="utf-8")
    print(f"updated {thresholds_path}")
    print(
        f"\nthe deepfake signal now enters the risk score, weighted "
        f"{load_config().thresholds.risk.weights.deepfake_score:.0%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
