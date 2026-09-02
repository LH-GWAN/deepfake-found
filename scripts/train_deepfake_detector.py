"""Fit and evaluate the blending-artefact deepfake detector.

The synthetic-media signal is the one piece of evidence DeepShield computes but
does not score, because no threshold had ever been fitted for it. This script
fits one, on the manipulation set built by ``build_manipulation_set.py``, and
reports the operating characteristics that decide whether the signal may enter
the risk score at all.

Two rules keep the result honest:

identity-disjoint splits
    No identity appears in both the training and the test half. Faces of the
    same person share pose, lighting and camera, so a random split would let the
    model memorise people rather than learn manipulation, and would report an
    accuracy the detector does not have.
features come from the same crop the pipeline uses
    The analysis pipeline hands the detector a face crop with a margin, so
    training on whole photographs would fit a different input distribution than
    the one the detector sees in production.
degraded evaluation
    The test half is also scored after JPEG re-encoding and rescaling, because
    suspect content arrives compressed and blending traces are exactly what
    compression erodes.

If the held-out result is not clearly better than chance, the threshold stays
uncalibrated and the signal stays out of the risk score. That is a real possible
outcome of running this script, not a failure of it.

Usage:
    python scripts/train_deepfake_detector.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepshield.config import load_config
from deepshield.detection.blending import FEATURE_NAMES, extract_features
from deepshield.experiments import environment
from deepshield.media import load_image
from deepshield.risk.calibration import roc_curve, threshold_for_precision
from deepshield.transforms import Transformation

MIN_USEFUL_AUC = 0.75
DEGRADATIONS = {
    "clean": ("identity", {}),
    "jpeg50": ("jpeg_compression", {"quality": 50}),
    "resize50": ("resize", {"scale": 0.5}),
}


def load_dataset(manifest_path: Path) -> list[dict[str, str]]:
    """Read the manipulation-set manifest."""
    if not manifest_path.is_file():
        raise SystemExit(
            f"missing {manifest_path}; run scripts/build_manipulation_set.py first"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))["records"]


def build_matrix(
    records: list[dict[str, str]], degradation: str, detector: object
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return the feature matrix, labels and identities for one condition.

    Features are measured on the padded face crop the analysis pipeline produces,
    not on the whole photograph, so the training distribution matches the one the
    detector will see.
    """
    from deepshield.pipeline.analysis_pipeline import DEEPFAKE_CROP_MARGIN, crop_with_margin

    kind, params = DEGRADATIONS[degradation]
    transformation = Transformation(degradation, kind, params)

    rows: list[np.ndarray] = []
    labels: list[int] = []
    identities: list[str] = []
    for record in records:
        path = Path(record["path"])
        if not path.is_file():
            continue
        image = load_image(path)
        if degradation != "clean":
            image = transformation.apply(image, seed=1)
        faces = detector.detect(image)
        if not faces:
            continue
        crop = crop_with_margin(image, faces[0], DEEPFAKE_CROP_MARGIN)
        try:
            rows.append(extract_features(crop))
        except Exception:
            continue
        labels.append(1 if record["label"] == "fake" else 0)
        identities.append(record["identity"])
    return np.vstack(rows), np.asarray(labels), identities


def fit_logistic(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Standardise the features and fit a logistic regression."""
    from sklearn.linear_model import LogisticRegression

    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardised = (features - mean) / scale

    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(standardised, labels)
    return mean, scale, model.coef_.ravel(), float(model.intercept_[0])


def score(
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
) -> np.ndarray:
    """Return the predicted synthetic probability for every row."""
    logits = ((features - mean) / scale) @ coefficients + intercept
    return 1.0 / (1.0 + np.exp(-logits))


def main(argv: list[str] | None = None) -> int:
    """Fit the detector, evaluate it out of sample and write the model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/test/manipulated/manifest.json")
    )
    parser.add_argument("--model-out", type=Path, default=Path("models/blending_detector.json"))
    parser.add_argument("--report-out", type=Path, default=Path("data/results"))
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--test-fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write", action="store_true", help="update configs/thresholds.yaml")
    args = parser.parse_args(argv)

    config = load_config()
    records = load_dataset(args.manifest)
    print(f"manipulation set: {len(records)} images")

    from deepshield.face.detector import build_detector

    detector = build_detector(config.face.detector)
    conditions: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for degradation in DEGRADATIONS:
        conditions[degradation] = build_matrix(records, degradation, detector)
        features, labels, _ = conditions[degradation]
        print(f"  {degradation:9s} {features.shape[0]} rows, {int(labels.sum())} manipulated")

    features, labels, identities = conditions["clean"]
    unique = sorted(set(identities))
    rng = np.random.default_rng(args.seed)
    shuffled = list(rng.permutation(unique))
    cut = max(1, int(len(shuffled) * args.test_fraction))
    test_identities = set(shuffled[:cut])
    print(
        f"\nidentity-disjoint split: {len(unique) - len(test_identities)} train, "
        f"{len(test_identities)} test identities"
    )

    membership = np.asarray([identity in test_identities for identity in identities])
    mean, scale, coefficients, intercept = fit_logistic(
        features[~membership], labels[~membership]
    )

    print("\nlearned weights, standardised units:")
    for name, weight in sorted(
        zip(FEATURE_NAMES, coefficients, strict=True), key=lambda kv: -abs(kv[1])
    ):
        print(f"  {name:22s} {weight:+.3f}")

    per_condition: dict[str, dict[str, float]] = {}
    print(f"\nheld-out results at precision >= {args.target_precision:.2f}:")
    for degradation, (matrix, condition_labels, condition_identities) in conditions.items():
        held = np.asarray([i in test_identities for i in condition_identities])
        if held.sum() == 0:
            continue
        probabilities = score(matrix[held], mean, scale, coefficients, intercept)
        truth = condition_labels[held]
        if truth.sum() == 0 or (truth == 0).sum() == 0:
            continue
        curve = roc_curve(probabilities, truth)
        threshold, point = threshold_for_precision(
            probabilities, truth, args.target_precision
        )
        per_condition[degradation] = {
            "rows": int(held.sum()),
            "auc": round(curve.auc, 6),
            "eer": round(curve.eer, 6),
            "threshold": round(threshold, 6),
            "precision": round(point["precision"], 6),
            "recall": round(point["recall"], 6),
            "false_positives": int(point["false_positives"]),
        }
        print(
            f"  {degradation:9s} n={int(held.sum()):3d} AUC={curve.auc:.4f} "
            f"EER={curve.eer:.4f} thr={threshold:.4f} "
            f"P={point['precision']:.3f} R={point['recall']:.3f}"
        )

    clean = per_condition.get("clean", {})
    usable = float(clean.get("auc", 0.0)) >= MIN_USEFUL_AUC
    pooled_probabilities = []
    pooled_labels = []
    for matrix, condition_labels, condition_identities in conditions.values():
        held = np.asarray([i in test_identities for i in condition_identities])
        pooled_probabilities.append(score(matrix[held], mean, scale, coefficients, intercept))
        pooled_labels.append(condition_labels[held])
    probabilities = np.concatenate(pooled_probabilities)
    truth = np.concatenate(pooled_labels)
    pooled_curve = roc_curve(probabilities, truth)
    suspicious, suspicious_point = threshold_for_precision(
        probabilities, truth, args.target_precision
    )
    high, high_point = threshold_for_precision(probabilities, truth, 0.99)

    print(f"\npooled over conditions: AUC={pooled_curve.auc:.4f} EER={pooled_curve.eer:.4f}")
    print(
        f"  suspicious threshold {suspicious:.4f}  "
        f"P={suspicious_point['precision']:.3f} R={suspicious_point['recall']:.3f}"
    )
    print(
        f"  high-confidence      {high:.4f}  "
        f"P={high_point['precision']:.3f} R={high_point['recall']:.3f}"
    )
    print(f"\nverdict: {'usable' if usable else 'NOT usable'} "
          f"(clean AUC {clean.get('auc', 0.0):.4f}, floor {MIN_USEFUL_AUC})")

    trained_on = json.loads(args.manifest.read_text(encoding="utf-8")).get("method", "unknown")
    model = {
        "version": "0.1.0",
        "features": list(FEATURE_NAMES),
        "mean": [float(v) for v in mean],
        "scale": [float(v) for v in scale],
        "coefficients": [float(v) for v in coefficients],
        "intercept": intercept,
        "trained_on": trained_on,
        "covers": ["graphics-based face swap"],
        "does_not_cover": ["GAN synthesis", "diffusion synthesis", "reenactment"],
        "usable": usable,
        "usable_floor": MIN_USEFUL_AUC,
        "metrics": {
            "pooled_auc": round(pooled_curve.auc, 6),
            "clean_auc": float(clean.get("auc", 0.0)),
            "per_condition": per_condition,
        },
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"\nwrote {args.model_out}")

    report = {
        "environment": environment(config),
        "manifest": str(args.manifest),
        "train_identities": sorted(set(unique) - test_identities),
        "test_identities": sorted(test_identities),
        "per_condition": per_condition,
        "pooled": {
            "auc": round(pooled_curve.auc, 6),
            "eer": round(pooled_curve.eer, 6),
            "suspicious": {"threshold": round(suspicious, 6), **suspicious_point},
            "high_confidence": {"threshold": round(high, 6), **high_point},
        },
        "usable": usable,
        "caveat": (
            "Fitted on graphics-based face swaps generated by this repository. It "
            "says nothing about GAN or diffusion output, and the swaps carry visible "
            "artefacts, so these numbers are an upper bound on real-world performance."
        ),
    }
    args.report_out.mkdir(parents=True, exist_ok=True)
    report_path = args.report_out / "calibration_deepfake.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {report_path}")

    if args.write:
        if not usable:
            print("\nrefusing to mark the threshold calibrated: held-out AUC is too low")
            return 0
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
