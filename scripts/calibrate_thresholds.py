"""Fit the identity decision thresholds on the evaluation set.

A threshold copied from a paper describes that paper's dataset. This script
measures the score distribution the configured pipeline actually produces, then
fits the two boundaries the system uses and reports the evidence behind them.

The protocol is the one the system runs at inference: a clean gallery of the
identity's other photos, a probe that may be degraded, and the configured
aggregation. Probes are degraded because suspect content is, and because on
clean frontal portraits every configuration separates perfectly and any
threshold would look justified.

Two thresholds are fitted, for two different jobs:

candidate
    What is worth spending an expensive detector on. Leans toward recall, since
    a missed candidate is never examined again.
high confidence
    What may be reported to a user as a match. Leans toward precision, since a
    false identity claim is the most damaging error this system can make.

When the two score distributions separate, both are placed relative to the
midpoint of the gap rather than on top of either distribution's edge. Putting a
boundary at the lowest genuine score looks perfect on the measured data and
leaves no headroom at all: the next genuine pair that is slightly harder falls
straight through it. The midpoint is the maximum-margin choice, and the two
thresholds are offset from it by a quarter of the gap in each direction, so one
errs toward catching things and the other toward being sure.

Usage:
    python scripts/calibrate_thresholds.py --faces data/test/eval_faces --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_face_pipeline import (
    DEGRADATIONS,
    embed_all,
    leave_one_out_scores,
    read_manifest,
)

from deepshield.config import DeepShieldConfig, load_config
from deepshield.experiments import environment
from deepshield.quality import face_quality_score
from deepshield.risk.calibration import (
    ThresholdCalibrator,
    precision_recall_at,
    roc_curve,
    threshold_for_precision,
)

DEFAULT_DEGRADATIONS = ("clean", "jpeg30", "crop_20", "blur_3", "screenshot")


def probe_quality_distribution(
    config: DeepShieldConfig,
    faces_dir: Path,
    manifest: dict[str, str],
    degradation: str = "clean",
) -> list[float]:
    """Measure the quality scores the pipeline assigns to probes."""
    from deepshield.face.aligner import build_aligner
    from deepshield.face.detector import build_detector
    from deepshield.media import load_image
    from deepshield.transforms import Transformation

    detector = build_detector(config.face.detector)
    aligner = build_aligner(config.face.aligner)
    kind, params = DEGRADATIONS[degradation]
    transformation = Transformation(degradation, kind, params)
    scores: list[float] = []
    for name in sorted(manifest):
        path = faces_dir / name
        if not path.is_file():
            continue
        image = load_image(path)
        if degradation != "clean":
            image = transformation.apply(image, seed=1)
        faces = detector.detect(image)
        if not faces:
            continue
        aligned = aligner.align(image, faces[0])
        scores.append(
            face_quality_score(
                min(faces[0].bbox.width, faces[0].bbox.height), aligned.image
            )
        )
    return scores


def main(argv: list[str] | None = None) -> int:
    """Fit both thresholds and write the calibration report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    parser.add_argument("--degradations", nargs="*", default=list(DEFAULT_DEGRADATIONS))
    parser.add_argument("--target-precision", type=float, default=0.999)
    parser.add_argument(
        "--gap-offset",
        type=float,
        default=0.25,
        help="how far from the gap midpoint to place each threshold, as a fraction",
    )
    parser.add_argument("--write", action="store_true", help="update configs/thresholds.yaml")
    args = parser.parse_args(argv)

    if not args.faces.is_dir():
        raise SystemExit(
            f"no evaluation set at {args.faces}; run scripts/build_evaluation_set.py first"
        )

    config = load_config()
    manifest = read_manifest(args.faces)
    identities = len(set(manifest.values()))
    print(f"evaluation set: {len(manifest)} photos, {identities} identities")
    print(
        f"pipeline: {config.face.detector.backend} / {config.face.embedder.backend}"
        f"{'+tta' if config.face.embedder.flip_tta else ''} / {config.face.matcher.aggregation}"
    )

    gallery, failures, per_image = embed_all(config, args.faces, manifest)
    print(f"embedded gallery: {len(gallery)} photos, {len(failures)} failures, "
          f"{per_image:.3f}s per image")

    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    degradation_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    degradation_quality: dict[str, float] = {}
    per_degradation: dict[str, dict[str, float]] = {}

    for degradation in args.degradations:
        if degradation not in DEGRADATIONS:
            print(f"  skipping unknown degradation '{degradation}'")
            continue
        probes = (
            gallery
            if degradation == "clean"
            else embed_all(config, args.faces, manifest, degradation)[0]
        )
        if not probes:
            print(f"  {degradation:12s} no probe survived detection; skipped")
            per_degradation[degradation] = {"probes": 0.0}
            continue
        scores, labels = leave_one_out_scores(
            gallery, probes, manifest, config.face.matcher.aggregation,
            config.face.matcher.top_k,
        )
        curve = roc_curve(scores, labels)
        genuine = scores[labels == 1]
        impostor = scores[labels == 0]
        per_degradation[degradation] = {
            "probes": float(len(probes)),
            "auc": round(curve.auc, 6),
            "eer": round(curve.eer, 6),
            "genuine_min": round(float(genuine.min()), 6),
            "genuine_median": round(float(np.median(genuine)), 6),
            "impostor_max": round(float(impostor.max()), 6),
        }
        print(
            f"  {degradation:12s} probes={len(probes):3d} AUC={curve.auc:.4f} "
            f"EER={curve.eer:.4f}  genuine min={genuine.min():.3f}  "
            f"impostor max={impostor.max():.3f}"
        )
        all_scores.append(scores)
        all_labels.append(labels)
        degradation_scores[degradation] = (scores, labels)

    if not all_scores:
        raise SystemExit("no degradation produced usable comparisons")

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    genuine = scores[labels == 1]
    impostor = scores[labels == 0]

    curve = roc_curve(scores, labels)
    candidate = ThresholdCalibrator("max_margin").calibrate(
        scores, labels, independent_pairs=identities
    )

    gap = float(genuine.min()) - float(impostor.max())
    separable = gap > 0
    if separable:
        midpoint = candidate.threshold
        offset = gap * args.gap_offset
        candidate_threshold = midpoint - offset
        high_threshold = midpoint + offset
        placement = (
            f"gap [{impostor.max():.4f}, {genuine.min():.4f}] width {gap:.4f}; "
            f"placed at the midpoint -/+ {args.gap_offset:.0%} of the gap"
        )
    else:
        candidate_threshold = candidate.threshold
        high_threshold, _ = threshold_for_precision(scores, labels, args.target_precision)
        placement = (
            "score distributions overlap; candidate placed at the equal-error rate "
            f"and high confidence at the lowest threshold reaching precision "
            f"{args.target_precision}"
        )

    candidate_point = precision_recall_at(scores, labels, candidate_threshold)
    high_point = precision_recall_at(scores, labels, high_threshold)

    margins: list[float] = []
    by_identity: dict[str, list[str]] = {}
    for name in gallery:
        by_identity.setdefault(manifest[name], []).append(name)
    for probe_name, probe in gallery.items():
        ranked = sorted(
            (
                float(
                    np.max(
                        np.vstack([gallery[g] for g in members if g != probe_name]) @ probe
                    )
                )
                for identity, members in by_identity.items()
                if [g for g in members if g != probe_name]
            ),
            reverse=True,
        )
        if len(ranked) > 1:
            margins.append(ranked[0] - ranked[1])
    margin_array = np.asarray(margins)

    quality = np.asarray(probe_quality_distribution(config, args.faces, manifest))

    for degradation in degradation_scores:
        probe_quality = probe_quality_distribution(
            config, args.faces, manifest, degradation
        )
        degradation_quality[degradation] = (
            float(np.median(probe_quality)) if probe_quality else 0.0
        )

    print(f"\npooled over {len(args.degradations)} conditions: "
          f"{int(labels.sum())} genuine, {int((labels == 0).sum())} impostor")
    print(f"ROC-AUC {curve.auc:.4f}   EER {curve.eer:.4f}")
    print(f"genuine  min={genuine.min():.4f} p1={np.percentile(genuine, 1):.4f} "
          f"median={np.median(genuine):.4f}")
    print(f"impostor max={impostor.max():.4f} p99={np.percentile(impostor, 99):.4f}")
    print(f"\nplacement: {placement}")
    print(f"candidate threshold       {candidate_threshold:.4f}  "
          f"(recall {candidate_point['recall']:.4f}, "
          f"{int(candidate_point['false_positives'])} false positives)")
    print(f"high-confidence threshold {high_threshold:.4f}  "
          f"(precision {high_point['precision']:.4f}, recall {high_point['recall']:.4f}, "
          f"{int(high_point['false_positives'])} false positives)")
    print(f"top-1 margin: min={margin_array.min():.4f} p1={np.percentile(margin_array, 1):.4f}")
    print(f"probe quality: min={quality.min():.3f} p5={np.percentile(quality, 5):.3f} "
          f"median={np.median(quality):.3f}")

    penalty = config.thresholds.face_similarity.low_quality_penalty
    print(f"\ncost of the quality guard (penalty {penalty:.2f}) at the "
          f"high-confidence threshold:")
    guard_cost: dict[str, dict[str, float]] = {}
    for degradation, (scores_d, labels_d) in degradation_scores.items():
        median_quality = degradation_quality.get(degradation, 1.0)
        raised = min(1.0, high_threshold + penalty * max(0.0, 1.0 - median_quality))
        plain = precision_recall_at(scores_d, labels_d, high_threshold)
        guarded = precision_recall_at(scores_d, labels_d, raised)
        guard_cost[degradation] = {
            "median_probe_quality": round(median_quality, 4),
            "raised_threshold": round(raised, 4),
            "recall_without_guard": round(plain["recall"], 4),
            "recall_with_guard": round(guarded["recall"], 4),
            "false_positives_without_guard": plain["false_positives"],
            "false_positives_with_guard": guarded["false_positives"],
        }
        print(
            f"  {degradation:12s} quality={median_quality:.3f} thr {high_threshold:.3f}"
            f"->{raised:.3f}  recall {plain['recall']:.3f}->{guarded['recall']:.3f}  "
            f"FP {int(plain['false_positives'])}->{int(guarded['false_positives'])}"
        )
    for note in candidate.notes:
        print(f"  note: {note}")

    report = {
        "environment": environment(config),
        "evaluation_set": {
            "path": str(args.faces),
            "photos": len(manifest),
            "identities": identities,
            "degradations": args.degradations,
        },
        "pooled": {
            "genuine": int(labels.sum()),
            "impostor": int((labels == 0).sum()),
            "roc": curve.to_dict(),
            "genuine_min": round(float(genuine.min()), 6),
            "impostor_max": round(float(impostor.max()), 6),
        },
        "per_degradation": per_degradation,
        "placement": placement,
        "separable": separable,
        "gap": round(gap, 6),
        "candidate": {
            **candidate.to_dict(),
            "threshold": round(candidate_threshold, 6),
            "operating_point": candidate_point,
        },
        "high_confidence": {
            "threshold": round(high_threshold, 6),
            "criterion": placement,
            "operating_point": high_point,
        },
        "margin": {
            "min": round(float(margin_array.min()), 6),
            "p1": round(float(np.percentile(margin_array, 1)), 6),
            "median": round(float(np.median(margin_array)), 6),
        },
        "quality_guard": guard_cost,
        "probe_quality": {
            "min": round(float(quality.min()), 6),
            "p5": round(float(np.percentile(quality, 5)), 6),
            "median": round(float(np.median(quality)), 6),
        },
        "caveat": (
            "Comparisons within one identity are not independent, and the evaluation "
            "set is drawn from a benchmark skewed by demographic and pose. Refit on "
            "data resembling the deployment population before relying on these values."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "calibration_face_similarity.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {report_path}")

    if args.write:
        thresholds_path = ROOT / "configs" / "thresholds.yaml"
        text = thresholds_path.read_text(encoding="utf-8")
        block = (
            "face_similarity:\n"
            f"  candidate_threshold: {candidate_threshold:.4f}\n"
            f"  high_confidence_threshold: {high_threshold:.4f}\n"
            f"  min_margin: {config.thresholds.face_similarity.min_margin:.4f}\n"
            f"  low_quality_penalty: "
            f"{config.thresholds.face_similarity.low_quality_penalty:.4f}\n"
            f"  min_probe_face_pixels: "
            f"{config.thresholds.face_similarity.min_probe_face_pixels}\n"
            "  calibrated: true\n"
            f"  calibration_source: {report_path.as_posix()}\n"
        )
        head, _, rest = text.partition("face_similarity:")
        _, _, tail = rest.partition("\n\n")
        thresholds_path.write_text(head + block + "\n" + tail, encoding="utf-8")
        print(f"updated {thresholds_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
