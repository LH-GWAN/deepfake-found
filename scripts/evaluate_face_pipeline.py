"""Measure identity-matching precision across face pipeline configurations.

The system's actual decision is not "are these two photos the same person" but
"does this probe face match an enrolled gallery of several photos". This script
evaluates that decision directly, under a leave-one-out protocol:

for every photo, the remaining photos of the same identity form the genuine
gallery, every other identity forms an impostor gallery, and the configured
aggregation reduces each comparison to one score.

Precision is the headline metric rather than accuracy. Telling someone their
face appears in synthetic content when it does not is far more damaging than
missing one instance that a later scan can still catch, and on a realistic
population impostor comparisons vastly outnumber genuine ones, so accuracy would
look excellent for a matcher that never matches anything. Recall at the chosen
operating point is reported beside it so the cost of the choice is visible.

Probes are degraded before matching while the gallery stays clean. That mirrors
reality - enrollment photos are chosen by the user, suspect content arrives
re-encoded, rescaled, cropped or screenshotted - and it is also the only way to
tell configurations apart: on clean frontal portraits every backend here reaches
a perfect AUC and the comparison is vacuous.

Comparisons within one identity are also not independent, so the number of
identities and photos is reported with every result and treated as the real
limit on how much any of it can be trusted.

Usage:
    python scripts/evaluate_face_pipeline.py --faces data/test/eval_faces
    python scripts/evaluate_face_pipeline.py --sweep embedder --target-precision 0.99
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from deepshield.config import DeepShieldConfig, load_config
from deepshield.experiments import environment
from deepshield.face.aligner import build_aligner
from deepshield.face.detector import build_detector
from deepshield.face.embedder import build_embedder
from deepshield.media import IMAGE_SUFFIXES, load_image
from deepshield.risk.calibration import (
    precision_recall_at,
    roc_curve,
    threshold_for_precision,
)
from deepshield.transforms import Transformation

AGGREGATIONS = ("max", "mean", "topk_mean", "centroid")

DEGRADATIONS: dict[str, tuple[str, dict[str, Any]]] = {
    "clean": ("identity", {}),
    "jpeg30": ("jpeg_compression", {"quality": 30}),
    "downscale_25": ("downscale", {"scale": 0.25}),
    "crop_20": ("crop", {"ratio": 0.2}),
    "blur_3": ("blur", {"sigma": 3.0}),
    "screenshot": ("screenshot_simulation", {"scale": 0.6, "quality": 60}),
}


@dataclass
class PipelineVariant:
    """One configuration of the face pipeline to evaluate."""

    detector: str
    embedder: str
    flip_tta: bool
    ensemble: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Short human-readable name used in reports."""
        name = "+".join(self.ensemble) if self.ensemble else self.embedder
        return f"{self.detector}/{name}{'+tta' if self.flip_tta else ''}"

    def apply(self, config: DeepShieldConfig) -> DeepShieldConfig:
        """Return a configuration with this variant's backends selected."""
        return config.model_copy(
            update={
                "face": config.face.model_copy(
                    update={
                        "detector": config.face.detector.model_copy(
                            update={"backend": self.detector}
                        ),
                        "embedder": config.face.embedder.model_copy(
                            update={
                                "backend": self.embedder,
                                "flip_tta": self.flip_tta,
                                "ensemble": list(self.ensemble),
                            }
                        ),
                    }
                )
            }
        )


@dataclass
class VariantResult:
    """Metrics for one variant, degradation and aggregation strategy."""

    variant: str
    degradation: str
    aggregation: str
    auc: float
    eer: float
    genuine: int
    impostor: int
    identities: int
    photos: int
    detection_failures: int
    probe_detection_failures: int
    seconds_per_image: float
    operating_points: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        """Flatten to one CSV row."""
        row: dict[str, Any] = {
            "variant": self.variant,
            "degradation": self.degradation,
            "aggregation": self.aggregation,
            "roc_auc": round(self.auc, 6),
            "eer": round(self.eer, 6),
            "genuine_comparisons": self.genuine,
            "impostor_comparisons": self.impostor,
            "identities": self.identities,
            "photos": self.photos,
            "detection_failures": self.detection_failures,
            "probe_detection_failures": self.probe_detection_failures,
            "seconds_per_image": round(self.seconds_per_image, 4),
        }
        for name, point in self.operating_points.items():
            row[f"{name}_threshold"] = round(point["threshold"], 6)
            row[f"{name}_precision"] = round(point["precision"], 6)
            row[f"{name}_recall"] = round(point["recall"], 6)
            row[f"{name}_false_positives"] = int(point["false_positives"])
        return row


def read_manifest(faces_dir: Path) -> dict[str, str]:
    """Read the sample-to-identity mapping, or infer it from filename prefixes."""
    manifest = faces_dir / "identities.txt"
    if manifest.is_file():
        mapping = {}
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                name, identity = line.split("\t")
                mapping[name] = identity
        return mapping
    return {
        path.name: path.stem.rsplit("_", 1)[0]
        for path in sorted(faces_dir.iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES
    }


def embed_all(
    config: DeepShieldConfig,
    faces_dir: Path,
    manifest: dict[str, str],
    degradation: str = "clean",
    seed: int = 1,
) -> tuple[dict[str, np.ndarray], list[str], float]:
    """Embed every labelled photo, optionally degraded first.

    Degradation is applied before detection, not after alignment, so a
    transformation severe enough to break detection is recorded as a detection
    failure rather than silently producing a low similarity. The two failures
    have different causes and different fixes.
    """
    detector = build_detector(config.face.detector)
    aligner = build_aligner(config.face.aligner)
    embedder = build_embedder(config.face.embedder)

    kind, params = DEGRADATIONS[degradation]
    transformation = Transformation(degradation, kind, params)

    vectors: dict[str, np.ndarray] = {}
    failures: list[str] = []
    started = time.perf_counter()

    for name in sorted(manifest):
        path = faces_dir / name
        if not path.is_file():
            failures.append(f"{name}: missing")
            continue
        image = load_image(path)
        if degradation != "clean":
            image = transformation.apply(image, seed=seed)
        faces = detector.detect(image)
        if not faces:
            failures.append(f"{name}: no face detected")
            continue
        crop = aligner.align(image, faces[0]).image
        vectors[name] = embedder.embed(crop).vector

    elapsed = time.perf_counter() - started
    per_image = elapsed / max(len(manifest), 1)
    return vectors, failures, per_image


def aggregate(similarities: np.ndarray, strategy: str, top_k: int) -> float:
    """Reduce per-reference similarities to one gallery score."""
    if strategy == "max":
        return float(similarities.max())
    if strategy == "mean":
        return float(similarities.mean())
    if strategy == "topk_mean":
        k = min(top_k, similarities.size)
        return float(np.sort(similarities)[-k:].mean())
    raise ValueError(f"unsupported aggregation '{strategy}'")


def leave_one_out_scores(
    gallery_vectors: dict[str, np.ndarray],
    probe_vectors: dict[str, np.ndarray],
    manifest: dict[str, str],
    strategy: str,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every degraded probe against every clean identity gallery."""
    by_identity: dict[str, list[str]] = {}
    for name in gallery_vectors:
        by_identity.setdefault(manifest[name], []).append(name)

    scores: list[float] = []
    labels: list[int] = []

    for probe_name, probe in probe_vectors.items():
        probe_identity = manifest[probe_name]
        for identity, members in by_identity.items():
            gallery = [name for name in members if name != probe_name]
            if not gallery:
                continue
            matrix = np.vstack([gallery_vectors[name] for name in gallery])
            if strategy == "centroid":
                centroid = matrix.mean(axis=0)
                centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
                score = float(np.dot(probe, centroid))
            else:
                score = aggregate(matrix @ probe, strategy, top_k)
            scores.append(score)
            labels.append(int(identity == probe_identity))

    return np.asarray(scores), np.asarray(labels)


def evaluate_variant(
    config: DeepShieldConfig,
    variant: PipelineVariant,
    faces_dir: Path,
    manifest: dict[str, str],
    aggregations: tuple[str, ...],
    degradations: tuple[str, ...],
    target_precision: float,
) -> list[VariantResult]:
    """Embed a clean gallery once, then score each degradation and aggregation."""
    variant_config = variant.apply(config)
    gallery, gallery_failures, per_image = embed_all(variant_config, faces_dir, manifest)
    identities = len({manifest[name] for name in gallery})

    results: list[VariantResult] = []
    for degradation in degradations:
        if degradation == "clean":
            probes, probe_failures = gallery, gallery_failures
        else:
            probes, probe_failures, _ = embed_all(
                variant_config, faces_dir, manifest, degradation
            )

        for strategy in aggregations:
            scores, labels = leave_one_out_scores(
                gallery, probes, manifest, strategy, variant_config.face.matcher.top_k
            )
            if labels.sum() == 0 or (labels == 0).sum() == 0:
                continue
            curve = roc_curve(scores, labels)
            threshold, point = threshold_for_precision(scores, labels, target_precision)
            results.append(
                VariantResult(
                    variant=variant.label,
                    degradation=degradation,
                    aggregation=strategy,
                    auc=curve.auc,
                    eer=curve.eer,
                    genuine=int(labels.sum()),
                    impostor=int((labels == 0).sum()),
                    identities=identities,
                    photos=len(gallery),
                    detection_failures=len(gallery_failures),
                    probe_detection_failures=len(probe_failures),
                    seconds_per_image=per_image,
                    operating_points={
                        f"p{int(target_precision * 100)}": point,
                        "eer": precision_recall_at(scores, labels, curve.eer_threshold),
                    },
                )
            )
    return results


def main(argv: list[str] | None = None) -> int:
    """Sweep pipeline variants and report identity-matching precision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    parser.add_argument("--target-precision", type=float, default=0.99)
    parser.add_argument("--detectors", nargs="*", default=["opencv_yunet"])
    parser.add_argument("--embedders", nargs="*", default=["opencv_sface", "insightface"])
    parser.add_argument("--tta", nargs="*", type=int, default=[0, 1])
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="also evaluate the fusion of all listed embedders",
    )
    parser.add_argument("--aggregations", nargs="*", default=list(AGGREGATIONS))
    parser.add_argument("--degradations", nargs="*", default=list(DEGRADATIONS))
    args = parser.parse_args(argv)

    if not args.faces.is_dir():
        raise SystemExit(f"no evaluation set at {args.faces}; run build_evaluation_set.py first")

    config = load_config()
    manifest = read_manifest(args.faces)
    if not manifest:
        raise SystemExit(f"no labelled images found in {args.faces}")

    counts: dict[str, int] = {}
    for identity in manifest.values():
        counts[identity] = counts.get(identity, 0) + 1
    usable = {k: v for k, v in counts.items() if v >= 2}
    print(f"evaluation set: {len(manifest)} photos, {len(counts)} identities")
    print(f"  usable for genuine pairs: {len(usable)} identities with 2+ photos")
    for identity, count in sorted(counts.items()):
        print(f"    {identity:14s} {count}")

    variants = [
        PipelineVariant(detector=d, embedder=e, flip_tta=bool(t))
        for d, e, t in itertools.product(args.detectors, args.embedders, args.tta)
    ]
    if args.ensemble and len(args.embedders) >= 2:
        for d, t in itertools.product(args.detectors, args.tta):
            variants.append(
                PipelineVariant(
                    detector=d,
                    embedder=args.embedders[0],
                    flip_tta=bool(t),
                    ensemble=tuple(args.embedders),
                )
            )

    rows: list[dict[str, Any]] = []
    all_results: list[VariantResult] = []
    for variant in variants:
        print(f"\nevaluating {variant.label}", flush=True)
        try:
            results = evaluate_variant(
                config, variant, args.faces, manifest,
                tuple(args.aggregations), tuple(args.degradations), args.target_precision,
            )
        except Exception as exc:
            print(f"  skipped: {exc}")
            continue
        for result in results:
            all_results.append(result)
            rows.append(result.to_row())
            key = f"p{int(args.target_precision * 100)}"
            point = result.operating_points[key]
            print(
                f"  {result.degradation:13s} {result.aggregation:10s} "
                f"AUC={result.auc:.4f} EER={result.eer:.4f}  "
                f"@P>={args.target_precision:.2f}: thr={point['threshold']:.4f} "
                f"R={point['recall']:.3f} FP={int(point['false_positives'])}/{result.impostor} "
                f"probe_miss={result.probe_detection_failures}"
            )

    if not rows:
        raise SystemExit("no variant produced usable comparisons")

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "face_pipeline_evaluation.csv"
    provenance = {f"env_{k}": v for k, v in environment(config).items()}
    enriched = [{**row, **provenance} for row in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(enriched[0].keys()))
        writer.writeheader()
        writer.writerows(enriched)

    key = f"p{int(args.target_precision * 100)}"
    by_variant: dict[tuple[str, str], list[float]] = {}
    for result in all_results:
        by_variant.setdefault((result.variant, result.aggregation), []).append(
            result.operating_points[key]["recall"]
        )
    ranked = sorted(
        ((k, float(np.mean(v))) for k, v in by_variant.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    print(f"\nmean recall at precision >= {args.target_precision:.2f}, "
          "averaged over degradations:")
    for (variant_label, strategy), mean_recall in ranked[:10]:
        print(f"  {mean_recall:.4f}  {variant_label} / {strategy}")
    (best_variant, best_strategy), best_recall = ranked[0]

    report = {
        "created_at": environment(config)["recorded_at"],
        "target_precision": args.target_precision,
        "identities": len(counts),
        "photos": len(manifest),
        "best": {
            "variant": best_variant,
            "aggregation": best_strategy,
            "mean_recall_at_target_precision": round(best_recall, 6),
        },
        "ranking": [
            {"variant": v, "aggregation": a, "mean_recall": round(r, 6)}
            for (v, a), r in ranked
        ],
        "results": rows,
        "environment": environment(config),
        "caveat": (
            "Comparisons within one identity are not independent; treat these "
            "numbers as a ranking of configurations, not as absolute accuracy."
        ),
    }
    json_path = args.output / "face_pipeline_evaluation.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {csv_path}\nwrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
