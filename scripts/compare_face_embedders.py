r"""Screen a second face recogniser against ArcFace where ArcFace is weakest.

On a face shrunk to about 55 pixels and re-encoded at crf 35 (``video_small``)
ArcFace puts 44% of genuine probes above the high-confidence threshold, yet the
genuine and impostor scores barely overlap there (AUC 0.99989): the loss is in
where the threshold sits, not in the information. A recogniser trained for low
quality might keep genuine scores higher under that degradation without raising
impostor scores. This screen measures it.

The protocol is ``evaluate_face_pipeline.py``'s: 170 photographs of 30
identities, a clean gallery, degraded probes, leave-one-out, max aggregation,
the deployed detector and aligner. Only the recogniser changes.

Scores are not comparable across recognisers, so each gets thresholds by the
rule the deployed ones were placed with: over the five calibration conditions
(clean, jpeg30, crop_20, blur_3, screenshot) take the highest impostor and the
lowest genuine score, and put the candidate and high-confidence thresholds a
quarter of the gap below and above its midpoint. Each recogniser is then judged
at its own operating point, and also at the strictest fair one: the recall it
could reach in a condition with no impostor of any condition above it.

The two recognisers are also fused, by averaging their scores for the same
probe and identity, and judged by the same rule: a recogniser that fails on
different probes than ArcFace could still help alongside it.

Embeddings are cached per recogniser and condition under ``--work``, so an
interrupted run resumes, and a rerun with every condition cached only
re-summarises.

Usage:
    python scripts/compare_face_embedders.py --adaface models/adaface_ir101.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_face_pipeline as pipeline

from deepshield.config import DeepShieldConfig, load_config
from deepshield.experiments import environment
from deepshield.risk.calibration import roc_curve

CALIBRATION = ("clean", "jpeg30", "crop_20", "blur_3", "screenshot")
TARGET = "video_small"


def place_thresholds(genuine: np.ndarray, impostor: np.ndarray) -> dict[str, float]:
    """Place candidate and high-confidence thresholds the way the deployed ones were.

    Both sit a quarter of the gap from its midpoint. When the distributions
    overlap there is no gap to place them in, and the result says so instead
    of inventing one.
    """
    impostor_max = float(impostor.max())
    genuine_min = float(genuine.min())
    gap = genuine_min - impostor_max
    middle = (genuine_min + impostor_max) / 2.0
    return {
        "impostor_max": round(impostor_max, 4),
        "genuine_min": round(genuine_min, 4),
        "gap": round(gap, 4),
        "separated": gap > 0,
        "candidate": round(middle - gap / 4.0, 4),
        "high_confidence": round(middle + gap / 4.0, 4),
    }


def bands(genuine: np.ndarray, thresholds: dict[str, float]) -> dict[str, int]:
    """Count genuine probes at high confidence, in review and below candidate."""
    high, candidate = thresholds["high_confidence"], thresholds["candidate"]
    return {
        "high_confidence": int((genuine >= high).sum()),
        "review": int(((genuine >= candidate) & (genuine < high)).sum()),
        "below_candidate": int((genuine < candidate).sum()),
    }


def configure(config: DeepShieldConfig, name: str, adaface: Path) -> DeepShieldConfig:
    """Return the deployed configuration with only the recogniser swapped."""
    if name == "arcface":
        return config
    embedder = config.face.embedder.model_copy(
        update={
            "backend": "onnx_arcface",
            "model_path": adaface,
            "model_name": "adaface_ir101_webface12m",
            "model_version": "cvlface",
            "ensemble": [],
        }
    )
    return config.model_copy(
        update={"face": config.face.model_copy(update={"embedder": embedder})}
    )


def scores_for(
    config: DeepShieldConfig,
    faces: Path,
    manifest: dict[str, str],
    conditions: list[str],
    work: Path,
    name: str,
) -> dict[str, dict[str, Any]]:
    """Return leave-one-out genuine and impostor scores per condition, cached.

    The clean gallery is embedded only when some condition is not cached yet.
    """
    gallery: dict[str, np.ndarray] = {}
    gallery_failures: list[str] = []
    per_image: float | None = None
    results: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        cache = work / f"{name}_{condition}.npz"
        if cache.is_file():
            saved = np.load(cache)
            results[condition] = {
                "scores": saved["scores"], "labels": saved["labels"],
                "probe_failures": int(saved["probe_failures"]),
            }
            continue
        if per_image is None:
            gallery, gallery_failures, per_image = pipeline.embed_all(config, faces, manifest)
        if condition == "clean":
            probes, failures = gallery, gallery_failures
        else:
            probes, failures, _ = pipeline.embed_all(config, faces, manifest, condition)
        scores, labels = pipeline.leave_one_out_scores(
            gallery, probes, manifest, "max", config.face.matcher.top_k
        )
        work.mkdir(parents=True, exist_ok=True)
        np.savez(cache, scores=scores, labels=labels, probe_failures=len(failures))
        results[condition] = {"scores": scores, "labels": labels, "probe_failures": len(failures)}
        print(f"  {name} {condition}: {int(labels.sum())} genuine, "
              f"{int((labels == 0).sum())} impostor", flush=True)
    results["_meta"] = {"gallery_failures": len(gallery_failures),
                        "seconds_per_image": None if per_image is None else round(per_image, 4)}
    return results


def fuse(first: dict[str, dict[str, Any]], second: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Average two recognisers' scores for the same probes and identities.

    Both score lists come from the same leave-one-out loop over the same
    photographs, so they line up entry for entry; the labels are checked to
    make sure they do.

    Raises:
        ValueError: If a condition's pairs do not line up.

    """
    fused: dict[str, Any] = {}
    for condition in first:
        if condition.startswith("_"):
            continue
        a, b = first[condition], second[condition]
        if a["labels"].shape != b["labels"].shape or not np.array_equal(a["labels"], b["labels"]):
            raise ValueError(f"{condition}: the two recognisers scored different pairs")
        fused[condition] = {
            "scores": (a["scores"] + b["scores"]) / 2.0,
            "labels": a["labels"],
            "probe_failures": max(a["probe_failures"], b["probe_failures"]),
        }
    fused["_meta"] = {
        "gallery_failures": max(first["_meta"]["gallery_failures"],
                                second["_meta"]["gallery_failures"]),
        "seconds_per_image": None,
    }
    return fused


def summarise(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Place thresholds on the calibration conditions and judge every condition."""
    genuine = np.concatenate(
        [results[c]["scores"][results[c]["labels"] == 1] for c in CALIBRATION]
    )
    impostor = np.concatenate(
        [results[c]["scores"][results[c]["labels"] == 0] for c in CALIBRATION]
    )
    thresholds = place_thresholds(genuine, impostor)
    conditions = [c for c in results if not c.startswith("_")]
    every_impostor = max(
        float(results[c]["scores"][results[c]["labels"] == 0].max()) for c in conditions
    )
    report: dict[str, Any] = {"thresholds": thresholds, "conditions": {},
                              "impostor_max_all_conditions": round(every_impostor, 4),
                              **results["_meta"]}
    for condition in conditions:
        scores, labels = results[condition]["scores"], results[condition]["labels"]
        condition_genuine = scores[labels == 1]
        condition_impostor = scores[labels == 0]
        report["conditions"][condition] = {
            "genuine": int(condition_genuine.size),
            "impostor": int(condition_impostor.size),
            "probe_detection_failures": results[condition]["probe_failures"],
            "auc": round(roc_curve(scores, labels).auc, 5),
            "genuine_min": round(float(condition_genuine.min()), 4),
            "genuine_median": round(float(np.median(condition_genuine)), 4),
            "impostor_max": round(float(condition_impostor.max()), 4),
            "bands_at_own_thresholds": bands(condition_genuine, thresholds),
            "recall_above_every_impostor": round(
                float((condition_genuine > every_impostor).mean()), 4
            ),
            "impostors_above_high_confidence": int(
                (condition_impostor >= thresholds["high_confidence"]).sum()
            ),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    """Score both recognisers and write the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument("--adaface", type=Path, default=Path("models/adaface_ir101.onnx"))
    parser.add_argument("--conditions", nargs="+",
                        default=[*CALIBRATION, TARGET, "downscale_25"],
                        choices=sorted(pipeline.DEGRADATIONS))
    parser.add_argument("--work", type=Path, default=Path("data/results/face_embedder_cache"))
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)

    missing = [c for c in (*CALIBRATION, TARGET) if c not in args.conditions]
    if missing:
        raise SystemExit(f"--conditions must include {missing}")
    if not args.adaface.is_file():
        raise SystemExit(f"{args.adaface} not found; run scripts/fetch_face_embedder.py first")
    manifest = pipeline.read_manifest(args.faces)
    base = load_config()
    report: dict[str, Any] = {
        "question": "does a quality-adaptive recogniser recover small, compressed faces?",
        "protocol": "evaluate_face_pipeline.py leave-one-out, max aggregation, deployed "
                    "detector and aligner, flip TTA for both recognisers",
        "threshold_rule": "a quarter of the calibration gap either side of its midpoint, "
                          f"over {', '.join(CALIBRATION)}",
        "photos": len(manifest),
        "identities": len(set(manifest.values())),
        "recognisers": {},
    }
    scored: dict[str, dict[str, dict[str, Any]]] = {}
    for name in ("arcface", "adaface"):
        print(name, flush=True)
        config = configure(base, name, args.adaface)
        scored[name] = scores_for(config, args.faces, manifest, args.conditions, args.work, name)
    scored["arcface+adaface"] = fuse(scored["arcface"], scored["adaface"])
    for name, results in scored.items():
        report["recognisers"][name] = summarise(results)
        target = report["recognisers"][name]["conditions"][TARGET]
        print(name)
        print(f"  thresholds {report['recognisers'][name]['thresholds']}")
        print(f"  {TARGET}: {target['bands_at_own_thresholds']}, auc {target['auc']}, "
              f"recall above every impostor {target['recall_above_every_impostor']}",
              flush=True)
    report["environment"] = environment(base)
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "face_embedder_screen.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
