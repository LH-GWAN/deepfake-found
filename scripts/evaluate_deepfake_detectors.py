r"""Decide whether any deepfake detector may change a verdict, on data it never saw.

A checkpoint's headline accuracy describes the dataset it was trained on. This
script asks the only question that matters here: can the detector call a face
synthetic, on this system's own material, while being wrong about genuine
photographs rarely enough that a person could live with the alert?

The adoption gate, fixed before any new detector is measured:

threshold
    The 99.5th percentile of the detector's scores on genuine photographs of
    the calibration identities. It is fitted on genuine photographs alone, so
    it does not depend on which fakes happen to be in the set.
false alarms
    On genuine photographs of the test identities, the upper end of the 95%
    Clopper-Pearson interval of the false-positive rate must be at most 1%, in
    every condition (clean, JPEG q50, a small face under H.264 crf 35). A
    point estimate of zero on a hundred photographs is not a rate of zero.
recall
    On test identities, at least 20% of each manipulation family's fakes must
    be flagged on clean photographs. A detector that never fires passes the
    false-alarm bar trivially.
no leakage
    A detector whose metadata lists an evaluated manifest among its training
    data is reported but never adopted: the manipulation sets share their
    genuine photographs, so even a family it never saw is in-sample.

Identities are split into calibration and test halves; a fake belongs to a half
only when both the person in the photograph and the person whose face was
pasted in belong to it. The genuine photographs of the manipulation sets are
too few to bound a false-positive rate near 1% (85 test photographs with no
false alarm still allow 4.2%), so ``--genuine`` adds one photograph each from
other LFW identities, split the same way.

``separation`` columns report ROC-AUC over every record of a family, the number
earlier versions of this script gated on, so old and new results can be read
side by side. With ``--write`` a detector that passes the gate becomes the
configured backend with the fitted threshold; that is what lets the verdict
engine raise an identity match to ``synthetic_suspected``.

Usage:
    python scripts/evaluate_deepfake_detectors.py \\
        --manifest data/test/manipulated/manifest.json \\
        data/test/manipulated_inswapper/manifest.json \\
        --genuine data/sklearn/lfw_home/lfw_funneled --genuine-limit 3000
    python scripts/evaluate_deepfake_detectors.py ... --write
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepshield.config import DeepfakeDetectorConfig, load_config
from deepshield.detection.deepfake_backends import OnnxDeepfakeDetector
from deepshield.experiments import environment
from deepshield.face.detector import build_detector
from deepshield.media import load_image
from deepshield.pipeline.analysis_pipeline import DEEPFAKE_CROP_MARGIN, crop_with_margin
from deepshield.risk.calibration import clopper_pearson_upper, roc_curve
from deepshield.transforms import Transformation

DEGRADATIONS = {
    "clean": ("identity", {}),
    "jpeg50": ("jpeg_compression", {"quality": 50}),
    "video_small": ("video_compression", {"scale": 0.55, "crf": 35}),
}
GENUINE_QUANTILE = 0.995
MAX_FPR_BOUND = 0.01
MIN_FAMILY_RECALL = 0.20
CALIBRATION_SHARE = 0.5


def discover_models(model_dir: Path) -> list[tuple[str, Path, dict[str, Any]]]:
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


def identity_of(filename: str) -> str:
    """Return the identity in an evaluation-set file name such as ``thomas_fargo_3.png``."""
    return Path(filename).stem.rsplit("_", 1)[0].lower()


def family_of(manifest: Path, payload: dict[str, Any]) -> str:
    """Name a manifest's family the way ``train_deepfake_cnn.py`` does."""
    swapper = payload.get("swapper")
    if swapper:
        return str(swapper)
    name = manifest.parent.name
    return name.replace("manipulated_", "") if "_" in name else "graphics"


def load_records(manifests: list[Path]) -> list[dict[str, Any]]:
    """Return every record of every manifest with its family and the people in it."""
    records: list[dict[str, Any]] = []
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        family = family_of(manifest, payload)
        for record in payload["records"]:
            people = {record["identity"].lower()}
            if record.get("donor"):
                people.add(identity_of(record["donor"]))
            records.append(
                {
                    "path": record["path"],
                    "label": record["label"],
                    "family": family,
                    "manifest": manifest.as_posix(),
                    "people": sorted(people),
                }
            )
    return records


def genuine_photos(lfw: Path, excluded: set[str], limit: int, seed: int) -> list[dict[str, Any]]:
    """Return one photograph each from up to ``limit`` LFW identities outside ``excluded``."""
    folders = sorted(
        folder for folder in lfw.iterdir()
        if folder.is_dir() and folder.name.lower() not in excluded
    )
    random.Random(seed).shuffle(folders)
    chosen = []
    for folder in folders[:limit]:
        photos = sorted(folder.glob("*.jpg"))
        if photos:
            chosen.append(
                {
                    "path": photos[0].as_posix(),
                    "label": "real",
                    "family": "lfw",
                    "manifest": "genuine",
                    "people": [folder.name.lower()],
                }
            )
    return chosen


def assign_halves(items: list[dict[str, Any]], share: float, seed: int) -> None:
    """Mark each item ``calibration``, ``test`` or ``split`` by the people in it.

    Identities are shuffled once and the first ``share`` of them calibrate. An
    item whose people fall on both sides is ``split`` and used by neither half,
    so no face seen while fitting the threshold is judged afterwards.
    """
    people = sorted({person for item in items for person in item["people"]})
    random.Random(seed).shuffle(people)
    calibration = set(people[: int(round(len(people) * share))])
    for item in items:
        sides = {person in calibration for person in item["people"]}
        item["half"] = (
            "split" if len(sides) > 1 else "calibration" if sides == {True} else "test"
        )


def score_items(
    detector: OnnxDeepfakeDetector,
    face_detector: Any,
    items: list[dict[str, Any]],
    degradation: str,
) -> np.ndarray:
    """Return a score per item on its largest face; ``nan`` where no face is found."""
    kind, params = DEGRADATIONS[degradation]
    transformation = Transformation(degradation, kind, params)
    scores = np.full(len(items), np.nan)
    for index, item in enumerate(items):
        path = Path(item["path"])
        if not path.is_file():
            continue
        image = load_image(path)
        if degradation != "clean":
            image = transformation.apply(image, seed=1)
        faces = face_detector.detect(image)
        if not faces:
            continue
        face = max(faces, key=lambda f: f.bbox.width * f.bbox.height)
        scores[index] = detector.predict_image(
            crop_with_margin(image, face, DEEPFAKE_CROP_MARGIN)
        ).score
    return scores


def in_sample(metadata: dict[str, Any], manifests: list[Path]) -> bool:
    """Return whether the detector was trained on any of the evaluated manifests."""
    trained = {Path(item).as_posix() for item in metadata.get("training_families_manifests", [])}
    return any(manifest.as_posix() in trained for manifest in manifests)


def evaluate(
    items: list[dict[str, Any]], scores: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Fit the threshold on calibration genuine photos and judge the test half."""
    labels = np.asarray([item["label"] == "fake" for item in items])
    halves = np.asarray([item["half"] for item in items])
    families = sorted({item["family"] for item in items if item["label"] == "fake"})
    of_family = np.asarray([item["family"] for item in items])

    clean = scores["clean"]
    fitting = clean[(halves == "calibration") & ~labels & np.isfinite(clean)]
    threshold = float(np.quantile(fitting, GENUINE_QUANTILE))

    report: dict[str, Any] = {
        "threshold": round(threshold, 6),
        "calibration_genuine": int(fitting.size),
        "genuine": {},
        "families": {},
    }
    for degradation, values in scores.items():
        judged = (halves == "test") & ~labels & np.isfinite(values)
        alarms = int((values[judged] >= threshold).sum())
        report["genuine"][degradation] = {
            "test_photos": int(judged.sum()),
            "false_positives": alarms,
            "false_positive_rate": round(alarms / max(int(judged.sum()), 1), 6),
            "false_positive_rate_upper_95": round(
                clopper_pearson_upper(alarms, int(judged.sum())), 6
            ) if judged.any() else None,
            "undetected": int(((halves == "test") & ~labels & ~np.isfinite(values)).sum()),
        }
    for family in families:
        member = of_family == family
        entry: dict[str, Any] = {}
        for degradation, values in scores.items():
            fakes = (halves == "test") & labels & (of_family == family) & np.isfinite(values)
            both = member & np.isfinite(values)
            separation = (
                roc_curve(values[both], labels[both].astype(int)).auc
                if labels[both].any() and (~labels[both]).any()
                else None
            )
            entry[degradation] = {
                "separation_auc": None if separation is None else round(separation, 6),
                "test_fakes": int(fakes.sum()),
                "test_recall": round(float((values[fakes] >= threshold).mean()), 6)
                if fakes.any() else None,
            }
        report["families"][family] = entry
    return report


def verdict(report: dict[str, Any], leaked: bool) -> tuple[bool, list[str]]:
    """Apply the gate and list every reason a detector fails it."""
    reasons: list[str] = []
    if leaked:
        reasons.append("trained on an evaluated manifest: every number here is in-sample")
    for degradation, row in report["genuine"].items():
        bound = row["false_positive_rate_upper_95"]
        if bound is None or bound > MAX_FPR_BOUND:
            reasons.append(
                f"{degradation}: false-positive rate could be as high as "
                f"{'unknown' if bound is None else f'{bound:.2%}'} "
                f"({row['false_positives']}/{row['test_photos']}), above {MAX_FPR_BOUND:.0%}"
            )
    for family, entry in report["families"].items():
        recall = entry["clean"]["test_recall"]
        if recall is None or recall < MIN_FAMILY_RECALL:
            reasons.append(
                f"{family}: flags {'no' if recall is None else f'{recall:.0%} of'} test fakes, "
                f"below {MIN_FAMILY_RECALL:.0%}"
            )
    return not reasons, reasons


def adopt(alias: str, onnx_path: Path, metadata: dict[str, Any], threshold: float,
          source: Path) -> None:
    """Make the detector the configured backend and mark its threshold calibrated."""
    high = max(threshold, min(0.99, threshold + (1.0 - threshold) / 2.0))
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
        f"  suspicious_threshold: {threshold:.4f}\n"
        f"  high_confidence_threshold: {high:.4f}\n"
        "  calibrated: true\n"
        f"  calibration_source: {source.as_posix()}\n"
    )
    head, _, rest = text.partition("deepfake:")
    _, _, tail = rest.partition("\n\n")
    thresholds_path.write_text(head + block + "\n" + tail, encoding="utf-8")
    print(f"updated {thresholds_path}; {alias} can now raise an identity match "
          f"to 'synthetic_suspected' at {threshold:.4f}")


def main(argv: list[str] | None = None) -> int:
    """Score every exported detector, apply the gate, and adopt a passing one if asked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, nargs="+",
        default=[Path("data/test/manipulated/manifest.json")],
    )
    parser.add_argument("--genuine", type=Path, default=None,
                        help="LFW-style folder of extra genuine photographs, one per identity")
    parser.add_argument("--genuine-limit", type=int, default=3000)
    parser.add_argument("--faces", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--only", nargs="*", default=None, help="detector aliases to score")
    parser.add_argument("--limit", type=int, default=None, help="cap records per manifest")
    parser.add_argument("--conditions", nargs="+", choices=sorted(DEGRADATIONS),
                        default=list(DEGRADATIONS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    parser.add_argument("--tag", default=None, help="suffix for the report file name")
    parser.add_argument("--write", action="store_true",
                        help="adopt the best detector that passes the gate")
    args = parser.parse_args(argv)

    if "clean" not in args.conditions:
        args.conditions.insert(0, "clean")
    missing = [str(manifest) for manifest in args.manifest if not manifest.is_file()]
    if missing:
        raise SystemExit(f"missing {missing}; run scripts/build_manipulation_set.py first")
    records = load_records(args.manifest)
    if args.limit:
        kept: list[dict[str, Any]] = []
        for manifest in args.manifest:
            kept += [r for r in records if r["manifest"] == manifest.as_posix()][: args.limit]
        records = kept
    items = list(records)
    if args.genuine is not None:
        excluded = {person for record in records for person in record["people"]}
        listing = args.faces / "identities.txt"
        if listing.is_file():
            excluded |= {
                line.split("\t")[1].strip()
                for line in listing.read_text(encoding="utf-8").splitlines() if "\t" in line
            }
        items += genuine_photos(args.genuine, excluded, args.genuine_limit, args.seed)
    assign_halves(items, CALIBRATION_SHARE, args.seed)

    models = [
        model for model in discover_models(args.models)
        if args.only is None or model[0] in args.only
    ]
    if not models:
        raise SystemExit(f"no exported detectors in {args.models}; run "
                         "scripts/fetch_deepfake_detector.py first")

    config = load_config()
    face_detector = build_detector(config.face.detector)
    halves = [item["half"] for item in items]
    print(f"{len(items)} images ({sum(r['label'] == 'fake' for r in items)} fake), "
          f"halves: calibration {halves.count('calibration')}, test {halves.count('test')}, "
          f"straddling {halves.count('split')}; {len(models)} detectors\n")

    results: dict[str, dict[str, Any]] = {}
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
        scores = {
            degradation: score_items(detector, face_detector, items, degradation)
            for degradation in args.conditions
        }
        report = evaluate(items, scores)
        leaked = in_sample(metadata, args.manifest)
        usable, reasons = verdict(report, leaked)
        results[alias] = {
            "repo": metadata["repo"],
            "in_sample": leaked,
            "usable": usable,
            "reasons": reasons,
            **report,
        }
        for degradation, row in report["genuine"].items():
            print(f"  {degradation:11s} genuine false positives "
                  f"{row['false_positives']}/{row['test_photos']} "
                  f"(upper 95% {row['false_positive_rate_upper_95']})")
        for family, entry in report["families"].items():
            clean = entry["clean"]
            print(f"  {family:11s} separation AUC {clean['separation_auc']}, "
                  f"test recall {clean['test_recall']} of {clean['test_fakes']}")
        print(f"  threshold {report['threshold']:.4f}; "
              f"{'USABLE' if usable else 'not usable: ' + '; '.join(reasons)}\n")

    args.output.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    report_path = args.output / f"deepfake_detector_gate{suffix}.json"
    report_path.write_text(
        json.dumps(
            {
                "environment": environment(config),
                "manifests": [manifest.as_posix() for manifest in args.manifest],
                "genuine_source": None if args.genuine is None else args.genuine.as_posix(),
                "gate": {
                    "threshold": f"{GENUINE_QUANTILE:.1%} quantile of calibration genuine scores",
                    "max_false_positive_rate_upper_95": MAX_FPR_BOUND,
                    "min_recall_per_family": MIN_FAMILY_RECALL,
                    "conditions": list(args.conditions),
                },
                "detectors": results,
            },
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {report_path}")

    passing = [(alias, result) for alias, result in results.items() if result["usable"]]
    if not args.write:
        return 0
    if not passing:
        print("\nnot adopting any detector: none passed the gate. "
              "The signal still never changes a verdict.")
        return 0
    alias, result = max(
        passing,
        key=lambda item: min(e["clean"]["test_recall"] for e in item[1]["families"].values()),
    )
    onnx_path = next(path for name, path, _ in models if name == alias)
    metadata = next(meta for name, _, meta in models if name == alias)
    adopt(alias, onnx_path, metadata, result["threshold"], report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
