r"""Choose identity thresholds that hold for Korean users, and test cohort normalisation.

``evaluate_korean_faces.py`` found that on KoDF's 40 validation people two
comparisons between different Korean women passed the high-confidence
threshold fitted on LFW. Forty people make 780 pairs, too few to fit a new
threshold on, and every KoDF person was filmed on one day in one place, so a
KoDF genuine score compares a person with the same session and says little
about recognising them in a different photograph. The two sides are therefore
taken from where each is measured properly:

impostor
    KoDF's training split, as many people as have been downloaded (about 20
    per zip), one frame from each of several videos per person: every probe
    against every other person's enrollment, clean and as ``video_small``.
genuine
    The LFW evaluation set (30 people, 170 photographs taken on different
    occasions): each photograph against an enrollment from that person's other
    photographs, clean and as ``video_small``. KoDF's same-session genuine
    scores are reported beside it, as an upper bound.

KoDF people are split in two halves. A threshold is fitted on one half's
impostors (the lowest that no impostor reaches, plus a margin) and tested on
the other half's impostors and on the genuine sets, so the numbers are not
read off the data that chose them.

Cohort normalisation is scored the same way. A face that resembles many
people scores high against everyone, including a user it is not; subtracting
how much the probe resembles a cohort of other people removes that part. The
cohort is LFW's East Asian identities (public, selected by
``evaluate_impostor_tails.population``), not KoDF, whose terms forbid passing
its people on: a shipped cohort could only be public faces. The score is
``raw - mean of the probe's top-k cohort similarities``.

The detector's false detections on KoDF's pegboard backdrop are recorded too:
for every frame with more than one detection, the geometry of each box and
its landmarks, so a filter can be designed from what separates them.

Usage:
    python scripts/evaluate_korean_thresholds.py \
        --kodf "~/Downloads/딥페이크 변조 영상/1.Training" \
        --meta ../kodf_work/meta/train/train_meta_data --work ../kodf_work
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_impostor_tails import evaluation_identities, identities, population
from evaluate_korean_faces import index_zips, sample_frames

from deepshield.config import load_config
from deepshield.experiments import environment
from deepshield.face.aligner import build_aligner
from deepshield.face.detector import build_detector
from deepshield.face.embedder import build_embedder
from deepshield.media import load_image
from deepshield.transforms import Transformation

ENROLL_VIDEOS = 3
PROBE_VIDEOS = 3
SMALL_FACE = 55.0
LFW_SMALL = Transformation("video_small", "video_compression", {"scale": 0.55, "crf": 35})
MARGIN = 0.01
COHORT_K = 10
CONDITIONS = ("clean", "video_small")


class Stack:
    """The configured detector, aligner and embedder."""

    def __init__(self) -> None:
        """Build the face stack."""
        config = load_config()
        self.detector = build_detector(config.face.detector)
        self.aligner = build_aligner(config.face.aligner)
        self.embedder = build_embedder(config.face.embedder)

    def embed(self, image: np.ndarray, face: Any) -> list[float]:
        """Return one face's embedding."""
        return self.embedder.embed(self.aligner.align(image, face).image).vector.tolist()

    def best(self, image: np.ndarray) -> dict[str, Any] | None:
        """Return the most confident face's embedding and size, or ``None``."""
        faces = self.detector.detect(image)
        if not faces:
            return None
        face = max(faces, key=lambda f: f.detection_confidence)
        return {
            "vector": self.embed(image, face),
            "pixels": float(min(face.bbox.width, face.bbox.height)),
        }


def geometry(face: Any, image: np.ndarray) -> dict[str, float]:
    """Describe a detection's box and landmarks, for telling backdrops from faces."""
    height, width = image.shape[:2]
    box = face.bbox
    out = {
        "confidence": float(face.detection_confidence),
        "side_over_frame_height": float(min(box.width, box.height) / height),
        "aspect": float(box.width / max(box.height, 1e-6)),
        "centre_x": float((box.x1 + box.x2) / 2 / width),
    }
    if face.landmarks is not None:
        points = np.asarray(face.landmarks, dtype=np.float64)
        eyes = float(np.linalg.norm(points[1] - points[0]))
        out["eye_distance_over_width"] = eyes / max(box.width, 1e-6)
        mouth = (points[3] + points[4]) / 2
        out["eye_to_mouth_over_height"] = float(
            (mouth[1] - (points[0][1] + points[1][1]) / 2) / max(box.height, 1e-6)
        )
    return out


def embed_video(stack: Stack, path: Path, role: str) -> dict[str, Any]:
    """Read the middle frame of a video: the best face clean and small, and every detection."""
    frames = sample_frames(path, 1)
    if not frames:
        return {"frames": 0}
    image = frames[0]
    faces = stack.detector.detect(image)
    record: dict[str, Any] = {"frames": 1, "detections": [geometry(f, image) for f in faces]}
    if not faces:
        return record
    best = max(faces, key=lambda f: f.detection_confidence)
    record["best_index"] = faces.index(best)
    record["clean"] = {
        "vector": stack.embed(image, best),
        "pixels": float(min(best.bbox.width, best.bbox.height)),
    }
    if role == "probe":
        scale = min(1.0, SMALL_FACE / record["clean"]["pixels"])
        small = Transformation(
            "video_small", "video_compression", {"scale": scale, "crf": 35}
        ).apply(image)
        record["video_small"] = stack.best(small)
    return record


def kodf_embeddings(
    kodf: Path, meta: Path, work: Path, stack: Stack, seed: int
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Embed every KoDF person's enrollment and probe videos, with a cache.

    A person already in the cache keeps the videos embedded for them, so the
    zips they came from need not be downloaded again when more people are
    added. A new person's videos are drawn with a seed of their own, so adding
    people never changes whom anyone else was measured on.
    """
    members = index_zips(kodf)
    rows = list(csv.DictReader(next(meta.glob("원본영상_*.csv")).open(encoding="utf-8-sig")))
    owner = {row["영상ID"]: row["UUID"] for row in rows}
    videos: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        videos[row["UUID"]].append(row["영상ID"])
    cache = work / "threshold_cache"
    cache.mkdir(parents=True, exist_ok=True)
    cached: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for stored in cache.glob("*.json"):
        stem, role = stored.stem.rsplit("_", 1)
        cached[owner[f"{stem}.mp4"]].append((f"{stem}.mp4", role))
    wanted = ENROLL_VIDEOS + PROBE_VIDEOS
    picks: dict[str, list[tuple[str, str]]] = {}
    for person in sorted(videos):
        if len(cached[person]) >= wanted:
            picks[person] = sorted(cached[person])
            continue
        names = sorted(n for n in videos[person] if n in members)
        if len(names) < wanted:
            continue
        own = np.random.default_rng([seed, int.from_bytes(person.encode()[:8], "big")])
        chosen = [names[i] for i in own.choice(len(names), wanted, replace=False)]
        picks[person] = [
            (name, "enroll" if index < ENROLL_VIDEOS else "probe")
            for index, name in enumerate(chosen)
        ]
    people: dict[str, dict[str, list[dict[str, Any]]]] = {
        person: {"enroll": [], "probe": []} for person in picks
    }
    by_archive: dict[Path, list[tuple[str, str, str]]] = defaultdict(list)
    for person, chosen in picks.items():
        for name, role in chosen:
            stored = cache / f"{Path(name).stem}_{role}.json"
            if stored.exists():
                people[person][role].append(json.loads(stored.read_text()))
            else:
                by_archive[members[name][0]].append((person, name, role))
    done = 0
    for archive, missing in by_archive.items():
        with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory() as scratch:
            for person, name, role in missing:
                target = Path(scratch) / "video.mp4"
                with bundle.open(members[name][1]) as source, target.open("wb") as sink:
                    while chunk := source.read(1 << 22):
                        sink.write(chunk)
                record = embed_video(stack, target, role)
                target.unlink()
                (cache / f"{Path(name).stem}_{role}.json").write_text(json.dumps(record))
                people[person][role].append(record)
                done += 1
                if done % 50 == 0:
                    print(f"embedded {done} KoDF videos", flush=True)
    return people


def lfw_genuine(faces: Path, stack: Stack) -> list[dict[str, Any]]:
    """Embed the LFW evaluation photographs clean and small, grouped by person."""
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(faces.glob("*.png")):
        grouped[path.stem.rsplit("_", 1)[0]].append(path)
    out = []
    for person, paths in grouped.items():
        if len(paths) < 3:
            continue
        for path in paths:
            image = load_image(path)
            clean = stack.best(image)
            if clean is None:
                continue
            out.append(
                {
                    "person": person,
                    "photo": path.name,
                    "clean": clean,
                    "video_small": stack.best(LFW_SMALL.apply(image)),
                }
            )
    return out


def cohort_vectors(lfw: Path, faces: Path, stack: Stack, limit: int) -> np.ndarray:
    """Embed one photograph of each LFW East Asian identity outside the evaluation set."""
    excluded = evaluation_identities(faces)
    chosen = [
        paths[0]
        for name, paths in identities(lfw, excluded).items()
        if population(name) == "east_asian" and paths
    ][:limit]
    vectors = []
    for path in chosen:
        found = stack.best(load_image(path))
        if found is not None:
            vectors.append(found["vector"])
    return np.asarray(vectors, dtype=np.float32)


def unit(vectors: list[list[float]]) -> np.ndarray:
    """Return rows as L2-normalised float32 vectors."""
    array = np.asarray(vectors, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


def cohort_offset(probe: np.ndarray, cohort: np.ndarray) -> float:
    """Return the mean of the probe's top-k similarities to the cohort."""
    if cohort.size == 0:
        return 0.0
    scores = np.sort(cohort @ probe)[::-1][:COHORT_K]
    return float(scores.mean())


def fit_threshold(impostors: np.ndarray) -> float:
    """Return the lowest threshold no fitting impostor reaches, plus the margin."""
    return float(impostors.max() + MARGIN) if impostors.size else float("nan")


def main(argv: list[str] | None = None) -> int:
    """Score KoDF impostors and LFW genuines, raw and cohort-normalised, and fit thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kodf", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--faces", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument(
        "--lfw", type=Path, default=Path("data/sklearn/lfw_home/lfw_funneled")
    )
    parser.add_argument("--cohort", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    stack = Stack()
    people = kodf_embeddings(
        args.kodf.expanduser(), args.meta.expanduser(), args.work.expanduser(), stack, args.seed
    )
    genuine_lfw = lfw_genuine(args.faces, stack)
    cohort = cohort_vectors(args.lfw, args.faces, stack, args.cohort)
    print(f"{len(people)} KoDF people, {len(genuine_lfw)} LFW photos, cohort {len(cohort)}")

    enrolled = {
        person: unit([r["clean"]["vector"] for r in roles["enroll"] if "clean" in r])
        for person, roles in people.items()
    }
    enrolled = {p: v for p, v in enrolled.items() if len(v)}
    order = sorted(enrolled)
    halves = rng.permutation(len(order))
    fit_people = {order[i] for i in halves[: len(order) // 2]}

    def impostor_scores(condition: str, members: set[str], normalise: bool) -> np.ndarray:
        scores = []
        for person in members:
            for record in people[person]["probe"]:
                face = record.get(condition)
                if not face:
                    continue
                probe = unit([face["vector"]])[0]
                offset = cohort_offset(probe, cohort) if normalise else 0.0
                for other in members:
                    if other != person:
                        scores.append(float((enrolled[other] @ probe).max()) - offset)
        return np.asarray(scores)

    def kodf_genuine(condition: str, members: set[str], normalise: bool) -> np.ndarray:
        scores = []
        for person in members:
            for record in people[person]["probe"]:
                face = record.get(condition)
                if face:
                    probe = unit([face["vector"]])[0]
                    offset = cohort_offset(probe, cohort) if normalise else 0.0
                    scores.append(float((enrolled[person] @ probe).max()) - offset)
        return np.asarray(scores)

    def lfw_scores(condition: str, normalise: bool) -> np.ndarray:
        scores = []
        for row in genuine_lfw:
            face = row.get(condition)
            if not face:
                continue
            references = unit(
                [
                    other["clean"]["vector"]
                    for other in genuine_lfw
                    if other["person"] == row["person"] and other["photo"] != row["photo"]
                ]
            )
            probe = unit([face["vector"]])[0]
            offset = cohort_offset(probe, cohort) if normalise else 0.0
            scores.append(float((references @ probe).max()) - offset)
        return np.asarray(scores)

    config = load_config()
    current = config.thresholds.face_similarity.high_confidence_threshold
    test_people = set(order) - fit_people
    methods: dict[str, Any] = {}
    for method, normalise in (("raw", False), ("cohort_normalised", True)):
        fit_imp = impostor_scores("clean", fit_people, normalise)
        test_imp = {c: impostor_scores(c, test_people, normalise) for c in CONDITIONS}
        lfw = {c: lfw_scores(c, normalise) for c in CONDITIONS}
        kodf = {c: kodf_genuine(c, test_people, normalise) for c in CONDITIONS}
        fitted = fit_threshold(fit_imp)
        candidates = {"fitted_on_half": fitted}
        if method == "raw":
            candidates["current"] = current

        def at(
            threshold: float,
            lfw: dict[str, np.ndarray] = lfw,
            kodf: dict[str, np.ndarray] = kodf,
            test_imp: dict[str, np.ndarray] = test_imp,
        ) -> dict[str, Any]:
            return {
                "threshold": round(threshold, 4),
                "lfw_genuine_recall": {
                    c: round(float((lfw[c] >= threshold).mean()), 4) for c in CONDITIONS
                },
                "kodf_same_session_genuine_recall": {
                    c: round(float((kodf[c] >= threshold).mean()), 4) for c in CONDITIONS
                },
                "kodf_test_impostors_at_or_above": {
                    c: int((test_imp[c] >= threshold).sum()) for c in CONDITIONS
                },
            }

        methods[method] = {
            "fit_impostor_max": round(float(fit_imp.max()), 4),
            "test_impostor_max": {c: round(float(test_imp[c].max()), 4) for c in CONDITIONS},
            "test_impostor_comparisons": {c: int(test_imp[c].size) for c in CONDITIONS},
            "lfw_genuine_min": {c: round(float(lfw[c].min()), 4) for c in CONDITIONS},
            "separation_gap_clean": round(float(lfw["clean"].min() - fit_imp.max()), 4),
            "operating_points": {name: at(value) for name, value in candidates.items()},
        }

    detections = [
        (index == record.get("best_index"), geometry_row)
        for roles in people.values()
        for record in roles["enroll"] + roles["probe"]
        for index, geometry_row in enumerate(record.get("detections", []))
    ]
    extra = [g for best, g in detections if not best]
    kept = [g for best, g in detections if best]
    keys = (
        "confidence", "side_over_frame_height", "eye_distance_over_width",
        "eye_to_mouth_over_height",
    )

    def spread(rows: list[dict[str, float]], key: str) -> dict[str, float] | None:
        values = [r[key] for r in rows if key in r]
        if not values:
            return None
        return {
            "min": round(float(np.min(values)), 3),
            "median": round(float(np.median(values)), 3),
            "max": round(float(np.max(values)), 3),
        }

    report = {
        "question": "which high-confidence threshold keeps Korean impostors out while "
        "recognising people in other photographs, and does cohort normalisation help?",
        "data": "KoDF training split (AI Hub '딥페이크 변조 영상', National Information "
        "Society Agency) for impostors; the LFW evaluation set for genuine scores. Subject "
        "IDs are not reported.",
        "kodf_people": len(order),
        "fit_half_people": len(fit_people),
        "test_half_people": len(test_people),
        "lfw_photos": len(genuine_lfw),
        "cohort": f"{len(cohort)} LFW East Asian identities, top-{COHORT_K} mean",
        "margin": MARGIN,
        "methods": methods,
        "detections": {
            "frames": sum(len(r["enroll"]) + len(r["probe"]) for r in people.values()),
            "frames_with_more_than_one": sum(
                1
                for roles in people.values()
                for record in roles["enroll"] + roles["probe"]
                if len(record.get("detections", [])) > 1
            ),
            "chosen": {k: spread(kept, k) for k in keys},
            "others": {k: spread(extra, k) for k in keys},
        },
        "environment": environment(config),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    (args.output / "korean_thresholds_kodf.json").write_text(text + "\n", encoding="utf-8")
    (args.work.expanduser() / "detections.json").write_text(
        json.dumps({"chosen": kept, "others": extra})
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
