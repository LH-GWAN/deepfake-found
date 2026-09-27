r"""Re-measure the identity thresholds on Korean faces (KoDF), and the swap donor on them.

Every identity threshold here was fitted on thirty LFW identities, almost all
of them Western, and LFW's own East Asian names showed a heavier impostor tail
than matched controls (``evaluate_impostor_tails.py``). The person most likely
to use DeepShield is Korean. KoDF (AI Hub, "딥페이크 변조 영상") films 403
Korean people; its validation split holds 40 of them with about 150 real
videos each, plus face swaps made from them. This script asks three things of
the thresholds as configured, without refitting them:

genuine
    Does a Korean person's face match their own enrollment? Each person is
    enrolled from one frame of each of a few videos, the way three to ten
    photographs enroll a user, and probed with frames of their other videos.
impostor
    Do different Korean people match each other? Every probe is scored against
    every other person's enrollment: the tail of those scores against the
    thresholds, and how often the product, with all of them enrolled, would
    name the wrong person at high confidence.
swap donor
    In a face swap whose face came from one of the enrolled people, is the
    donor found? This is the question the project began with, "was my face
    used", asked of Korean faces and of KoDF's swappers (DeepFaceLab-style
    ``dffs``, ``dfl`` and FSGAN), none of which built the evaluation set here.

Every probe is read clean and as ``video_small``: the frame shrunk until the
face is about 55 pixels and re-encoded as H.264 crf 35, as a reposted video
frame would be.

Frames are read straight out of the downloaded zips, one video at a time, so
nothing is extracted to disk beyond one temporary file. Embeddings are cached
under ``--work`` (outside the repository) and should be deleted with the
videos. The report holds counts and score statistics only, no subject IDs:
KoDF's terms forbid passing the data on and any attempt to re-identify people.

KoDF is provided by the National Information Society Agency (NIA) through
AI Hub; results derived from it must say so, and the data may not leave Korea.

Usage:
    python scripts/evaluate_korean_faces.py \
        --kodf "~/Downloads/딥페이크 변조 영상/2.Validation" \
        --meta ../kodf_work/meta/validate/validate_meta_data --work ../kodf_work
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepshield.config import load_config
from deepshield.experiments import environment
from deepshield.face.aligner import build_aligner
from deepshield.face.detector import build_detector
from deepshield.face.embedder import build_embedder
from deepshield.face.matcher import build_matcher
from deepshield.quality import face_quality_score
from deepshield.transforms import Transformation
from deepshield.types import IdentityProfile

SMALL_FACE = 55.0
CRF = 35
ENROLL_VIDEOS = 3
PROBE_VIDEOS = 7
PROBE_FRAMES = 2
FAKES_PER_MODEL = 300
FAKE_MODELS = ("dffs", "dfl", "fsgan")
TAIL = 0.25


def read_meta(folder: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return KoDF's real-video and fake-video metadata rows."""
    real = next(folder.glob("원본영상_*메타데이터.csv"))
    fake = next(folder.glob("변조영상_*메타데이터.csv"))
    with real.open(encoding="utf-8-sig") as handle:
        real_rows = list(csv.DictReader(handle))
    with fake.open(encoding="utf-8-sig") as handle:
        fake_rows = list(csv.DictReader(handle))
    return real_rows, fake_rows


def index_zips(root: Path) -> dict[str, tuple[Path, str]]:
    """Map every video file name inside the zips under ``root`` to its zip and member."""
    members: dict[str, tuple[Path, str]] = {}
    for archive in sorted(root.rglob("*.zip")):
        if "라벨링" in archive.name:
            continue
        try:
            with zipfile.ZipFile(archive) as bundle:
                names = bundle.namelist()
        except zipfile.BadZipFile:
            print(f"skipping {archive.name}: not a complete zip yet", flush=True)
            continue
        for name in names:
            if name.lower().endswith(".mp4"):
                members[Path(name).name] = (archive, name)
    return members


def sample_frames(path: Path, count: int) -> list[np.ndarray]:
    """Return ``count`` RGB frames spread evenly through a video."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    positions = [int(total * (i + 1) / (count + 1)) for i in range(count)] if total > 0 else []
    for position in positions:
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        ok, frame = capture.read()
        if ok:
            frames.append(np.ascontiguousarray(frame[:, :, ::-1]))
    capture.release()
    return frames


class FaceReader:
    """The pipeline's own detector, aligner and embedder, reading the largest face."""

    def __init__(self) -> None:
        """Build the configured face stack."""
        config = load_config()
        self.detector = build_detector(config.face.detector)
        self.aligner = build_aligner(config.face.aligner)
        self.embedder = build_embedder(config.face.embedder)

    def read(self, image: np.ndarray) -> dict[str, Any] | None:
        """Return the most confident face's embedding, size and quality, or ``None``.

        Not the largest: on KoDF's pegboard backdrop the detector also finds a
        frame-sized "face" in some frames, and taking the largest box compared
        backdrops with each other at 0.96. ``extra_faces`` counts the other
        detections so those frames stay visible in the report.
        """
        faces = self.detector.detect(image)
        if not faces:
            return None
        face = max(faces, key=lambda f: f.detection_confidence)
        aligned = self.aligner.align(image, face).image
        pixels = float(min(face.bbox.width, face.bbox.height))
        return {
            "vector": self.embedder.embed(aligned).vector,
            "pixels": pixels,
            "quality": face_quality_score(pixels, aligned),
            "extra_faces": len(faces) - 1,
            "extra_face_pixels": [
                float(min(f.bbox.width, f.bbox.height)) for f in faces if f is not face
            ],
        }

    def both(self, image: np.ndarray) -> dict[str, dict[str, Any] | None]:
        """Read the frame clean and as a small, crf 35 video frame."""
        clean = self.read(image)
        if clean is None:
            return {"clean": None, "video_small": None}
        scale = min(1.0, SMALL_FACE / clean["pixels"])
        small = Transformation(
            "video_small", "video_compression", {"scale": scale, "crf": CRF}
        ).apply(image)
        return {"clean": clean, "video_small": self.read(small)}


def embed_videos(
    names: list[str],
    members: dict[str, tuple[Path, str]],
    reader: FaceReader,
    frames: int,
    cache: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Embed ``frames`` frames of every named video, caching per video."""
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[dict[str, Any]]] = {}
    by_archive: dict[Path, list[str]] = defaultdict(list)
    for name in sorted(set(names)):
        if name in members:
            by_archive[members[name][0]].append(name)
    done = 0
    for archive, videos in by_archive.items():
        with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory() as scratch:
            for name in videos:
                stored = cache / f"{Path(name).stem}.npz"
                if stored.exists():
                    out[name] = json.loads(str(np.load(stored, allow_pickle=True)["rows"]))
                    for row in out[name]:
                        for condition in ("clean", "video_small"):
                            if row[condition] is not None:
                                row[condition]["vector"] = np.asarray(row[condition]["vector"])
                    continue
                target = Path(scratch) / "video.mp4"
                with bundle.open(members[name][1]) as source, target.open("wb") as sink:
                    while chunk := source.read(1 << 22):
                        sink.write(chunk)
                rows = [reader.both(frame) for frame in sample_frames(target, frames)]
                target.unlink()
                serialisable = [
                    {
                        condition: None
                        if row[condition] is None
                        else {**row[condition], "vector": row[condition]["vector"].tolist()}
                        for condition in ("clean", "video_small")
                    }
                    for row in rows
                ]
                np.savez(stored, rows=json.dumps(serialisable))
                out[name] = rows
                done += 1
                if done % 25 == 0:
                    print(f"embedded {done} videos", flush=True)
    return out


def profile(user: str, vectors: list[np.ndarray], embedder: Any) -> IdentityProfile:
    """Build an enrollment profile from reference embeddings, as enrollment stores it."""
    references = np.stack(vectors).astype(np.float32)
    centroid = references.mean(axis=0)
    return IdentityProfile(
        user_id=user,
        reference_embeddings=references,
        centroid_embedding=centroid / np.linalg.norm(centroid),
        image_count=len(vectors),
        model=embedder.model_info,
        embedding_dimension=int(references.shape[1]),
    )


def main(argv: list[str] | None = None) -> int:
    """Embed the sampled KoDF frames and score them against the configured thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kodf", type=Path, required=True, help="folder holding the zips")
    parser.add_argument("--meta", type=Path, required=True, help="folder of the metadata CSVs")
    parser.add_argument("--work", type=Path, required=True, help="cache outside the repository")
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    real_rows, fake_rows = read_meta(args.meta.expanduser())
    members = index_zips(args.kodf.expanduser())
    videos: dict[str, list[str]] = defaultdict(list)
    for row in real_rows:
        if row["영상ID"] in members:
            videos[row["UUID"]].append(row["영상ID"])
    people = sorted(p for p, names in videos.items() if len(names) >= ENROLL_VIDEOS + 1)
    enroll_names: dict[str, list[str]] = {}
    probe_names: dict[str, list[str]] = {}
    for person in people:
        names = sorted(videos[person])
        size = min(len(names), ENROLL_VIDEOS + PROBE_VIDEOS)
        picked = [names[i] for i in rng.choice(len(names), size=size, replace=False)]
        enroll_names[person] = picked[:ENROLL_VIDEOS]
        probe_names[person] = picked[ENROLL_VIDEOS:]

    enrolled = set(people)
    fakes: list[dict[str, str]] = []
    for model in FAKE_MODELS:
        rows = [
            r for r in fake_rows
            if r["변조모델"] == model and r["영상ID"] in members
            and r["소스UUID"] in enrolled and r["타겟UUID"] in enrolled
            and r["소스UUID"] != r["타겟UUID"]
        ]
        if len(rows) > FAKES_PER_MODEL:
            rows = [rows[i] for i in sorted(rng.choice(len(rows), FAKES_PER_MODEL, replace=False))]
        fakes.extend(rows)

    reader = FaceReader()
    cache = args.work.expanduser() / "embeddings"
    enroll_rows = embed_videos(
        [n for names in enroll_names.values() for n in names], members, reader, 1,
        cache / "enroll",
    )
    probe_rows = embed_videos(
        [n for names in probe_names.values() for n in names], members, reader, PROBE_FRAMES,
        cache / "probe",
    )
    fake_rows_embedded = embed_videos(
        [r["영상ID"] for r in fakes], members, reader, PROBE_FRAMES, cache / "fake"
    )

    config = load_config()
    thresholds = config.thresholds.face_similarity
    matcher = build_matcher(config.face.matcher, thresholds)
    profiles = {}
    for person in people:
        vectors = [
            row["clean"]["vector"]
            for name in enroll_names[person]
            for row in enroll_rows.get(name, [])
            if row["clean"] is not None
        ]
        if vectors:
            profiles[person] = profile(person, vectors, reader.embedder)
    everyone = list(profiles.values())

    def score_probe(vector: np.ndarray, quality: float) -> list[Any]:
        return matcher.match_many(vector, everyone, quality)

    report_conditions: dict[str, Any] = {}
    for condition in ("clean", "video_small"):
        genuine: list[float] = []
        genuine_high = genuine_candidate = 0
        rank1 = named_wrong = 0
        impostor: list[float] = []
        missed_detection = probes = extra = 0
        pixels: list[float] = []
        top_impostor: list[float] = []
        for person in profiles:
            for name in probe_names[person]:
                for row in probe_rows.get(name, []):
                    probes += 1
                    face = row[condition]
                    if face is None:
                        missed_detection += 1
                        continue
                    pixels.append(face["pixels"])
                    extra += int(face.get("extra_faces", 0) > 0)
                    ranked = score_probe(face["vector"], face["quality"])
                    own = matcher.match(face["vector"], profiles[person], face["quality"])
                    genuine.append(own.similarity)
                    genuine_high += int(own.decision == "high_confidence")
                    genuine_candidate += int(own.decision in ("high_confidence", "candidate"))
                    rank1 += int(ranked[0].matched_user_id == person)
                    named_wrong += int(
                        ranked[0].matched_user_id != person
                        and ranked[0].decision == "high_confidence"
                    )
                    others = [r.similarity for r in ranked if r.matched_user_id != person]
                    impostor.extend(others)
                    top_impostor.append(max(others))
        scored = len(genuine)
        impostors = np.asarray(impostor)
        report_conditions[condition] = {
            "probes": probes,
            "face_not_detected": missed_detection,
            "median_face_pixels": round(float(np.median(pixels)), 1) if pixels else None,
            "frames_with_another_detection": extra,
            "genuine": {
                "scored": scored,
                "min": round(float(np.min(genuine)), 4) if genuine else None,
                "p05": round(float(np.percentile(genuine, 5)), 4) if genuine else None,
                "median": round(float(np.median(genuine)), 4) if genuine else None,
                "at_or_above_high_confidence": genuine_high,
                "at_or_above_candidate": genuine_candidate,
                "rank1_among_enrolled": rank1,
            },
            "impostor": {
                "comparisons": int(impostors.size),
                "max": round(float(impostors.max()), 4) if impostors.size else None,
                "p999": round(float(np.percentile(impostors, 99.9)), 4) if impostors.size else None,
                f"at_or_above_{TAIL}": int((impostors >= TAIL).sum()),
                "at_or_above_candidate": int((impostors >= thresholds.candidate_threshold).sum()),
                "at_or_above_high_confidence": int(
                    (impostors >= thresholds.high_confidence_threshold).sum()
                ),
                "probes_named_as_someone_else_at_high_confidence": named_wrong,
            },
        }

    donor_report: dict[str, Any] = {}
    for model in FAKE_MODELS:
        for condition in ("clean", "video_small"):
            counts: Counter[str] = Counter()
            donor_similarity: list[float] = []
            for row in (r for r in fakes if r["변조모델"] == model):
                for frame in fake_rows_embedded.get(row["영상ID"], []):
                    face = frame[condition]
                    if face is None:
                        counts["no_face"] += 1
                        continue
                    ranked = score_probe(face["vector"], face["quality"])
                    top = ranked[0]
                    donor = next(r for r in ranked if r.matched_user_id == row["소스UUID"])
                    donor_similarity.append(donor.similarity)
                    counts["frames"] += 1
                    counts["donor_first"] += int(top.matched_user_id == row["소스UUID"])
                    counts["target_first"] += int(top.matched_user_id == row["타겟UUID"])
                    counts["donor_high_confidence"] += int(
                        donor.similarity >= thresholds.high_confidence_threshold
                    )
                    counts["named_donor_at_high_confidence"] += int(
                        top.matched_user_id == row["소스UUID"] and top.decision == "high_confidence"
                    )
            donor_report[f"{model}_{condition}"] = {
                **dict(counts),
                "median_donor_similarity": (
                    round(float(np.median(donor_similarity)), 4) if donor_similarity else None
                ),
            }

    report = {
        "question": "do the identity thresholds fitted on LFW hold for Korean faces, and is a "
        "swap's donor found among them?",
        "data": "KoDF validation split (AI Hub '딥페이크 변조 영상', National Information Society "
        "Agency). Subject IDs are not reported.",
        "thresholds": {
            "candidate": thresholds.candidate_threshold,
            "high_confidence": thresholds.high_confidence_threshold,
            "min_margin": thresholds.min_margin,
            "source": "LFW, 30 identities (data/results/calibration_face_similarity.json)",
        },
        "protocol": {
            "people": len(profiles),
            "enrollment": f"{ENROLL_VIDEOS} videos x 1 frame per person",
            "probes": f"{PROBE_VIDEOS} other videos x {PROBE_FRAMES} frames per person",
            "video_small": f"face shrunk to ~{SMALL_FACE:.0f} px, H.264 crf {CRF}",
            "fakes": f"up to {FAKES_PER_MODEL} per swapper whose donor and target are both "
            f"enrolled, {PROBE_FRAMES} frames each",
            "seed": args.seed,
        },
        "conditions": report_conditions,
        "swap_donor": donor_report,
        "environment": environment(config),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "korean_faces_kodf.json"
    text = json.dumps(report, indent=2, ensure_ascii=False)
    destination.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
