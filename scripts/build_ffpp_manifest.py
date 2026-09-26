r"""Turn a FaceForensics++ download into manifests the detector trainer reads.

Every detector this repository trained so far learned from fakes it made
itself, so none of them says anything about a corpus it did not generate.
FaceForensics++ is the corpus the published detectors were trained on. This
script turns a copy of it, a zip or an unpacked folder from any mirror, into
the same record format ``train_deepfake_cnn.py`` already reads, so training on
it changes nothing downstream.

What it does, per clip:

sample frames
    A few frames spread over the first half of each video (more for real
    clips than for each manipulation, so real and fake end up roughly
    balanced), read in one pass rather than by seeking. Mirrors that ship
    frames instead of videos are read the same way, one folder per clip.
crop loosely
    The project's own face detector finds the largest face and the crop keeps
    a margin wider than the pipeline's, so the trainer can find the face again
    and cut it exactly as the analysis pipeline would. Crops are capped in size
    and saved losslessly.
group identities
    FaceForensics++ names a manipulation ``000_003``: the scene of clip 000 with
    the face or the expressions of clip 003. Every clip joined by such a name
    is one identity, so the trainer's identity hold-out can never train on one
    of the two people and test on the other.

Reading from a zip extracts one video at a time into a temporary file and
deletes it straight after, so the disk never holds more than the zip, the
crops and one video. Crops already on disk are kept, so an interrupted run
resumes; the settings that shape a crop are written next to them, and a run
with different settings refuses to mix its crops with the old ones. A clip
found twice (a mask video beside the real one, or two compressions under
``--compression any``) is kept once. ``--survey`` only reports what the source
holds, which is the first thing to run on a mirror whose layout is unknown.

Usage:
    python scripts/build_ffpp_manifest.py --source ffpp.zip --survey
    python scripts/build_ffpp_manifest.py --source ffpp.zip --output data/test/ffpp
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

METHODS: dict[str, str] = {
    "deepfakedetection": "DeepFakeDetection",
    "neuraltextures": "NeuralTextures",
    "faceshifter": "FaceShifter",
    "face2face": "Face2Face",
    "deepfakes": "Deepfakes",
    "faceswap": "FaceSwap",
}
DEFAULT_METHODS = ("deepfakes", "face2face", "faceswap", "neuraltextures")
REAL_MARKERS = ("original", "real", "youtube")
COMPRESSIONS = ("raw", "c0", "c23", "c40")
# Under --compression any, the clip kept when a mirror ships several: the
# benchmark's own c23 first, then the lossless copies, then c40.
COMPRESSION_PREFERENCE = ("c23", "c0", "raw", "c40")
MASK_FOLDERS = ("mask", "masks")
PARAMETERS_FILE = "crop_parameters.json"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
CONTEXT_MARGIN = 0.6
VIDEO_ID = re.compile(r"(?<!\d)(\d{3})(?:_(\d{3}))?(?!\d)")


@dataclass(frozen=True)
class Clip:
    """One real or manipulated video, as a video file or a folder of its frames."""

    kind: str
    video: str
    members: tuple[str, ...]
    is_video: bool


def _normalise(part: str) -> str:
    return re.sub(r"[^a-z0-9]", "", part.lower())


def classify(name: str, compression: str | None = "c23") -> tuple[str, str] | None:
    """Return ``(kind, video id)`` for one path in a download, or ``None`` to skip it.

    ``kind`` is ``"real"`` or a key of :data:`METHODS`. A path naming a
    compression other than the one asked for is skipped; a mirror that names
    none is taken as it is. The manipulation is read from any folder or from
    the file name, because mirrors disagree about where they put it.
    """
    path = PurePosixPath(name)
    parts = [_normalise(part) for part in path.parts]
    if any(part in MASK_FOLDERS for part in parts[:-1]):
        # FaceForensics++ ships each manipulation's binary masks as videos with
        # the clip's own name; they are not faces.
        return None
    if compression is not None:
        named = {part for part in parts for level in COMPRESSIONS if part == level}
        named |= {level for part in parts for level in COMPRESSIONS if part.endswith(level)}
        if named and compression not in named:
            return None
    method = next(
        (key for part in parts for key in METHODS if part.startswith(key)), None
    )
    if method is None and not any(
        part.startswith(marker) for part in parts for marker in REAL_MARKERS
    ):
        return None
    stem_match = VIDEO_ID.search(path.stem)
    folder_match = VIDEO_ID.search(path.parent.name)
    match = stem_match if path.suffix.lower() in VIDEO_SUFFIXES else folder_match or stem_match
    if match is None:
        return None
    video = match.group(0)
    return (method or "real"), video


def _compression_rank(name: str) -> int:
    """Rank a path by the compression it names, best first; unnamed ranks last."""
    parts = [_normalise(part) for part in PurePosixPath(name).parts]
    for rank, level in enumerate(COMPRESSION_PREFERENCE):
        if any(part == level or part.endswith(level) for part in parts):
            return rank
    return len(COMPRESSION_PREFERENCE)


def find_clips(
    names: Iterable[str], methods: Iterable[str], compression: str | None
) -> list[Clip]:
    """Group a download's files into clips of the wanted kinds, one clip per video.

    A mirror can hold the same clip more than once: as a video and as a folder
    of frames, or, under ``--compression any``, at several compressions. Each
    copy would add the clip's frames again and weight that clip, and so that
    person, twice, so one is kept: a video over a frame folder, then the
    preferred compression, then the first path in sorted order.
    """
    wanted = set(methods) | {"real"}
    videos: dict[tuple[str, str], list[str]] = {}
    frames: dict[tuple[str, str], dict[str, list[str]]] = {}
    for name in names:
        suffix = PurePosixPath(name).suffix.lower()
        if suffix not in VIDEO_SUFFIXES and suffix not in IMAGE_SUFFIXES:
            continue
        found = classify(name, compression)
        if found is None or found[0] not in wanted:
            continue
        if suffix in VIDEO_SUFFIXES:
            videos.setdefault(found, []).append(name)
        else:
            folder = PurePosixPath(name).parent.as_posix()
            frames.setdefault(found, {}).setdefault(folder, []).append(name)
    clips = []
    for key in sorted(set(videos) | set(frames)):
        if key in videos:
            best = min(videos[key], key=lambda name: (_compression_rank(name), name))
            clips.append(Clip(key[0], key[1], (best,), True))
        else:
            folder = min(frames[key], key=lambda name: (_compression_rank(name), name))
            clips.append(Clip(key[0], key[1], tuple(sorted(frames[key][folder])), False))
    return clips


def identity_groups(videos: Iterable[str]) -> dict[str, str]:
    """Map every clip number to one identity shared by all clips a fake name joins."""
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for video in videos:
        numbers = video.split("_")
        for number in numbers:
            parent.setdefault(number, number)
        if len(numbers) == 2:
            first, second = find(numbers[0]), find(numbers[1])
            if first != second:
                parent[max(first, second)] = min(first, second)
    return {number: f"ffpp_{find(number)}" for number in parent}


def spread(total: int, count: int) -> list[int]:
    """Return ``count`` indices spread over ``total`` frames, avoiding both ends."""
    if total <= 0 or count <= 0:
        return []
    if total <= count:
        return list(range(total))
    return sorted({int(round(i)) for i in np.linspace(0, total - 1, count + 2)[1:-1]})


class Source:
    """A download read the same way whether it is a zip or an unpacked folder."""

    def __init__(self, path: Path) -> None:
        """Open a zip or remember a folder."""
        self.path = path
        self.archive = zipfile.ZipFile(path) if path.is_file() else None

    def names(self) -> list[str]:
        """Return every file in the download, as posix paths."""
        if self.archive is not None:
            return [info.filename for info in self.archive.infolist() if not info.is_dir()]
        return [p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()]

    def image(self, name: str) -> np.ndarray:
        """Return one stored frame as RGB."""
        if self.archive is not None:
            data = self.archive.read(name)
        else:
            data = (self.path / name).read_bytes()
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))

    @contextmanager
    def local(self, name: str) -> Iterator[Path]:
        """Yield a file path for one member, extracting it from a zip only while in use."""
        if self.archive is None:
            yield self.path / name
            return
        folder = Path(tempfile.mkdtemp(prefix="ffpp_"))
        target = folder / PurePosixPath(name).name
        try:
            with self.archive.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            yield target
        finally:
            shutil.rmtree(folder, ignore_errors=True)


def frame_total(reported: int, count: Callable[[], int]) -> tuple[int, bool]:
    """Return a video's frame count and whether its header had to be bypassed.

    Some containers report zero or a negative number of frames. Sampling from
    that would read one frame and move on, so the frames are counted by
    decoding instead, which costs one extra pass over that video only.
    """
    if reported > 0:
        return reported, False
    return max(0, count()), True


def video_frames(
    path: Path, count: int, window: float = 0.5, notes: Counter[str] | None = None
) -> list[np.ndarray]:
    """Return ``count`` RGB frames spread over the first ``window`` of a video.

    The video is read once from the start and stops at the last frame wanted.
    Seeking looks cheaper but is not: on these H.264 files OpenCV decodes from
    the previous keyframe on every seek, which measured 0.74 s per frame
    against 0.17 s for the face detector, so eight seeks cost several full
    decodes. Sampling the first half keeps a clip's frames distinct while
    halving what has to be decoded. A header without a usable frame count is
    counted by decoding and tallied in ``notes`` under ``frame_count_unknown``.
    """
    import cv2

    def decode_count() -> int:
        counter = cv2.VideoCapture(str(path))
        try:
            frames = 0
            while counter.grab():
                frames += 1
            return frames
        finally:
            counter.release()

    capture = cv2.VideoCapture(str(path))
    try:
        total, counted = frame_total(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), decode_count)
        if counted:
            print(f"warning: {path.name} reports no frame count; counted {total} by decoding",
                  flush=True)
            if notes is not None:
                notes["frame_count_unknown"] += 1
        wanted = set(spread(max(1, int(total * window)), count))
        frames = []
        for index in range(max(wanted, default=-1) + 1):
            if not capture.grab():
                break
            if index in wanted:
                ok, frame = capture.retrieve()
                if ok:
                    frames.append(np.ascontiguousarray(frame[:, :, ::-1]))
        return frames
    finally:
        capture.release()


def context_crop(detector: Any, image: np.ndarray, max_side: int) -> np.ndarray | None:
    """Return a loose crop around the largest face, no longer than ``max_side``."""
    from deepshield.pipeline.analysis_pipeline import crop_with_margin

    faces = detector.detect(image)
    if not faces:
        return None
    face = max(faces, key=lambda f: f.bbox.width * f.bbox.height)
    crop = Image.fromarray(crop_with_margin(image, face, CONTEXT_MARGIN))
    if max(crop.size) > max_side:
        scale = max_side / max(crop.size)
        crop = crop.resize(
            (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(crop)


def check_parameters(output: Path, parameters: dict[str, Any]) -> None:
    """Refuse to add crops to a folder whose crops were made another way.

    Crops on disk are reused so an interrupted run resumes, which is only
    right when they were cut with the same settings. The settings are written
    beside the crops on the first run and compared on every later one.

    Raises:
        SystemExit: If the folder's recorded settings differ from these.

    """
    record = output / PARAMETERS_FILE
    if record.is_file():
        previous = json.loads(record.read_text(encoding="utf-8"))
        changed = {
            key: (previous.get(key), value)
            for key, value in parameters.items() if previous.get(key) != value
        }
        if changed:
            detail = "; ".join(f"{key}: {old!r} -> {new!r}" for key, (old, new) in changed.items())
            raise SystemExit(
                f"{output} holds crops made with other settings ({detail}); "
                "choose another --output or remove the old crops first"
            )
        return
    if any(output.glob("*/*/f*.png")):
        print(f"warning: {output} holds crops from a run that did not record its settings; "
              "they are reused as if they matched these", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")


def survey(clips: list[Clip], names: list[str], compression: str | None) -> dict[str, Any]:
    """Summarise what a download holds before anything is extracted."""
    kinds = Counter(clip.kind for clip in clips)
    skipped = [
        name for name in names
        if PurePosixPath(name).suffix.lower() in VIDEO_SUFFIXES | IMAGE_SUFFIXES
        and classify(name, compression) is None
    ]
    return {
        "files": len(names),
        "clips": dict(sorted(kinds.items())),
        "as_videos": sum(clip.is_video for clip in clips),
        "as_frame_folders": sum(not clip.is_video for clip in clips),
        "fakes_without_a_pair": sum(
            1 for clip in clips if clip.kind != "real" and "_" not in clip.video
        ),
        "examples": {kind: next(c.members[0] for c in clips if c.kind == kind) for kind in kinds},
        "media_files_not_recognised": len(skipped),
        "unrecognised_examples": skipped[:5],
    }


def main(argv: list[str] | None = None) -> int:
    """Sample, crop and write one manifest per manipulation, each with the real clips."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="zip or unpacked folder")
    parser.add_argument("--output", type=Path, default=Path("data/test/ffpp"))
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS),
                        choices=sorted(METHODS))
    parser.add_argument("--compression", default="c23",
                        help="keep paths naming this compression; 'any' keeps all")
    parser.add_argument("--frames-real", type=int, default=8)
    parser.add_argument("--frames-fake", type=int, default=2)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--window", type=float, default=0.5,
                        help="sample frames from this leading fraction of each video")
    parser.add_argument("--limit", type=int, default=None, help="clips per kind, for a trial")
    parser.add_argument("--survey", action="store_true", help="report the layout and stop")
    args = parser.parse_args(argv)

    compression = None if args.compression == "any" else args.compression
    source = Source(args.source)
    names = source.names()
    clips = find_clips(names, args.methods, compression)
    if args.survey:
        print(json.dumps(survey(clips, names, compression), indent=2))
        return 0
    if not clips:
        raise SystemExit("no clips recognised; run with --survey to see the layout")
    if args.limit:
        per_kind: Counter[str] = Counter()
        kept = []
        for clip in clips:
            if per_kind[clip.kind] < args.limit:
                per_kind[clip.kind] += 1
                kept.append(clip)
        clips = kept

    from deepshield.config import load_config
    from deepshield.face.detector import build_detector

    detector_config = load_config().face.detector
    check_parameters(args.output, {
        "compression": args.compression,
        "frames_real": args.frames_real,
        "frames_fake": args.frames_fake,
        "max_side": args.max_side,
        "window": args.window,
        "context_margin": CONTEXT_MARGIN,
        "face_detector": detector_config.backend,
    })
    detector = build_detector(detector_config)
    notes: Counter[str] = Counter()
    groups = identity_groups(clip.video for clip in clips)
    records: dict[str, list[dict[str, str]]] = {}
    for position, clip in enumerate(clips, start=1):
        count = args.frames_real if clip.kind == "real" else args.frames_fake
        folder = args.output / clip.kind / clip.video
        wanted = [folder / f"f{index}.png" for index in range(count)]
        if not all(path.is_file() for path in wanted):
            if clip.is_video:
                with source.local(clip.members[0]) as local:
                    frames = video_frames(local, count, args.window, notes)
            else:
                frames = [source.image(clip.members[i])
                          for i in spread(len(clip.members), count)]
            folder.mkdir(parents=True, exist_ok=True)
            for path, frame in zip(wanted, frames, strict=False):
                crop = context_crop(detector, frame, args.max_side)
                if crop is not None:
                    # Written under another name and renamed, so a snapshot taken
                    # mid-run never holds a half-written crop under its real name.
                    staging = path.with_suffix(".partial")
                    Image.fromarray(crop).save(staging, format="PNG")
                    staging.replace(path)
        for index, path in enumerate(wanted):
            if path.is_file():
                records.setdefault(clip.kind, []).append({
                    "path": path.as_posix(),
                    "label": "real" if clip.kind == "real" else "fake",
                    "identity": groups[clip.video.split("_")[0]],
                    "source": f"ffpp/{clip.kind}/{clip.video}/f{index}",
                })
        if position % 100 == 0 or position == len(clips):
            print(f"{position}/{len(clips)} clips, {sum(map(len, records.values()))} crops",
                  flush=True)

    if notes["frame_count_unknown"]:
        print(f"{notes['frame_count_unknown']} videos reported no frame count and were "
              "counted by decoding", flush=True)
    reals = records.get("real", [])
    for method in args.methods:
        fakes = records.get(method, [])
        if not fakes:
            print(f"no {METHODS[method]} crops; skipping its manifest")
            continue
        manifest = {
            "method": f"FaceForensics++ {METHODS[method]}, faces sampled from videos",
            "swapper": f"ffpp_{method}",
            "source_dataset": "FaceForensics++",
            "compression": args.compression,
            "frames_per_clip": {"real": args.frames_real, "fake": args.frames_fake},
            "videos_without_frame_count": notes["frame_count_unknown"],
            "covers": [f"{METHODS[method]} manipulations as FaceForensics++ ships them"],
            "does_not_cover": [
                "any generator outside FaceForensics++",
                "this repository's own graphics, GAN and diffusion fakes",
            ],
            "records": reals + fakes,
        }
        destination = args.output / f"manifest_{method}.json"
        destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {destination}: {len(reals)} real, {len(fakes)} {METHODS[method]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
