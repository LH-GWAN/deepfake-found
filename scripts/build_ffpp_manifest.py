r"""Turn a FaceForensics++ download into manifests the detector trainer reads.

Every detector this repository trained so far learned from fakes it made
itself, so none of them says anything about a corpus it did not generate.
FaceForensics++ is the corpus the published detectors were trained on. This
script turns a copy of it, a zip or an unpacked folder from any mirror, into
the same record format ``train_deepfake_cnn.py`` already reads, so training on
it changes nothing downstream.

What it does, per clip:

sample frames
    A few frames spread over each video (more for real clips than for each
    manipulation, so real and fake end up roughly balanced), skipping the
    first and last frame. Mirrors that ship frames instead of videos are read
    the same way, one folder per clip.
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
resumes. ``--survey`` only reports what the source holds, which is the first
thing to run on a mirror whose layout is unknown.

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
from collections.abc import Iterable, Iterator
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


def find_clips(
    names: Iterable[str], methods: Iterable[str], compression: str | None
) -> list[Clip]:
    """Group a download's files into clips of the wanted kinds."""
    wanted = set(methods) | {"real"}
    videos: list[Clip] = []
    frames: dict[tuple[str, str], list[str]] = {}
    for name in names:
        suffix = PurePosixPath(name).suffix.lower()
        if suffix not in VIDEO_SUFFIXES and suffix not in IMAGE_SUFFIXES:
            continue
        found = classify(name, compression)
        if found is None or found[0] not in wanted:
            continue
        kind, video = found
        if suffix in VIDEO_SUFFIXES:
            videos.append(Clip(kind, video, (name,), True))
        else:
            frames.setdefault((kind, video), []).append(name)
    clips = videos + [
        Clip(kind, video, tuple(sorted(members)), False)
        for (kind, video), members in frames.items()
    ]
    return sorted(clips, key=lambda clip: (clip.kind, clip.video))


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


def video_frames(path: Path, count: int) -> list[np.ndarray]:
    """Return ``count`` RGB frames spread over a video."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        frames = []
        for index in spread(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), count):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
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

    detector = build_detector(load_config().face.detector)
    groups = identity_groups(clip.video for clip in clips)
    records: dict[str, list[dict[str, str]]] = {}
    for position, clip in enumerate(clips, start=1):
        count = args.frames_real if clip.kind == "real" else args.frames_fake
        folder = args.output / clip.kind / clip.video
        wanted = [folder / f"f{index}.png" for index in range(count)]
        if not all(path.is_file() for path in wanted):
            if clip.is_video:
                with source.local(clip.members[0]) as local:
                    frames = video_frames(local, count)
            else:
                frames = [source.image(clip.members[i])
                          for i in spread(len(clip.members), count)]
            folder.mkdir(parents=True, exist_ok=True)
            for path, frame in zip(wanted, frames, strict=False):
                crop = context_crop(detector, frame, args.max_side)
                if crop is not None:
                    Image.fromarray(crop).save(path)
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
