"""Measure whether the video pipeline finds the user's face swapped into a video.

This is the project's original question asked of video: "was my face used in
this clip?". Every clip here is built from the repository's sample clip, two
scenes of 75 frames at 25 fps, by running the GAN swapper ``inswapper_128`` on
each frame with one evaluation identity as the donor, then re-encoding with
H.264 the way a clip is distributed. Each identity is enrolled from its other
photographs and the clips are analysed through the full video pipeline for
that user:

original        the untouched clip, a control that must stay ``unrelated``
full            every frame swapped
second_scene    only the second scene swapped, a separate face track
partial_2s      two seconds in the middle of the first scene swapped
partial_1s      one second in the middle of the first scene swapped
partial_half_s  half a second in the middle of the first scene swapped
full_crf35      every frame swapped, then heavily compressed

The partial clips are the case that matters: the swapped span sits inside a
track whose other frames are genuine, which is how a spliced deepfake looks.
A swap shorter than the sampling interval can fall between sampled frames
entirely; the per-situation ``sampled`` count separates that cost of sampling
from a failure to recognise a frame that was sampled.

Swapped frames are cached under ``--workdir`` so repeated runs, for example
before and after a pipeline change, analyse byte-identical clips.

The default clip is a slideshow of two still portraits with slow zooms, not
footage with motion, lighting change or occlusion. It exercises sampling,
tracking and identity; it does not stand in for real video. ``--clip`` takes
any other video; a clip that is one continuous shot skips ``second_scene`` and
places the partial swaps around its middle.

Usage:
    python scripts/evaluate_video_verdicts.py --identities 10
    python scripts/evaluate_video_verdicts.py --fps 2 --tag fps2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_verdicts import GanSwapper, enroll, isolated, photographs

from deepshield.config import DeepShieldConfig, load_config
from deepshield.experiments import environment
from deepshield.media import load_image
from deepshield.pipeline.analysis_pipeline import DefaultAnalysisPipeline
from deepshield.video.processor import DefaultVideoProcessor

EXPECTED_UNRELATED = {"original"}
SCENE_CUT_DIFFERENCE = 30.0


def read_frames(path: Path) -> tuple[list[np.ndarray], float]:
    """Decode every frame of a video as RGB."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(np.ascontiguousarray(frame[:, :, ::-1]))
    capture.release()
    return frames, fps


def encode(frames: list[np.ndarray], fps: float, destination: Path, crf: int) -> Path:
    """Write frames as H.264 at the given constant rate factor."""
    import cv2

    raw = destination.with_suffix(".raw.mp4")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(np.ascontiguousarray(frame[:, :, ::-1]))
    writer.release()
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
            "-crf", str(crf), "-pix_fmt", "yuv420p", str(destination),
        ],
        check=True,
    )
    raw.unlink()
    return destination


def scene_cut(frames: list[np.ndarray]) -> int | None:
    """Return the first frame after the largest content jump, or ``None`` for one shot."""
    differences = [
        float(np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)).mean())
        for i in range(1, len(frames))
    ]
    if max(differences) < SCENE_CUT_DIFFERENCE:
        return None
    return int(np.argmax(differences)) + 1


def swapped_frames(
    frames: list[np.ndarray], donor: np.ndarray, swapper: GanSwapper, cache: Path
) -> tuple[list[np.ndarray], int]:
    """Return every frame with the donor's identity swapped in, cached on disk."""
    if cache.is_file():
        stored = np.load(cache)
        return list(stored["frames"]), int(stored["failed"])
    out, failed = [], 0
    for frame in frames:
        result = swapper(donor, frame)
        if result is None:
            failed += 1
            result = frame
        out.append(result)
    np.savez_compressed(cache, frames=np.stack(out), failed=failed)
    return out, failed


def situations(
    frames: list[np.ndarray], swapped: list[np.ndarray], fps: float
) -> dict[str, tuple[list[np.ndarray], tuple[int, int] | None, int]]:
    """Return each situation's frames, its swapped span and its encoding quality."""
    cut = scene_cut(frames)
    middle = (cut if cut is not None else len(frames)) // 2

    def splice(start: int, stop: int) -> list[np.ndarray]:
        return [swapped[i] if start <= i < stop else frames[i] for i in range(len(frames))]

    def span(seconds: float) -> tuple[int, int]:
        half = int(round(fps * seconds / 2))
        return middle - half, middle + half

    spans = {name: span(seconds) for name, seconds in
             (("partial_2s", 2.0), ("partial_1s", 1.0), ("partial_half_s", 0.5))}
    built: dict[str, tuple[list[np.ndarray], tuple[int, int] | None, int]] = {
        "original": (frames, None, 20),
        "full": (swapped, (0, len(frames)), 20),
    }
    if cut is not None:
        built["second_scene"] = (splice(cut, len(frames)), (cut, len(frames)), 20)
    built.update({name: (splice(*bounds), bounds, 20) for name, bounds in spans.items()})
    built["full_crf35"] = (swapped, (0, len(frames)), 35)
    return built


def sampled_in(span: tuple[int, int] | None, fps: float, sample_fps: float) -> int:
    """Count the sampled frames inside the swapped span, using the sampler's stride."""
    if span is None:
        return 0
    stride = max(1, int(round(fps / sample_fps)))
    return sum(1 for index in range(0, span[1], stride) if index >= span[0])


def main(argv: list[str] | None = None) -> int:
    """Swap each identity into the sample clip and tabulate the video verdicts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, default=Path("data/test/sample_clip.mp4"))
    parser.add_argument("--faces", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument("--models", type=Path, default=Path("models/insightface"))
    parser.add_argument(
        "--inswapper", type=Path, default=Path("models/inswapper/inswapper_128.onnx")
    )
    parser.add_argument("--identities", type=int, default=10)
    parser.add_argument("--fps", type=float, default=None, help="override the sampling rate")
    parser.add_argument("--workdir", type=Path, default=None, help="cache for swapped frames")
    parser.add_argument("--tag", default=None, help="suffix for the report file name")
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)

    if not args.clip.is_file():
        raise SystemExit(f"missing {args.clip}")
    frames, fps = read_frames(args.clip)
    grouped = photographs(args.faces)
    names = sorted(grouped)[: args.identities]
    swapper = GanSwapper(args.models, args.inswapper)

    outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    sampled: dict[str, list[int]] = defaultdict(list)
    seconds: dict[str, list[float]] = defaultdict(list)
    swap_failures = 0

    with tempfile.TemporaryDirectory() as scratch:
        workspace = Path(scratch)
        cache_dir = args.workdir or workspace
        cache_dir.mkdir(parents=True, exist_ok=True)
        base: DeepShieldConfig = isolated(load_config(), workspace)
        if args.fps is not None:
            base = base.model_copy(
                update={
                    "video": base.video.model_copy(
                        update={
                            "sampling": base.video.sampling.model_copy(update={"fps": args.fps})
                        }
                    )
                }
            )
        sample_fps = base.video.sampling.fps
        pipeline = DefaultAnalysisPipeline(base)
        processor = DefaultVideoProcessor(base, analysis=pipeline)

        for index, user in enumerate(names):
            own = grouped[user]
            if not enroll(pipeline, user, own[:-1]):
                continue
            swapped, failed = swapped_frames(
                frames, load_image(own[-1]), swapper, cache_dir / f"{user}.npz"
            )
            swap_failures += failed
            for situation, (clip, span, crf) in situations(frames, swapped, fps).items():
                path = encode(clip, fps, workspace / f"{user}_{situation}.mp4", crf)
                started = time.perf_counter()
                record = processor.analyze(path, user)
                seconds[situation].append(time.perf_counter() - started)
                assert record.risk is not None
                outcomes[situation][record.risk.verdict.value] += 1
                sampled[situation].append(sampled_in(span, fps, sample_fps))
            print(f"{index + 1}/{len(names)} {user}", flush=True)

    table = {}
    for situation, verdicts in outcomes.items():
        expected = "unrelated" if situation in EXPECTED_UNRELATED else "identity_match"
        table[situation] = {
            "expected": expected,
            "cases": sum(verdicts.values()),
            "as_expected": verdicts[expected],
            "verdicts": dict(verdicts),
            "swapped_frames_sampled": int(np.min(sampled[situation])),
            "mean_seconds": round(float(np.mean(seconds[situation])), 3),
        }
    report = {
        "question": "does the video pipeline find the user's face swapped into a clip?",
        "clip": args.clip.name,
        "clip_frames": len(frames),
        "clip_fps": fps,
        "sampling_fps": sample_fps,
        "identities": len(names),
        "frames_the_swapper_could_not_process": swap_failures,
        "situations": table,
        "environment": environment(load_config()),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    destination = args.output / f"video_verdicts{suffix}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"\n{'situation':16s} {'expected':15s} {'as expected':>12s}  sampled  seconds")
    for situation, row in table.items():
        print(
            f"{situation:16s} {row['expected']:15s} {row['as_expected']:>5d}/{row['cases']:<6d}"
            f"  {row['swapped_frames_sampled']:>7d}  {row['mean_seconds']:>7.2f}"
        )
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
