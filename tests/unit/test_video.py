"""Phase 6: frame sampling, tracking and representative-frame selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepshield.config import RepresentativeFrameConfig, VideoSamplingConfig, VideoTrackingConfig
from deepshield.exceptions import InvalidMediaError
from deepshield.types import BoundingBox, DetectedFace
from deepshield.video.sampler import OpenCvFrameSampler
from deepshield.video.tracker import (
    IouFaceTracker,
    appearance_descriptor,
    appearance_similarity,
    crop_of,
    frontality,
)

cv2 = pytest.importorskip("cv2")


def write_video(path: Path, frames: int = 50, fps: float = 25.0, size: int = 160) -> Path:
    """Write a synthetic clip whose two halves look clearly different."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    for index in range(frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        frame[:, :, index * 3 // frames] = 200
        frame[20:60, 20:60] = 255 if index < frames // 2 else 30
        writer.write(frame)
    writer.release()
    return path


def face(x: float, confidence: float = 0.9, landmarks: np.ndarray | None = None) -> DetectedFace:
    return DetectedFace(BoundingBox(x, 0, x + 50, 50), confidence, landmarks=landmarks)


def test_sampler_probe_reports_metadata(tmp_path: Path) -> None:
    path = write_video(tmp_path / "clip.mp4")
    metadata = OpenCvFrameSampler().probe(path)
    assert metadata["fps"] == pytest.approx(25.0, abs=1.0)
    assert metadata["width"] == 160
    assert metadata["duration_seconds"] is not None


def test_uniform_sampling_reduces_the_frame_count(tmp_path: Path) -> None:
    path = write_video(tmp_path / "clip.mp4", frames=50, fps=25.0)
    frames = OpenCvFrameSampler(VideoSamplingConfig(fps=1.0)).sample(path)
    assert 1 <= len(frames) <= 4
    assert all(frame.image.ndim == 3 for frame in frames)


def test_higher_sampling_rate_yields_more_frames(tmp_path: Path) -> None:
    path = write_video(tmp_path / "clip.mp4", frames=50, fps=25.0)
    slow = OpenCvFrameSampler(VideoSamplingConfig(fps=1.0)).sample(path)
    fast = OpenCvFrameSampler(VideoSamplingConfig(fps=5.0)).sample(path)
    assert len(fast) > len(slow)


def test_max_frames_caps_the_job(tmp_path: Path) -> None:
    path = write_video(tmp_path / "clip.mp4", frames=50, fps=25.0)
    frames = OpenCvFrameSampler(VideoSamplingConfig(fps=25.0, max_frames=3)).sample(path)
    assert len(frames) == 3


def test_scene_change_strategy_runs(tmp_path: Path) -> None:
    path = write_video(tmp_path / "clip.mp4", frames=40, fps=25.0)
    frames = OpenCvFrameSampler(
        VideoSamplingConfig(strategy="scene_change", fps=1.0)
    ).sample(path)
    assert frames


def test_timestamps_increase(tmp_path: Path) -> None:
    path = write_video(tmp_path / "clip.mp4", frames=50, fps=25.0)
    frames = OpenCvFrameSampler(VideoSamplingConfig(fps=5.0)).sample(path)
    stamps = [frame.timestamp_seconds for frame in frames]
    assert stamps == sorted(stamps)


def test_missing_video_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(InvalidMediaError, match="not found"):
        OpenCvFrameSampler().sample(tmp_path / "missing.mp4")


def test_corrupt_video_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"definitely not a video")
    with pytest.raises(InvalidMediaError):
        OpenCvFrameSampler().sample(path)


def test_tracker_groups_a_moving_face() -> None:
    tracks = IouFaceTracker().track([[face(0)], [face(6)], [face(12)]])
    assert len(tracks) == 1
    assert len(tracks[0].faces) == 3


def test_tracker_separates_distant_faces() -> None:
    tracks = IouFaceTracker().track([[face(0), face(300)], [face(5), face(305)]])
    assert len(tracks) == 2


def test_tracker_starts_a_new_track_after_a_long_gap() -> None:
    tracker = IouFaceTracker(VideoTrackingConfig(max_gap_frames=0))
    tracks = tracker.track([[face(0)], [], [face(0)]])
    assert len(tracks) == 2


def test_appearance_check_splits_different_people_in_the_same_framing() -> None:
    """The dangerous failure is merging two people; geometry alone does exactly that."""
    size = 80
    bright = np.full((size, size, 3), 220, dtype=np.uint8)
    dark = np.zeros((size, size, 3), dtype=np.uint8)
    detections = [[face(10)], [face(10)], [face(10)], [face(10)]]
    frames = [bright, bright, dark, dark]

    geometry_only = IouFaceTracker(VideoTrackingConfig(appearance_threshold=0.0))
    with_appearance = IouFaceTracker(VideoTrackingConfig(appearance_threshold=0.75))
    assert len(geometry_only.track(detections, frames)) == 1
    assert len(with_appearance.track(detections, frames)) == 2


def test_tracking_without_frames_falls_back_to_geometry() -> None:
    assert len(IouFaceTracker().track([[face(0)], [face(4)]])) == 1


def test_appearance_similarity_is_permissive_when_unavailable() -> None:
    assert appearance_similarity(None, np.ones(4)) == 1.0


def test_appearance_descriptor_is_normalised() -> None:
    descriptor = appearance_descriptor(np.full((20, 20, 3), 100, dtype=np.uint8))
    assert float(descriptor.sum()) == pytest.approx(1.0)


def test_appearance_descriptor_handles_empty_crops() -> None:
    assert appearance_descriptor(np.zeros((0, 0, 3), dtype=np.uint8)).sum() == 0.0


def test_crop_of_clamps_to_the_frame() -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    crop = crop_of(image, BoundingBox(-20, -20, 500, 500))
    assert crop.shape[0] <= 40 and crop.shape[1] <= 40 and crop.size > 0


def test_representative_defaults_to_highest_confidence() -> None:
    tracks = IouFaceTracker().track([[face(0, 0.5)], [face(2, 0.95)], [face(4, 0.7)]])
    assert tracks[0].representative.detection_confidence == pytest.approx(0.95)


def test_representative_can_use_frontality() -> None:
    frontal = np.array([[0.0, 0.0], [20.0, 0.0], [10.0, 10.0], [2.0, 20.0], [18.0, 20.0]])
    turned = np.array([[0.0, 0.0], [20.0, 0.0], [18.0, 10.0], [2.0, 20.0], [18.0, 20.0]])
    tracker = IouFaceTracker(
        representative=RepresentativeFrameConfig(criterion="frontal_pose")
    )
    tracks = tracker.track([[face(0, 0.99, turned)], [face(2, 0.5, frontal)]])
    assert tracks[0].representative_index == 1


def test_frontality_is_neutral_without_landmarks() -> None:
    assert frontality(face(0)) == 0.5


def test_track_dict_is_serialisable() -> None:
    payload = IouFaceTracker().track([[face(0)]])[0].to_dict()
    assert payload["length"] == 1
    assert payload["track_id"] == 0
