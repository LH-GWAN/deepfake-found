"""Frame sampling: choosing which frames of a video get analysed.

Decoding and embedding every frame is waste. A 30 fps clip repeats a nearly
identical face thirty times a second, and every one of those frames would cost a
detection, an alignment, an embedding and possibly a deepfake inference.
Sampling at one or two frames per second keeps recall while cutting the work by
more than an order of magnitude, which is the trade-off research question RQ6
measures.

Two strategies are implemented:

``uniform_fps``
    Take a frame every ``1 / fps`` seconds. Predictable cost, and the sensible
    default.
``scene_change``
    Take a frame whenever the content changes materially, measured as mean
    absolute difference against the last kept frame. Spends the budget where the
    video actually changes, at the price of an unpredictable frame count.

Both respect ``max_frames`` so that a long video cannot silently turn into an
unbounded job.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from deepshield.config import VideoSamplingConfig
from deepshield.exceptions import InvalidMediaError, ModelNotAvailableError
from deepshield.logging_utils import get_logger

logger = get_logger(__name__)

SCENE_CHANGE_THRESHOLD = 12.0
SCENE_PROBE_SIDE = 64


@dataclass(frozen=True)
class SampledFrame:
    """One decoded frame together with its position in the source video."""

    image: np.ndarray
    frame_number: int
    timestamp_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping without pixel data."""
        return {
            "frame_number": self.frame_number,
            "timestamp_seconds": round(self.timestamp_seconds, 3),
            "shape": list(self.image.shape),
        }


class FrameSampler(ABC):
    """Contract for selecting which frames of a video get analysed."""

    @abstractmethod
    def sample(self, video_path: Path) -> list[SampledFrame]:
        """Return the frames selected from a video."""

    @abstractmethod
    def probe(self, video_path: Path) -> dict[str, Any]:
        """Return container metadata: duration, fps, resolution and frame count."""


def _open_capture(video_path: Path) -> tuple[Any, Any]:
    """Open a video with OpenCV, raising clear errors when that is impossible."""
    try:
        import cv2
    except ImportError as exc:
        raise ModelNotAvailableError(
            "OpenCV is not installed; install the 'video' extra: pip install -e '.[video]'"
        ) from exc

    path = Path(video_path)
    if not path.is_file():
        raise InvalidMediaError(f"video not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise InvalidMediaError(f"could not open video: {path}")
    return cv2, capture


class OpenCvFrameSampler(FrameSampler):
    """Frame sampling backed by OpenCV's video decoder."""

    name = "opencv"

    def __init__(self, config: VideoSamplingConfig | None = None) -> None:
        """Store the sampling policy."""
        self.config = config or VideoSamplingConfig()

    def probe(self, video_path: Path) -> dict[str, Any]:
        """Return container metadata without decoding the whole file."""
        cv2, capture = _open_capture(video_path)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        return {
            "fps": fps,
            "frame_count": frames,
            "width": width,
            "height": height,
            "duration_seconds": round(frames / fps, 3) if fps > 0 else None,
        }

    @staticmethod
    def _probe_plane(frame: np.ndarray) -> np.ndarray:
        """Return a small grayscale version used for scene-change comparison."""
        small = frame[:: max(1, frame.shape[0] // SCENE_PROBE_SIDE),
                      :: max(1, frame.shape[1] // SCENE_PROBE_SIDE)]
        return small.astype(np.float32).mean(axis=2)

    def sample(self, video_path: Path) -> list[SampledFrame]:
        """Decode a video and return the frames the policy selects.

        Raises:
            InvalidMediaError: If the file cannot be opened or holds no frames.

        """
        cv2, capture = _open_capture(video_path)
        sampled: list[SampledFrame] = []
        try:
            source_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
            if source_fps <= 0:
                logger.warning("video reports no frame rate; assuming 25 fps for sampling")
                source_fps = 25.0
            stride = max(1, int(round(source_fps / self.config.fps)))

            previous: np.ndarray | None = None
            index = 0
            while len(sampled) < self.config.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break

                keep = False
                if self.config.strategy == "uniform_fps":
                    keep = index % stride == 0
                elif self.config.strategy == "scene_change":
                    plane = self._probe_plane(frame)
                    if previous is None:
                        keep = True
                    else:
                        keep = float(np.mean(np.abs(plane - previous))) >= SCENE_CHANGE_THRESHOLD
                    if keep:
                        previous = plane
                else:
                    keep = index % stride == 0

                if keep:
                    rgb = np.ascontiguousarray(frame[:, :, ::-1])
                    sampled.append(
                        SampledFrame(
                            image=rgb,
                            frame_number=index,
                            timestamp_seconds=index / source_fps,
                        )
                    )
                index += 1
        finally:
            capture.release()

        if not sampled:
            raise InvalidMediaError(f"no frames could be decoded from {video_path}")
        logger.info(
            "sampled %d frames from %s using %s",
            len(sampled),
            Path(video_path).name,
            self.config.strategy,
        )
        return sampled


def build_sampler(config: VideoSamplingConfig) -> FrameSampler:
    """Instantiate the configured frame sampler."""
    return OpenCvFrameSampler(config)
