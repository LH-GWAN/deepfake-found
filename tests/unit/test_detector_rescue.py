"""Detector rescue passes: enlarge undersized images, pad frame-filling faces.

A fake backend stands in for YuNet. It only reports a face when the image it is
handed is at least a configured size and the face does not touch the frame
edge, which is the shape of the two measured failures the rescue exists for.
"""

from __future__ import annotations

import numpy as np
import pytest

from deepshield.config import FaceDetectorConfig
from deepshield.face.detector import FaceDetector
from deepshield.types import BoundingBox, DetectedFace


class PickyDetector(FaceDetector):
    """Reports one face only when the frame is big enough and the face has margin."""

    name = "picky"

    def __init__(self, config: FaceDetectorConfig, needs_side: int = 160) -> None:
        """Remember the size below which this backend refuses to see a face."""
        self.config = config
        self.needs_side = needs_side
        self.calls: list[tuple[int, int]] = []

    def _detect_once(self, image: np.ndarray) -> list[DetectedFace]:
        height, width = image.shape[:2]
        self.calls.append((height, width))
        if min(height, width) < self.needs_side:
            return []
        margin = 0.15
        box = BoundingBox(
            width * margin, height * margin, width * (1 - margin), height * (1 - margin)
        )
        landmarks = np.asarray(
            [[width * 0.4, height * 0.4], [width * 0.6, height * 0.4]], np.float32
        )
        return [DetectedFace(bbox=box, detection_confidence=0.9, landmarks=landmarks)]


def test_small_image_is_enlarged_and_boxes_mapped_back() -> None:
    detector = PickyDetector(FaceDetectorConfig(rescue_min_side=160, rescue_pad_fraction=0.0))
    faces = detector.detect(np.zeros((62, 62, 3), dtype=np.uint8))
    assert len(faces) == 1
    assert detector.calls == [(62, 62), (160, 160)]
    box = faces[0].bbox
    assert box.x2 <= 62.0 and box.y2 <= 62.0
    assert faces[0].landmarks is not None and faces[0].landmarks.max() <= 62.0


def test_padding_is_tried_after_enlarging() -> None:
    class EdgeShy(PickyDetector):
        def _detect_once(self, image: np.ndarray) -> list[DetectedFace]:
            height, width = image.shape[:2]
            self.calls.append((height, width))
            if (height, width) != (300, 300):
                return []
            return [DetectedFace(bbox=BoundingBox(50, 50, 250, 250), detection_confidence=0.8)]

    detector = EdgeShy(FaceDetectorConfig(rescue_min_side=160, rescue_pad_fraction=0.25))
    faces = detector.detect(np.zeros((200, 200, 3), dtype=np.uint8))
    assert detector.calls == [(200, 200), (300, 300)]
    assert len(faces) == 1
    assert (faces[0].bbox.x1, faces[0].bbox.y1) == (0.0, 0.0)
    assert (faces[0].bbox.x2, faces[0].bbox.y2) == (200.0, 200.0)


def test_rescue_never_invents_a_face() -> None:
    class Blind(PickyDetector):
        def _detect_once(self, image: np.ndarray) -> list[DetectedFace]:
            self.calls.append(image.shape[:2])
            return []

    detector = Blind(FaceDetectorConfig())
    assert detector.detect(np.zeros((62, 62, 3), dtype=np.uint8)) == []
    assert len(detector.calls) == 3


def test_rescue_can_be_disabled() -> None:
    detector = PickyDetector(FaceDetectorConfig(rescue_enabled=False))
    assert detector.detect(np.zeros((62, 62, 3), dtype=np.uint8)) == []
    assert detector.calls == [(62, 62)]


def test_large_images_skip_the_enlarging_pass() -> None:
    detector = PickyDetector(FaceDetectorConfig(rescue_pad_fraction=0.0), needs_side=10_000)
    assert detector.detect(np.zeros((400, 400, 3), dtype=np.uint8)) == []
    assert detector.calls == [(400, 400)]


@pytest.mark.parametrize("shape", [(62, 100), (100, 62)])
def test_upscale_targets_the_shorter_side(shape: tuple[int, int]) -> None:
    image = np.zeros((*shape, 3), dtype=np.uint8)
    enlarged, factor = FaceDetector.upscale_for_detection(image, 160)
    assert min(enlarged.shape[:2]) == 160
    assert factor == pytest.approx(160 / 62)
