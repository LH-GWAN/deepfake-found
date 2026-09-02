"""Face detection interface and the Phase 0 mock backend.

Plain language: find where the faces are in a picture and how sure we are.
Formally, a detector maps an ``H x W x 3`` uint8 image to zero or more
:class:`~deepshield.types.DetectedFace` records holding a bounding box, a
detection confidence and optionally five facial landmarks.

Detection has to come first because every later signal - alignment, embedding,
similarity, deepfake scoring - operates on a face crop rather than the full
frame. Failure modes to expect: very small faces, extreme pose, heavy motion
blur, occlusion, and images with no face at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from deepshield.config import FaceDetectorConfig
from deepshield.exceptions import InvalidMediaError
from deepshield.logging_utils import get_logger
from deepshield.registry import ComponentRegistry
from deepshield.types import BoundingBox, DetectedFace

logger = get_logger(__name__)


class FaceDetector(ABC):
    """Contract every face detection backend must satisfy."""

    name: str = "abstract"

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Locate faces in an RGB image.

        Args:
            image: ``H x W x 3`` uint8 array in RGB order.

        Returns:
            Detected faces sorted by descending confidence; empty when none pass
            the configured confidence threshold.

        Raises:
            InvalidMediaError: If the array is not a decodable RGB image.

        """

    @staticmethod
    def downscale_for_detection(
        image: np.ndarray, max_side: int
    ) -> tuple[np.ndarray, float]:
        """Shrink an oversized image before detection and report the scale used.

        Detectors are trained on a limited range of face-to-image ratios. A face
        occupying 500 pixels of a 4000-pixel-wide photo lands far outside that
        range and is missed outright, so large inputs are downscaled first and
        the resulting boxes are mapped back to original coordinates.

        Returns:
            The image to run detection on, and the factor its coordinates must
            be divided by to return to the original frame.

        """
        from PIL import Image

        height, width = image.shape[:2]
        longest = max(height, width)
        if longest <= max_side:
            return image, 1.0
        scale = max_side / float(longest)
        resized = Image.fromarray(image).resize(
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            Image.Resampling.LANCZOS,
        )
        return np.asarray(resized, dtype=np.uint8), scale

    @staticmethod
    def rescale_face(face: DetectedFace, scale: float) -> DetectedFace:
        """Map a detection made on a downscaled image back to original coordinates."""
        if scale == 1.0:
            return face
        factor = 1.0 / scale
        box = face.bbox
        return DetectedFace(
            bbox=BoundingBox(
                box.x1 * factor, box.y1 * factor, box.x2 * factor, box.y2 * factor
            ),
            detection_confidence=face.detection_confidence,
            landmarks=None if face.landmarks is None else face.landmarks * factor,
            frame_index=face.frame_index,
            timestamp_seconds=face.timestamp_seconds,
        )

    @staticmethod
    def validate_image(image: np.ndarray) -> np.ndarray:
        """Check that ``image`` is a usable RGB array and return it unchanged."""
        if not isinstance(image, np.ndarray):
            raise InvalidMediaError("expected a numpy array image")
        if image.ndim != 3 or image.shape[2] != 3:
            raise InvalidMediaError(f"expected an H x W x 3 RGB image, got shape {image.shape}")
        if image.size == 0:
            raise InvalidMediaError("image is empty")
        return image


DETECTOR_REGISTRY: ComponentRegistry[FaceDetector] = ComponentRegistry("face detector")


class MockFaceDetector(FaceDetector):
    """Deterministic stand-in that returns one centred box per image.

    It performs no real detection. It exists so that the pipeline, the CLI and
    the tests can run end to end before InsightFace weights are wired in during
    Phase 1, and so that downstream code is exercised against the real
    :class:`FaceDetector` contract.
    """

    name = "mock"

    def __init__(self, config: FaceDetectorConfig | None = None) -> None:
        """Store detector configuration."""
        self.config = config or FaceDetectorConfig()

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Return a single centred face box covering half of the image."""
        self.validate_image(image)
        height, width = image.shape[:2]
        box_w, box_h = width * 0.5, height * 0.5
        x1, y1 = (width - box_w) / 2.0, (height - box_h) / 2.0
        bbox = BoundingBox(x1, y1, x1 + box_w, y1 + box_h)

        if min(bbox.width, bbox.height) < self.config.min_face_size:
            logger.debug("mock detector: face below min_face_size, returning no faces")
            return []

        confidence = 0.99
        if confidence < self.config.detection_confidence_threshold:
            return []
        return [DetectedFace(bbox=bbox, detection_confidence=confidence)]


DETECTOR_REGISTRY.register("mock", MockFaceDetector)


def build_detector(config: FaceDetectorConfig) -> FaceDetector:
    """Instantiate the detector backend named in ``config``."""
    return DETECTOR_REGISTRY.create(config.backend, config)
