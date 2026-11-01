"""Face alignment interface and a baseline crop-and-resize implementation.

Plain language: cut the face out of the picture and put it in a standard pose
and size so that two photos of the same person line up.

Formally, alignment applies a similarity transform estimated from facial
landmarks so that eyes, nose and mouth land on canonical coordinates in a fixed
``S x S`` crop. Embedding models are trained on such crops, and feeding them
unaligned faces measurably lowers similarity between two images of the same
person.

The baseline here only crops and resizes. Landmark-driven warping arrives with
the real detector in Phase 1; until then this is documented as a weaker
approximation rather than presented as full alignment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from deepshield.config import FaceAlignerConfig
from deepshield.registry import ComponentRegistry
from deepshield.types import AlignedFace, DetectedFace


class FaceAligner(ABC):
    """Contract every alignment backend must satisfy."""

    name: str = "abstract"

    @abstractmethod
    def align(self, image: np.ndarray, face: DetectedFace) -> AlignedFace:
        """Warp one detected face into the canonical crop.

        Args:
            image: Full ``H x W x 3`` uint8 RGB frame.
            face: The face to align.

        Returns:
            An :class:`~deepshield.types.AlignedFace` holding a square crop.

        """


ALIGNER_REGISTRY: ComponentRegistry[FaceAligner] = ComponentRegistry("face aligner")


class SimpleCropAligner(FaceAligner):
    """Crop the bounding box with margin and resize to a square.

    This ignores landmarks, so it does not correct in-plane rotation. It is a
    deliberate baseline: it makes the pipeline runnable and gives Phase 1 a
    reference point for measuring how much landmark alignment actually helps.
    """

    name = "simple_crop"

    def __init__(
        self, config: FaceAlignerConfig | None = None, margin: float | None = None
    ) -> None:
        """Store alignment configuration and the relative crop margin."""
        self.config = config or FaceAlignerConfig()
        self.margin = self.config.margin if margin is None else margin

    def align(self, image: np.ndarray, face: DetectedFace) -> AlignedFace:
        """Return a square crop of the face, expanded by ``margin`` on each side."""
        height, width = image.shape[:2]
        bbox = face.bbox
        pad_x, pad_y = bbox.width * self.margin, bbox.height * self.margin

        x1 = int(max(0, round(bbox.x1 - pad_x)))
        y1 = int(max(0, round(bbox.y1 - pad_y)))
        x2 = int(min(width, round(bbox.x2 + pad_x)))
        y2 = int(min(height, round(bbox.y2 + pad_y)))

        if x2 <= x1 or y2 <= y1:
            crop = np.zeros((1, 1, 3), dtype=np.uint8)
        else:
            crop = image[y1:y2, x1:x2]

        size = self.config.output_size
        resized = np.asarray(
            Image.fromarray(crop.astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        return AlignedFace(image=resized, source=face, output_size=size)


ALIGNER_REGISTRY.register("mock", SimpleCropAligner)
ALIGNER_REGISTRY.register("simple_crop", SimpleCropAligner)


def build_aligner(config: FaceAlignerConfig) -> FaceAligner:
    """Instantiate the aligner backend named in ``config``."""
    return ALIGNER_REGISTRY.create(config.backend, config)
