"""Deepfake detection adapter interface and the Phase 0 mock backend.

Plain language: estimate how likely it is that a picture or clip was generated
or edited by an AI model.

Formally, a detector maps media to a scalar in ``[0, 1]``. That number is called
``synthetic_probability`` throughout the codebase, never ``is_fake``. Published
detectors reach high accuracy on the generator families they were trained on and
degrade sharply on unseen ones, so the score is treated as one signal among
several rather than a verdict.

The important part of this module is the boundary, not the model. Detectors are
plugged in behind a stable interface, and their name, version and training
dataset travel with every score so results stay reproducible when models change.

Planned signal families: spatial artefacts, frequency artefacts, blending
artefacts, temporal inconsistency and optical-flow inconsistency.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np

from deepshield.config import DeepfakeDetectorConfig
from deepshield.exceptions import InvalidMediaError
from deepshield.registry import ComponentRegistry
from deepshield.types import DeepfakeResult, ModelInfo


class DeepfakeDetector(ABC):
    """Contract every synthetic-media detector must satisfy."""

    name: str = "abstract"

    @property
    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Identity, version and training dataset of the detector."""

    @abstractmethod
    def predict_image(self, image: np.ndarray) -> DeepfakeResult:
        """Score a single image or face crop.

        Args:
            image: ``H x W x 3`` uint8 RGB array.

        Returns:
            A :class:`~deepshield.types.DeepfakeResult` whose ``score`` is a
            synthetic-media probability, not a verdict.

        """

    @abstractmethod
    def predict_video(self, frames: list[np.ndarray]) -> DeepfakeResult:
        """Score a sequence of sampled frames and aggregate to one score."""

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


DEEPFAKE_REGISTRY: ComponentRegistry[DeepfakeDetector] = ComponentRegistry("deepfake detector")


class MockDeepfakeDetector(DeepfakeDetector):
    """Deterministic stand-in that derives a stable pseudo-score from pixels.

    It carries no forensic meaning whatsoever. Its only jobs are to keep the
    analysis pipeline runnable before a pretrained detector is attached in
    Phase 7 and to prove that the adapter boundary holds. Every result it
    produces is tagged with a note saying so.
    """

    name = "mock"

    def __init__(self, config: DeepfakeDetectorConfig | None = None) -> None:
        """Store detector configuration."""
        self.config = config or DeepfakeDetectorConfig()

    @property
    def model_info(self) -> ModelInfo:
        """Return the mock detector metadata."""
        return ModelInfo(
            name=self.config.model_name,
            version=self.config.model_version,
            backend=self.name,
            training_dataset=None,
            input_size=self.config.input_size,
        )

    def _pseudo_score(self, image: np.ndarray) -> float:
        """Map image contents to a stable value in ``[0, 1]``."""
        thumbnail = image[::16, ::16, :].astype(np.uint8)
        digest = hashlib.sha256(thumbnail.tobytes()).digest()
        return int.from_bytes(digest[:4], "big") / float(2**32)

    def predict_image(self, image: np.ndarray) -> DeepfakeResult:
        """Return a deterministic pseudo-score for one image."""
        self.validate_image(image)
        return DeepfakeResult(
            score=self._pseudo_score(image),
            model=self.model_info,
            notes=["mock backend: score is deterministic noise with no forensic meaning"],
        )

    def predict_video(self, frames: list[np.ndarray]) -> DeepfakeResult:
        """Return the mean of per-frame pseudo-scores."""
        if not frames:
            raise InvalidMediaError("no frames supplied to predict_video")
        scores = [self.predict_image(frame).score for frame in frames]
        return DeepfakeResult(
            score=float(np.mean(scores)),
            model=self.model_info,
            per_frame_scores=scores,
            notes=["mock backend: score is deterministic noise with no forensic meaning"],
        )


DEEPFAKE_REGISTRY.register("mock", MockDeepfakeDetector)


def build_deepfake_detector(config: DeepfakeDetectorConfig) -> DeepfakeDetector:
    """Instantiate the deepfake detector backend named in ``config``."""
    return DEEPFAKE_REGISTRY.create(config.backend, config)
