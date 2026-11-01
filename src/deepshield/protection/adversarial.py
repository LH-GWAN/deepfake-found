"""Adversarial identity cloaking: research layer, disabled by default.

Plain language: nudge the pixels of a photo, too little for a person to notice,
so that automated face recognition extracts a less reliable signature from it.

Formally, given an embedder ``f`` and image ``x``, search for a perturbation
``delta`` that maximises the embedding displacement ``1 - cos(f(x + delta), f(x))``
subject to a perceptual budget ``||delta||_inf <= epsilon``.

Why the search is gradient-free here. The production embedders in this project
are ONNX inference sessions with no exposed gradients, and a PyTorch-only
implementation would restrict cloaking to models this project happens to ship.
SPSA - simultaneous perturbation stochastic approximation - estimates a descent
direction from two forward evaluations of a random sign perturbation, so it
works against any object satisfying the ``FaceEmbedder`` contract. It needs far
more forward passes than backpropagation would, which is acceptable for an
offline, one-image-at-a-time protection step.

Expectation Over Transformation is applied during the search: each evaluation
sees a randomly JPEG-compressed and rescaled version of the candidate image, so
the perturbation is pushed toward a region that survives ordinary processing
rather than a fragile point solution.

Three numbers must be reported separately and never conflated:

white-box protection
    effectiveness against the exact model that was attacked
cross-model transferability
    effectiveness against a different face model
post-transformation robustness
    effectiveness after JPEG, resize, crop or a screenshot

White-box success is the easy case and says almost nothing about a real
platform. This module makes no protection guarantee; it produces measurements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from deepshield.config import AdversarialConfig
from deepshield.exceptions import ConfigurationError
from deepshield.face.embedder import FaceEmbedder
from deepshield.logging_utils import get_logger
from deepshield.media import validate_rgb
from deepshield.transforms import Transformation

logger = get_logger(__name__)

SPSA_PERTURBATION = 1.0
EOT_TRANSFORMS: tuple[tuple[str, dict[str, float]], ...] = (
    ("identity", {}),
    ("jpeg_compression", {"quality": 85}),
    ("jpeg_compression", {"quality": 70}),
    ("resize", {"scale": 0.75}),
    ("blur", {"sigma": 0.6}),
)


class AdversarialProtector(ABC):
    """Contract for producing a cloaked version of a face image."""

    @abstractmethod
    def protect(
        self,
        image: np.ndarray,
        models: list[FaceEmbedder],
        epsilon: float,
        transformations: list[Any] | None = None,
    ) -> np.ndarray:
        """Return a perceptually similar image whose embedding is displaced."""


def embedding_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the cosine distance between two embeddings, in ``[0, 2]``."""
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(1.0 - np.dot(a, b) / denominator)


class SpsaAdversarialProtector(AdversarialProtector):
    """Gradient-free cloaking usable against any :class:`FaceEmbedder`.

    Args:
        config: Perturbation budget, step count and step size.
        use_eot: Apply Expectation Over Transformation during the search.
        rng: Seeded generator; cloaking is stochastic and must be reproducible.

    """

    name = "spsa"

    def __init__(
        self,
        config: AdversarialConfig | None = None,
        use_eot: bool = True,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Store the search configuration."""
        self.config = config or AdversarialConfig()
        self.use_eot = use_eot
        self.rng = rng or np.random.default_rng(0)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Return a randomly transformed view of a candidate image."""
        if not self.use_eot:
            return image
        kind, params = EOT_TRANSFORMS[int(self.rng.integers(len(EOT_TRANSFORMS)))]
        return Transformation("eot", kind, dict(params)).apply(
            image, seed=int(self.rng.integers(1 << 30))
        )

    def _objective(
        self, candidate: np.ndarray, models: list[FaceEmbedder], reference: list[np.ndarray]
    ) -> float:
        """Return the mean embedding displacement across the model ensemble.

        Averaging over an ensemble is what discourages the search from
        overfitting to one architecture's idiosyncrasies, which is the single
        biggest reason white-box cloaking fails to transfer.
        """
        view = self._augment(candidate)
        total = 0.0
        for model, baseline in zip(models, reference, strict=True):
            try:
                current = model.embed(view).vector
            except Exception:
                return 0.0
            total += embedding_distance(current, baseline)
        return total / len(models)

    def protect(
        self,
        image: np.ndarray,
        models: list[FaceEmbedder],
        epsilon: float | None = None,
        transformations: list[Any] | None = None,
    ) -> np.ndarray:
        """Search for a bounded perturbation that displaces the face embedding.

        Args:
            image: Aligned face crop, or any image the embedders accept.
            models: One or more embedders to optimise against.
            epsilon: L-infinity budget in ``[0, 1]`` units of pixel range.
            transformations: Reserved for a caller-supplied EOT set.

        Returns:
            A uint8 image within ``epsilon * 255`` of the input at every pixel.

        Raises:
            ConfigurationError: If no model is supplied.

        """
        if not models:
            raise ConfigurationError("adversarial protection needs at least one embedder")

        original = validate_rgb(image).astype(np.float64)
        budget = float(self.config.epsilon if epsilon is None else epsilon) * 255.0
        step_size = float(self.config.step_size) * 255.0

        reference = [model.embed(validate_rgb(image)).vector for model in models]
        best = original.copy()
        best_score = 0.0
        current = original.copy()

        for iteration in range(int(self.config.steps)):
            direction = self.rng.choice([-1.0, 1.0], size=original.shape)
            plus = np.clip(current + SPSA_PERTURBATION * direction, 0, 255)
            minus = np.clip(current - SPSA_PERTURBATION * direction, 0, 255)

            score_plus = self._objective(plus.astype(np.uint8), models, reference)
            score_minus = self._objective(minus.astype(np.uint8), models, reference)

            gradient = (score_plus - score_minus) / (2.0 * SPSA_PERTURBATION) * direction
            current = current + step_size * np.sign(gradient)
            current = np.clip(current, original - budget, original + budget)
            current = np.clip(current, 0, 255)

            score = self._objective(current.astype(np.uint8), models, reference)
            if score > best_score:
                best_score, best = score, current.copy()
            if iteration % 10 == 0:
                logger.debug("spsa step %d displacement %.4f", iteration, score)

        logger.info(
            "cloaking finished: displacement %.4f under an L-inf budget of %.1f/255",
            best_score,
            budget,
        )
        return best.astype(np.uint8)

    def evaluate(
        self,
        original: np.ndarray,
        cloaked: np.ndarray,
        attacked: FaceEmbedder,
        transfer_models: list[FaceEmbedder] | None = None,
    ) -> dict[str, Any]:
        """Measure white-box, transfer and post-transformation effectiveness.

        Reporting only the white-box number is the standard way cloaking results
        are overstated, so all three are computed together.
        """
        base = attacked.embed(validate_rgb(original)).vector
        white_box = embedding_distance(attacked.embed(validate_rgb(cloaked)).vector, base)

        transfer: dict[str, float] = {}
        for model in transfer_models or []:
            other_base = model.embed(validate_rgb(original)).vector
            transfer[model.model_info.name] = embedding_distance(
                model.embed(validate_rgb(cloaked)).vector, other_base
            )

        post: dict[str, float] = {}
        for label, kind, params in (
            ("jpeg70", "jpeg_compression", {"quality": 70}),
            ("resize50", "resize", {"scale": 0.5}),
            ("screenshot", "screenshot_simulation", {}),
        ):
            transformed = Transformation(label, kind, params).apply(cloaked, seed=1)
            post[label] = embedding_distance(attacked.embed(transformed).vector, base)

        return {
            "white_box_displacement": round(white_box, 6),
            "cross_model_displacement": {k: round(v, 6) for k, v in transfer.items()},
            "post_transformation_displacement": {k: round(v, 6) for k, v in post.items()},
            "caveat": (
                "displacement against the attacked model is the easy case; only the "
                "cross-model and post-transformation numbers say anything about a real platform"
            ),
        }
