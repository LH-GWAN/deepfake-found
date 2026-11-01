"""Face embedding interface and the Phase 0 mock backend.

Plain language: turn a face picture into a list of numbers so that two pictures
of the same person produce two similar lists.

Formally, an embedder is a function ``f: R^(S x S x 3) -> R^d`` trained with a
margin loss (ArcFace and its relatives) so that cosine similarity between
embeddings of the same identity is high and between different identities is low.
DeepShield L2-normalises every embedding, which makes cosine similarity a plain
dot product and keeps distances comparable across images.

This is the single most important component of the system: identity evidence is
built entirely on it, and candidate gating decides which content is worth the
expensive deepfake detector.

Known failure modes: extreme pose, heavy occlusion, very low resolution, strong
compression, large age gaps, and demographic bias inherited from the training
set. Those are exactly what the Phase 13 robustness benchmark measures.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

import numpy as np

from deepshield.config import FaceEmbedderConfig
from deepshield.exceptions import InvalidMediaError, ModelNotAvailableError
from deepshield.registry import ComponentRegistry
from deepshield.types import FaceEmbedding, ModelInfo


class FaceEmbedder(ABC):
    """Contract every face embedding backend must satisfy."""

    name: str = "abstract"

    @property
    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Identity and version of the underlying model, stored with every result."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality ``d`` of the produced embeddings."""

    @abstractmethod
    def embed(self, face_image: np.ndarray) -> FaceEmbedding:
        """Embed one aligned face crop.

        Args:
            face_image: ``S x S x 3`` uint8 RGB aligned crop.

        Returns:
            An L2-normalised :class:`~deepshield.types.FaceEmbedding`.

        Raises:
            InvalidMediaError: If the crop is not a usable RGB image.

        """

    def embed_batch(self, face_images: list[np.ndarray]) -> list[FaceEmbedding]:
        """Embed several crops. Backends with true batching should override this."""
        return [self.embed(image) for image in face_images]

    @staticmethod
    def validate_face_image(face_image: np.ndarray) -> np.ndarray:
        """Check that a crop is a usable RGB array and return it unchanged."""
        if not isinstance(face_image, np.ndarray):
            raise InvalidMediaError("expected a numpy array face crop")
        if face_image.ndim != 3 or face_image.shape[2] != 3:
            raise InvalidMediaError(
                f"expected an S x S x 3 RGB face crop, got shape {face_image.shape}"
            )
        if face_image.size == 0:
            raise InvalidMediaError("face crop is empty")
        return face_image

    @staticmethod
    def l2_normalize(vector: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
        """Return ``vector`` scaled to unit L2 norm."""
        norm = float(np.linalg.norm(vector))
        return vector / max(norm, epsilon)


EMBEDDER_REGISTRY: ComponentRegistry[FaceEmbedder] = ComponentRegistry("face embedder")


class MockFaceEmbedder(FaceEmbedder):
    """Deterministic pixel-hash embedder used before real weights are wired in.

    It derives a pseudo-random unit vector from a downsampled hash of the crop.
    Identical crops map to identical vectors and small pixel changes map to
    unrelated vectors, so it is useful for wiring and contract tests and useless
    for recognition. It must never be used to make an identity claim.
    """

    name = "mock"

    def __init__(self, config: FaceEmbedderConfig | None = None) -> None:
        """Store embedder configuration."""
        self.config = config or FaceEmbedderConfig()

    @property
    def model_info(self) -> ModelInfo:
        """Return the mock model metadata."""
        return ModelInfo(
            name=self.config.model_name,
            version=self.config.model_version,
            backend=self.name,
            training_dataset=None,
            input_size=None,
        )

    @property
    def dimension(self) -> int:
        """Return the configured embedding dimension."""
        return self.config.embedding_dimension

    def embed(self, face_image: np.ndarray) -> FaceEmbedding:
        """Return a deterministic unit vector derived from the crop contents."""
        self.validate_face_image(face_image)
        thumbnail = face_image[::8, ::8, :].astype(np.uint8)
        seed_bytes = hashlib.sha256(thumbnail.tobytes()).digest()[:8]
        seed = int.from_bytes(seed_bytes, "big") % (2**32)
        rng = np.random.default_rng(seed)
        vector = rng.normal(size=self.dimension).astype(np.float32)
        if self.config.normalize:
            vector = self.l2_normalize(vector)
        return FaceEmbedding(
            vector=vector,
            model=self.model_info,
            normalized=self.config.normalize,
        )


EMBEDDER_REGISTRY.register("mock", MockFaceEmbedder)


def build_embedder(config: FaceEmbedderConfig) -> FaceEmbedder:
    """Instantiate the configured embedder, applying ensembling and TTA wrappers.

    Composition order matters: flip augmentation is applied to each ensemble
    member rather than to the fused vector, because averaging a vector with its
    own mirror after concatenation would mix the members together.
    """
    names = list(config.ensemble) or [config.backend]
    members = []
    for name in names:
        member = EMBEDDER_REGISTRY.create(name, config.model_copy(update={"backend": name}))
        members.append(FlipTtaEmbedder(member) if config.flip_tta else member)
    return members[0] if len(members) == 1 else EnsembleEmbedder(members)


class FlipTtaEmbedder(FaceEmbedder):
    """Averages the embedding of a crop with the embedding of its mirror image.

    Test-time augmentation, and the cheapest accuracy gain available here. A face
    embedder is not perfectly symmetric: lighting from one side, a slight yaw, or
    an asymmetric crop all shift the vector. Embedding the horizontal flip too
    and averaging cancels part of that nuisance variation while leaving identity
    intact, because a mirrored face is still the same person.

    The cost is exactly one extra forward pass. It cannot fix a bad crop or a bad
    detection; it only reduces the noise around an already-correct one.
    """

    name = "flip_tta"

    def __init__(self, inner: FaceEmbedder) -> None:
        """Wrap another embedder."""
        self.inner = inner

    @property
    def model_info(self) -> ModelInfo:
        """Return the wrapped model's metadata, marked as flip-augmented."""
        info = self.inner.model_info
        return ModelInfo(
            name=f"{info.name}+flip_tta",
            version=info.version,
            backend=info.backend,
            training_dataset=info.training_dataset,
            input_size=info.input_size,
        )

    @property
    def dimension(self) -> int:
        """Same dimensionality as the wrapped embedder."""
        return self.inner.dimension

    def embed(self, face_image: np.ndarray) -> FaceEmbedding:
        """Return the L2-normalised mean of the crop and its mirror."""
        array = self.validate_face_image(face_image)
        original = self.inner.embed(array).vector
        mirrored = self.inner.embed(np.ascontiguousarray(array[:, ::-1, :])).vector
        averaged = self.l2_normalize(
            (np.asarray(original, np.float32) + np.asarray(mirrored, np.float32)) / 2.0
        )
        return FaceEmbedding(vector=averaged, model=self.model_info, normalized=True)


class EnsembleEmbedder(FaceEmbedder):
    """Concatenates several L2-normalised embeddings into one fused vector.

    Each member is normalised and scaled by ``1 / sqrt(n)`` before concatenation,
    which makes the cosine similarity of two fused vectors exactly the mean of
    the members' individual cosine similarities. Fusion therefore happens at the
    score level, with no learned weights and nothing to overfit, while the rest
    of the system keeps treating the result as an ordinary embedding.

    Different architectures make different mistakes, so a pair that one model
    scores as a borderline match rarely fools both. Averaging pulls those
    borderline impostors down more than it pulls genuine pairs down, which is
    where the precision gain comes from. It costs one forward pass per member.
    """

    name = "ensemble"

    def __init__(self, members: list[FaceEmbedder]) -> None:
        """Wrap two or more embedders.

        Raises:
            ModelNotAvailableError: If fewer than two members are supplied.

        """
        if len(members) < 2:
            raise ModelNotAvailableError("an ensemble embedder needs at least two members")
        self.members = members
        self._scale = 1.0 / math.sqrt(len(members))

    @property
    def model_info(self) -> ModelInfo:
        """Return metadata naming every member, so fused vectors stay traceable."""
        names = "+".join(member.model_info.name for member in self.members)
        return ModelInfo(
            name=f"ensemble({names})",
            version="1",
            backend=self.name,
            training_dataset=None,
            input_size=None,
        )

    @property
    def dimension(self) -> int:
        """Sum of the member dimensionalities."""
        return sum(member.dimension for member in self.members)

    def embed(self, face_image: np.ndarray) -> FaceEmbedding:
        """Return the scaled concatenation of every member's embedding."""
        array = self.validate_face_image(face_image)
        parts = [
            self.l2_normalize(np.asarray(member.embed(array).vector, np.float32)) * self._scale
            for member in self.members
        ]
        return FaceEmbedding(
            vector=np.concatenate(parts).astype(np.float32),
            model=self.model_info,
            normalized=True,
        )
