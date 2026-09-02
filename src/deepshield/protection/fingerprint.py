"""Image fingerprinting: exact, perceptual and semantic identity.

Three fingerprints are stored per asset because they answer three different
questions:

``SHA-256``
    Is this byte-for-byte the same file? Any re-encoding breaks it.
``pHash`` / ``dHash``
    Is this a re-encoded, resized or lightly edited copy of the same picture?
    Both survive compression and scaling; neither survives heavy cropping.
``deep embedding``
    Does this depict the same scene or subject? Optional and pluggable.

None of them tracks an image through a generative model. A deepfake built from a
user's photo shares neither its bytes nor its perceptual hash, so fingerprints
support source and copy tracking, not training-data attribution.

How the perceptual hashes work:

``dHash``
    Resize to ``(size + 1) x size`` grayscale, then emit one bit per horizontal
    neighbour pair saying whether brightness increased. It encodes gradients,
    which makes it robust to global brightness shifts.
``pHash``
    Resize to ``4 * size`` square grayscale, take a 2-D DCT, keep the top-left
    ``size x size`` low-frequency block, and threshold each coefficient against
    the block median. Low frequencies survive JPEG and blur, so pHash is the
    more robust of the two.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from deepshield.config import FingerprintConfig
from deepshield.exceptions import InvalidMediaError
from deepshield.media import sha256_array, sha256_file, to_grayscale, validate_rgb
from deepshield.types import AssetFingerprint


class Fingerprinter(ABC):
    """Contract for computing the three fingerprint families of one asset."""

    @abstractmethod
    def fingerprint_image(self, image: np.ndarray, asset_id: str) -> AssetFingerprint:
        """Compute fingerprints from a decoded image."""

    @abstractmethod
    def fingerprint_file(self, path: Path, asset_id: str) -> AssetFingerprint:
        """Compute fingerprints from a file on disk, hashing the exact bytes."""


@lru_cache(maxsize=8)
def _dct_matrix(size: int) -> np.ndarray:
    """Return the orthonormal DCT-II basis matrix of a given size."""
    indices = np.arange(size)
    basis = np.cos(np.pi * (2 * indices[None, :] + 1) * indices[:, None] / (2 * size))
    basis *= np.sqrt(2.0 / size)
    basis[0] *= np.sqrt(0.5)
    return basis


def dct2(plane: np.ndarray) -> np.ndarray:
    """Return the 2-D orthonormal DCT-II of a square float plane."""
    matrix = _dct_matrix(plane.shape[0])
    transformed: np.ndarray = matrix @ plane @ matrix.T
    return transformed


def idct2(coefficients: np.ndarray) -> np.ndarray:
    """Return the inverse of :func:`dct2`."""
    matrix = _dct_matrix(coefficients.shape[0])
    restored: np.ndarray = matrix.T @ coefficients @ matrix
    return restored


def _resize_grayscale(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return a grayscale float32 resize of an RGB image."""
    gray = to_grayscale(validate_rgb(image)).astype(np.uint8)
    resized = Image.fromarray(gray).resize((width, height), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.float32)


def _bits_to_hex(bits: np.ndarray) -> str:
    """Pack a flat boolean array into a lowercase hex string."""
    padded = np.packbits(bits.astype(np.uint8).ravel())
    return padded.tobytes().hex()


def average_hash(image: np.ndarray, hash_size: int = 8) -> str:
    """Return the aHash of an image: brightness thresholded against the mean."""
    plane = _resize_grayscale(image, hash_size, hash_size)
    return _bits_to_hex(plane > plane.mean())


def difference_hash(image: np.ndarray, hash_size: int = 8) -> str:
    """Return the dHash of an image: one bit per horizontal brightness gradient."""
    plane = _resize_grayscale(image, hash_size + 1, hash_size)
    return _bits_to_hex(plane[:, 1:] > plane[:, :-1])


def perceptual_hash(image: np.ndarray, hash_size: int = 8, factor: int = 4) -> str:
    """Return the pHash of an image: low-frequency DCT coefficients vs their median."""
    side = hash_size * factor
    plane = _resize_grayscale(image, side, side)
    low_frequency = dct2(plane.astype(np.float64))[:hash_size, :hash_size]
    median = np.median(low_frequency[1:, 1:])
    return _bits_to_hex(low_frequency > median)


def hamming_distance(left: str, right: str) -> int:
    """Return the number of differing bits between two hex-encoded hashes.

    Raises:
        ValueError: If the hashes have different lengths or are not hex.

    """
    if len(left) != len(right):
        raise ValueError(f"hash lengths differ: {len(left)} vs {len(right)}")
    try:
        left_bytes = bytes.fromhex(left)
        right_bytes = bytes.fromhex(right)
    except ValueError as exc:
        raise ValueError("hashes must be hex encoded") from exc
    return sum(bin(a ^ b).count("1") for a, b in zip(left_bytes, right_bytes, strict=True))


def hash_similarity(left: str, right: str) -> float:
    """Return ``1 - normalised hamming distance`` between two hashes, in ``[0, 1]``."""
    total_bits = len(left) * 4
    if total_bits == 0:
        return 0.0
    return 1.0 - hamming_distance(left, right) / total_bits


class DefaultFingerprinter(Fingerprinter):
    """Computes SHA-256, pHash and dHash, with an optional semantic embedding."""

    def __init__(
        self,
        config: FingerprintConfig | None = None,
        semantic_embedder: object | None = None,
    ) -> None:
        """Store fingerprint configuration and an optional semantic embedder."""
        self.config = config or FingerprintConfig()
        self.semantic_embedder = semantic_embedder

    @property
    def hash_size(self) -> int:
        """Side length in bits of the perceptual hash grids."""
        return self.config.hash_size

    def _semantic(self, image: np.ndarray) -> np.ndarray | None:
        """Return a semantic embedding when one is configured and available."""
        if not self.config.semantic_embedding or self.semantic_embedder is None:
            return None
        embed = getattr(self.semantic_embedder, "embed", None)
        if embed is None:
            return None
        return np.asarray(embed(image), dtype=np.float32)

    def fingerprint_image(self, image: np.ndarray, asset_id: str) -> AssetFingerprint:
        """Compute fingerprints for a decoded image, hashing its pixel bytes."""
        array = validate_rgb(image)
        return AssetFingerprint(
            asset_id=asset_id,
            sha256=sha256_array(array),
            phash=perceptual_hash(array, self.hash_size),
            dhash=difference_hash(array, self.hash_size),
            semantic_embedding=self._semantic(array),
        )

    def fingerprint_file(self, path: Path, asset_id: str) -> AssetFingerprint:
        """Compute fingerprints for a file, hashing its exact bytes."""
        from deepshield.media import load_image

        file_path = Path(path)
        image = load_image(file_path)
        return AssetFingerprint(
            asset_id=asset_id,
            sha256=sha256_file(file_path),
            phash=perceptual_hash(image, self.hash_size),
            dhash=difference_hash(image, self.hash_size),
            semantic_embedding=self._semantic(image),
        )

    def compare(self, left: AssetFingerprint, right: AssetFingerprint) -> dict[str, float]:
        """Return per-family similarity between two fingerprints.

        ``sha256_match`` is exact identity; the perceptual entries are
        ``1 - normalised hamming distance``; ``semantic_similarity`` is cosine
        similarity when both sides carry an embedding.
        """
        if len(left.phash) != len(right.phash):
            raise InvalidMediaError("fingerprints were computed with different hash sizes")

        result = {
            "sha256_match": float(left.sha256 == right.sha256),
            "phash_similarity": hash_similarity(left.phash, right.phash),
            "dhash_similarity": hash_similarity(left.dhash, right.dhash),
            "phash_hamming": float(hamming_distance(left.phash, right.phash)),
            "dhash_hamming": float(hamming_distance(left.dhash, right.dhash)),
        }
        if left.semantic_embedding is not None and right.semantic_embedding is not None:
            a = left.semantic_embedding.astype(np.float64)
            b = right.semantic_embedding.astype(np.float64)
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            result["semantic_similarity"] = (
                0.0 if denominator == 0.0 else float(np.dot(a, b) / denominator)
            )
        return result
