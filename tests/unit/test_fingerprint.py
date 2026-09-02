"""Phase 8: exact, perceptual and semantic fingerprints."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deepshield.config import FingerprintConfig
from deepshield.media import save_image
from deepshield.protection.fingerprint import (
    DefaultFingerprinter,
    average_hash,
    dct2,
    difference_hash,
    hamming_distance,
    hash_similarity,
    idct2,
    perceptual_hash,
)


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8)


def test_dct_round_trip_is_lossless() -> None:
    rng = np.random.default_rng(0)
    plane = rng.normal(size=(8, 8))
    np.testing.assert_allclose(idct2(dct2(plane)), plane, atol=1e-10)


def test_hashes_are_deterministic(photo: np.ndarray) -> None:
    assert perceptual_hash(photo) == perceptual_hash(photo.copy())
    assert difference_hash(photo) == difference_hash(photo.copy())
    assert average_hash(photo) == average_hash(photo.copy())


def test_hash_length_follows_hash_size(photo: np.ndarray) -> None:
    assert len(perceptual_hash(photo, hash_size=8)) == 16
    assert len(perceptual_hash(photo, hash_size=16)) == 64


def test_phash_survives_jpeg(photo: np.ndarray) -> None:
    similarity = hash_similarity(perceptual_hash(photo), perceptual_hash(_jpeg(photo, 50)))
    assert similarity > 0.9


def test_phash_is_weaker_on_low_texture_images() -> None:
    """A documented limitation: pHash needs texture to be stable.

    On a smooth gradient the DCT coefficients past DC are tiny, so thresholding
    them against their median is close to a coin flip and JPEG noise flips bits.
    """
    rng = np.random.default_rng(11)
    xx, yy = np.meshgrid(np.linspace(0, 255, 256), np.linspace(0, 255, 256))
    stacked = np.stack([xx, yy, (xx + yy) / 2], axis=-1)
    flat = np.clip(stacked + rng.normal(0, 10, stacked.shape), 0, 255).astype(np.uint8)
    assert hash_similarity(perceptual_hash(flat), perceptual_hash(_jpeg(flat, 50))) < 0.9


def test_phash_survives_resize(photo: np.ndarray) -> None:
    small = np.asarray(
        Image.fromarray(photo).resize((128, 128), Image.Resampling.LANCZOS), dtype=np.uint8
    )
    assert hash_similarity(perceptual_hash(photo), perceptual_hash(small)) > 0.9


def test_unrelated_images_have_low_similarity(
    photo: np.ndarray, other_photo: np.ndarray
) -> None:
    assert hash_similarity(perceptual_hash(photo), perceptual_hash(other_photo)) < 0.8


def test_hamming_distance_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="lengths differ"):
        hamming_distance("ff", "ffff")


def test_hamming_distance_rejects_non_hex() -> None:
    with pytest.raises(ValueError, match="hex"):
        hamming_distance("zz", "ff")


def test_hamming_distance_counts_bits() -> None:
    assert hamming_distance("00", "ff") == 8
    assert hamming_distance("0f", "0f") == 0


def test_fingerprint_image_populates_all_families(photo: np.ndarray) -> None:
    fingerprint = DefaultFingerprinter().fingerprint_image(photo, "asset-1")
    assert len(fingerprint.sha256) == 64
    assert fingerprint.phash and fingerprint.dhash
    assert fingerprint.semantic_embedding is None


def test_fingerprint_file_hashes_exact_bytes(tmp_path: Path, photo: np.ndarray) -> None:
    path = save_image(photo, tmp_path / "a.png")
    fingerprint = DefaultFingerprinter().fingerprint_file(path, "asset-1")
    from deepshield.media import sha256_file

    assert fingerprint.sha256 == sha256_file(path)


def test_compare_detects_exact_match(photo: np.ndarray) -> None:
    fingerprinter = DefaultFingerprinter()
    left = fingerprinter.fingerprint_image(photo, "a")
    right = fingerprinter.fingerprint_image(photo.copy(), "b")
    result = fingerprinter.compare(left, right)
    assert result["sha256_match"] == 1.0
    assert result["phash_similarity"] == 1.0


def test_compare_shows_recompression_is_not_exact(photo: np.ndarray) -> None:
    fingerprinter = DefaultFingerprinter(FingerprintConfig(hash_size=8))
    left = fingerprinter.fingerprint_image(photo, "a")
    right = fingerprinter.fingerprint_image(_jpeg(photo, 70), "b")
    result = fingerprinter.compare(left, right)
    assert result["sha256_match"] == 0.0
    assert result["phash_similarity"] > 0.9


def test_semantic_embedding_is_optional_and_pluggable(photo: np.ndarray) -> None:
    class StubEmbedder:
        def embed(self, image: np.ndarray) -> np.ndarray:
            return np.ones(8, dtype=np.float32)

    fingerprinter = DefaultFingerprinter(
        FingerprintConfig(semantic_embedding=True), semantic_embedder=StubEmbedder()
    )
    left = fingerprinter.fingerprint_image(photo, "a")
    right = fingerprinter.fingerprint_image(photo, "b")
    assert left.semantic_embedding is not None
    assert fingerprinter.compare(left, right)["semantic_similarity"] == pytest.approx(1.0)
