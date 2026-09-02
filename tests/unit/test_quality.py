"""PSNR and SSIM behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from deepshield.exceptions import InvalidMediaError
from deepshield.quality import image_quality, psnr, ssim


def test_identical_images_are_perfect(rgb_image: np.ndarray) -> None:
    assert psnr(rgb_image, rgb_image) == float("inf")
    assert ssim(rgb_image, rgb_image) == pytest.approx(1.0, abs=1e-9)


def test_psnr_decreases_with_distortion(rgb_image: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    small = np.clip(rgb_image + rng.normal(0, 2, rgb_image.shape), 0, 255).astype(np.uint8)
    large = np.clip(rgb_image + rng.normal(0, 20, rgb_image.shape), 0, 255).astype(np.uint8)
    assert psnr(rgb_image, small) > psnr(rgb_image, large)


def test_ssim_decreases_with_distortion(rgb_image: np.ndarray) -> None:
    rng = np.random.default_rng(1)
    small = np.clip(rgb_image + rng.normal(0, 2, rgb_image.shape), 0, 255).astype(np.uint8)
    large = np.clip(rgb_image + rng.normal(0, 40, rgb_image.shape), 0, 255).astype(np.uint8)
    assert ssim(rgb_image, small) > ssim(rgb_image, large)


def test_ssim_is_bounded(rgb_image: np.ndarray, other_rgb_image: np.ndarray) -> None:
    assert -1.0 <= ssim(rgb_image, other_rgb_image) <= 1.0


def test_shape_mismatch_is_rejected(rgb_image: np.ndarray) -> None:
    with pytest.raises(InvalidMediaError, match="same shape"):
        psnr(rgb_image, rgb_image[:100])


def test_image_quality_returns_both_metrics(rgb_image: np.ndarray) -> None:
    assert set(image_quality(rgb_image, rgb_image)) == {"psnr", "ssim"}


def test_ssim_handles_tiny_images() -> None:
    tiny = np.zeros((2, 2, 3), dtype=np.uint8)
    assert ssim(tiny, tiny) == 1.0


def test_laplacian_variance_drops_with_blur(photo: np.ndarray) -> None:
    from deepshield.quality import laplacian_variance
    from deepshield.transforms import Transformation

    blurred = Transformation("b", "blur", {"sigma": 4.0}).apply(photo, seed=0)
    assert laplacian_variance(blurred) < laplacian_variance(photo)


def test_laplacian_variance_handles_tiny_images() -> None:
    from deepshield.quality import laplacian_variance

    assert laplacian_variance(np.zeros((2, 2, 3), dtype=np.uint8)) == 0.0


def test_face_quality_is_bounded(photo: np.ndarray) -> None:
    from deepshield.quality import face_quality_score

    assert 0.0 <= face_quality_score(200, photo) <= 1.0
    assert 0.0 <= face_quality_score(5, photo) <= 1.0


def test_small_faces_score_low(photo: np.ndarray) -> None:
    from deepshield.quality import face_quality_score

    assert face_quality_score(20, photo) < face_quality_score(200, photo)


def test_blur_lowers_face_quality(photo: np.ndarray) -> None:
    from deepshield.quality import face_quality_score
    from deepshield.transforms import Transformation

    blurred = Transformation("b", "blur", {"sigma": 5.0}).apply(photo, seed=0)
    assert face_quality_score(200, blurred) < face_quality_score(200, photo)


def test_face_quality_takes_the_worse_of_the_two_factors(photo: np.ndarray) -> None:
    """Either a tiny face or a blurred one ruins the embedding on its own."""
    from deepshield.quality import face_quality_score
    from deepshield.transforms import Transformation

    blurred = Transformation("b", "blur", {"sigma": 6.0}).apply(photo, seed=0)
    assert face_quality_score(10, blurred) <= min(
        face_quality_score(10, photo), face_quality_score(200, blurred)
    )
