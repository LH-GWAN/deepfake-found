"""Perceptual quality metrics shared by every protection experiment.

Protection is a trade-off: a watermark or an adversarial perturbation is only
useful if the user is still willing to publish the result. PSNR and SSIM put a
number on that cost so the trade-off can be reported rather than asserted.

PSNR measures pixel-wise error in decibels and is easy to read but poorly
correlated with what people notice. SSIM compares local luminance, contrast and
structure and tracks perception better. Both are reported; neither is treated as
the last word on visible quality.
"""

from __future__ import annotations

import numpy as np

from deepshield.exceptions import InvalidMediaError
from deepshield.media import to_grayscale, validate_rgb

MAX_PIXEL_VALUE = 255.0
SSIM_C1 = (0.01 * MAX_PIXEL_VALUE) ** 2
SSIM_C2 = (0.03 * MAX_PIXEL_VALUE) ** 2
SSIM_WINDOW = 7


def _match_shapes(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate both images and confirm they share a shape."""
    left = validate_rgb(reference, "reference")
    right = validate_rgb(candidate, "candidate")
    if left.shape != right.shape:
        raise InvalidMediaError(
            f"images must have the same shape, got {left.shape} and {right.shape}"
        )
    return left, right


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Return peak signal-to-noise ratio in dB, or ``inf`` for identical images."""
    left, right = _match_shapes(reference, candidate)
    mse = float(np.mean((left.astype(np.float64) - right.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((MAX_PIXEL_VALUE**2) / mse))


def _uniform_filter(plane: np.ndarray, size: int) -> np.ndarray:
    """Return a box-filtered copy of ``plane`` using a cumulative-sum sliding window."""
    pad = size // 2
    padded = np.pad(plane, pad, mode="reflect")
    cumulative = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cumulative = np.pad(cumulative, ((1, 0), (1, 0)), mode="constant")
    height, width = plane.shape
    top, left = np.arange(height), np.arange(width)
    bottom, right = top + size, left + size
    total = (
        cumulative[np.ix_(bottom, right)]
        - cumulative[np.ix_(top, right)]
        - cumulative[np.ix_(bottom, left)]
        + cumulative[np.ix_(top, left)]
    )
    return total / float(size * size)


def ssim(reference: np.ndarray, candidate: np.ndarray, window: int = SSIM_WINDOW) -> float:
    """Return mean structural similarity over the luminance plane, in ``[-1, 1]``.

    Uses a uniform window rather than a Gaussian one, which keeps the metric
    dependency-free at a small cost in fidelity to the original formulation.
    """
    left, right = _match_shapes(reference, candidate)
    x = to_grayscale(left).astype(np.float64)
    y = to_grayscale(right).astype(np.float64)

    size = min(window, x.shape[0], x.shape[1])
    if size % 2 == 0:
        size -= 1
    if size < 3:
        return 1.0 if np.array_equal(x, y) else 0.0

    mu_x = _uniform_filter(x, size)
    mu_y = _uniform_filter(y, size)
    sigma_x = _uniform_filter(x * x, size) - mu_x * mu_x
    sigma_y = _uniform_filter(y * y, size) - mu_y * mu_y
    sigma_xy = _uniform_filter(x * y, size) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + SSIM_C1) * (2 * sigma_xy + SSIM_C2)
    denominator = (mu_x**2 + mu_y**2 + SSIM_C1) * (sigma_x + sigma_y + SSIM_C2)
    return float(np.mean(numerator / denominator))


def image_quality(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Return both quality metrics for one reference/candidate pair."""
    return {"psnr": psnr(reference, candidate), "ssim": ssim(reference, candidate)}


LAPLACIAN_KERNEL = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])


def laplacian_variance(image: np.ndarray) -> float:
    """Return the variance of the Laplacian, a standard sharpness proxy.

    A sharp image has strong second derivatives at edges, so their variance is
    high; blurring suppresses them and the variance collapses. It is only a
    proxy: a noisy image also scores high, and a genuinely flat scene scores low
    without being blurred. It is used to rank enrollment photos, never to reject
    an analysis input.
    """
    plane = to_grayscale(validate_rgb(image)).astype(np.float64)
    if plane.shape[0] < 3 or plane.shape[1] < 3:
        return 0.0
    response = np.zeros_like(plane)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            weight = LAPLACIAN_KERNEL[dy + 1, dx + 1]
            if weight != 0.0:
                response += weight * np.roll(np.roll(plane, dy, axis=0), dx, axis=1)
    return float(response[1:-1, 1:-1].var())


QUALITY_REFERENCE_PIXELS = 112.0
QUALITY_REFERENCE_SHARPNESS = 120.0


def face_quality_score(
    face_pixels: float,
    crop: np.ndarray,
    reference_pixels: float = QUALITY_REFERENCE_PIXELS,
    reference_sharpness: float = QUALITY_REFERENCE_SHARPNESS,
) -> float:
    """Score how much a probe face crop can be trusted, in ``[0, 1]``.

    Two independent ways a probe goes wrong are combined by taking the worse of
    them, because either one alone ruins the embedding:

    resolution
        A face upsampled from thirty pixels carries almost no identity signal,
        however sharp the interpolation looks.
    sharpness
        A blurred or heavily compressed crop loses exactly the fine texture the
        embedder relies on.

    This is a reliability estimate, not a rejection: a low score raises the
    similarity a probe must reach before it can make a claim, rather than
    discarding evidence outright.

    The reference values are measured, not guessed. ``112`` pixels is the
    embedder's own input size, below which the crop is genuinely upsampled. The
    sharpness reference is the median Laplacian variance of clean aligned crops
    in the evaluation set; a Gaussian blur of sigma 3 drives the same measurement
    down by a factor of about thirty-five, which is exactly the separation this
    score exists to expose.
    """
    size_score = min(1.0, max(0.0, float(face_pixels) / max(reference_pixels, 1e-9)))
    sharpness = laplacian_variance(crop)
    sharpness_score = min(1.0, max(0.0, sharpness / max(reference_sharpness, 1e-9)))
    return float(min(size_score, sharpness_score))
