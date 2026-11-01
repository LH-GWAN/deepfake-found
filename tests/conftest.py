"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepshield.config import DeepShieldConfig, default_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project_root() -> Path:
    """Return the repository root, used to load the shipped configuration files."""
    return PROJECT_ROOT


@pytest.fixture
def config() -> DeepShieldConfig:
    """Return a configuration built from model defaults, independent of any YAML file."""
    return default_config()


@pytest.fixture
def rgb_image() -> np.ndarray:
    """Return a deterministic 256x256 RGB test image."""
    rng = np.random.default_rng(1234)
    return rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)


@pytest.fixture
def other_rgb_image() -> np.ndarray:
    """Return a second, different deterministic RGB test image."""
    rng = np.random.default_rng(4321)
    return rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)


def synthetic_photo(seed: int = 5, size: int = 256) -> np.ndarray:
    """Build a deterministic textured image that behaves like a real photograph.

    Perceptual hashes and watermarks both need texture to work with; a flat
    gradient makes their low-frequency coefficients nearly degenerate. Each seed
    produces different spatial frequencies, phases and block structure so that
    two seeds are genuinely different content rather than the same picture with
    different noise.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    fx, fy, fd = rng.uniform(4, 20, size=3)
    px, py = rng.uniform(0, 6.28, size=2)
    block = int(rng.integers(16, 64))
    plane = (
        120.0
        + 55.0 * np.sin(xx / fx + px)
        + 45.0 * np.cos(yy / fy + py)
        + 35.0 * np.sin((xx + yy) / fd)
        + 30.0 * ((xx // block + yy // block) % 2)
    )
    stacked = np.stack([plane, plane * 0.9 + 20.0, plane * 0.8 + 35.0], axis=-1)
    return np.clip(stacked + rng.normal(0, 6, stacked.shape), 0, 255).astype(np.uint8)


@pytest.fixture
def photo() -> np.ndarray:
    """Return a deterministic textured 256x256 image standing in for a photo."""
    return synthetic_photo(seed=5)


@pytest.fixture
def other_photo() -> np.ndarray:
    """Return a second textured image with genuinely different structure."""
    return synthetic_photo(seed=77)


@pytest.fixture
def large_photo() -> np.ndarray:
    """Return a 512x512 textured image, large enough to carry a watermark."""
    return synthetic_photo(seed=5, size=512)


@pytest.fixture
def face_crop() -> np.ndarray:
    """Return a deterministic 112x112 aligned face crop stand-in."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(112, 112, 3), dtype=np.uint8)
