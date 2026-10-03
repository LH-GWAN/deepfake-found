"""A face degraded with its picture is told from a face pasted into it."""

from __future__ import annotations

import numpy as np

from deepshield.pipeline.analysis_pipeline import FACE_CHANGE_RATIO, face_change_ratio
from deepshield.types import BoundingBox, DetectedFace

FACE = DetectedFace(bbox=BoundingBox(96, 96, 160, 160), detection_confidence=0.9)


def picture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    return np.kron(coarse, np.ones((8, 8, 1), dtype=np.uint8))


def test_noise_over_the_whole_picture_is_not_a_replaced_face() -> None:
    original = picture(1)
    noisy = np.clip(
        original.astype(int) + np.random.default_rng(2).normal(0, 12, original.shape), 0, 255
    ).astype(np.uint8)
    ratio = face_change_ratio(original, noisy, FACE, (1.0, 1.0))
    assert ratio is not None and ratio < FACE_CHANGE_RATIO


def test_a_face_pasted_over_the_original_is() -> None:
    original = picture(1)
    swapped = original.copy()
    swapped[96:160, 96:160] = picture(3)[96:160, 96:160]
    ratio = face_change_ratio(original, swapped, FACE, (1.0, 1.0))
    assert ratio is not None and ratio > FACE_CHANGE_RATIO


def test_the_face_box_is_mapped_onto_the_registered_size() -> None:
    original = picture(1)
    swapped = original.copy()
    swapped[96:160, 96:160] = picture(3)[96:160, 96:160]
    half = DetectedFace(bbox=BoundingBox(48, 48, 80, 80), detection_confidence=0.9)
    assert face_change_ratio(original, swapped, half, (2.0, 2.0)) > FACE_CHANGE_RATIO
