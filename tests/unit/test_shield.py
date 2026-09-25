"""The swap shield: alignment, the ONNX executor and the perturbation itself."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepshield.config import ShieldConfig
from deepshield.face.backends import ARCFACE_TEMPLATE_112
from deepshield.types import BoundingBox, DetectedFace

torch = pytest.importorskip("torch")

from deepshield.protection.shield import (  # noqa: E402
    FRAME,
    OnnxGraph,
    SwapShield,
    similarity_matrix,
)

ARCFACE = Path("models/insightface/models/buffalo_l/w600k_r50.onnx")


def test_the_template_maps_onto_itself() -> None:
    matrix = similarity_matrix(ARCFACE_TEMPLATE_112)
    assert np.allclose(matrix, [[1, 0, 0], [0, 1, 0]], atol=1e-9)


def test_a_moved_face_is_mapped_back_exactly() -> None:
    angle, scale, shift = 0.3, 2.5, np.array([40.0, -12.0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    moved = (ARCFACE_TEMPLATE_112 @ rotation.T) * scale + shift
    matrix = similarity_matrix(moved)
    back = moved @ matrix[:, :2].T + matrix[:, 2]
    assert np.allclose(back, ARCFACE_TEMPLATE_112, atol=1e-6)


@pytest.mark.skipif(not ARCFACE.is_file(), reason="buffalo_l weights not downloaded")
def test_the_torch_execution_matches_onnxruntime() -> None:
    onnxruntime = pytest.importorskip("onnxruntime")
    session = onnxruntime.InferenceSession(str(ARCFACE), providers=["CPUExecutionProvider"])
    sample = np.random.default_rng(0).uniform(-1, 1, (2, 3, FRAME, FRAME)).astype(np.float32)
    reference = session.run(None, {session.get_inputs()[0].name: sample})[0]
    with torch.no_grad():
        ours = OnnxGraph(ARCFACE)(torch.from_numpy(sample)).numpy()
    assert np.abs(reference - ours).max() < 1e-3


class TinyEncoder:
    """A fixed random convolutional embedder standing in for ArcFace."""

    def __init__(self) -> None:
        """Draw the fixed weights."""
        generator = torch.Generator().manual_seed(0)
        self.weight = torch.randn(8, 3, 4, 4, generator=generator)
        self.projection = torch.randn(64, 8 * 28 * 28, generator=generator)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        features = torch.nn.functional.conv2d(frames, self.weight, stride=4)
        return features.flatten(1) @ self.projection.T


def face_on(image: np.ndarray) -> DetectedFace:
    height, width = image.shape[:2]
    points = ARCFACE_TEMPLATE_112 * (min(height, width) / FRAME)
    return DetectedFace(
        bbox=BoundingBox(0, 0, width, height), detection_confidence=0.99, landmarks=points
    )


def shield(steps: int = 40) -> SwapShield:
    config = ShieldConfig(steps=steps, eot_samples=2)
    return SwapShield(config, Path("."), encoder=TinyEncoder(), device="cpu")


def test_the_shield_moves_the_face_embedding_within_its_budget() -> None:
    # Low contrast, so the stand-in encoder's (near-linear) response is within reach of the
    # budget; ArcFace itself moves from 1.0 to about -0.3 on real photos (README).
    image = np.random.default_rng(1).integers(118, 139, (160, 160, 3), dtype=np.uint8)
    protected, report = shield().shield(image, [face_on(image)])
    assert report["applied"] is True
    assert report["similarity_to_clean_face"][0] < 0.5
    change = np.abs(protected.astype(int) - image.astype(int)).max()
    assert change <= round(ShieldConfig().epsilon * 255) + 1


def test_the_shield_is_reproducible() -> None:
    image = np.random.default_rng(2).integers(0, 256, (120, 120, 3), dtype=np.uint8)
    first, _ = shield(steps=5).shield(image, [face_on(image)])
    second, _ = shield(steps=5).shield(image, [face_on(image)])
    assert np.array_equal(first, second)


def test_a_photo_without_landmarks_is_left_alone() -> None:
    image = np.full((100, 100, 3), 128, dtype=np.uint8)
    faceless = DetectedFace(bbox=BoundingBox(0, 0, 50, 50), detection_confidence=0.9)
    protected, report = shield().shield(image, [faceless])
    assert report["applied"] is False
    assert "no face" in report["reason"]
    assert np.array_equal(protected, image)
