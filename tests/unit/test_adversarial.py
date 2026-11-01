"""Phase 12: bounded perturbation, honest evaluation, reproducibility."""

from __future__ import annotations

import numpy as np
import pytest

from deepshield.config import AdversarialConfig, FaceEmbedderConfig
from deepshield.exceptions import ConfigurationError
from deepshield.face.embedder import MockFaceEmbedder
from deepshield.protection.adversarial import (
    SpsaAdversarialProtector,
    embedding_distance,
)
from deepshield.quality import psnr


@pytest.fixture
def embedder() -> MockFaceEmbedder:
    return MockFaceEmbedder(FaceEmbedderConfig(embedding_dimension=64))


@pytest.fixture
def protector() -> SpsaAdversarialProtector:
    return SpsaAdversarialProtector(
        AdversarialConfig(enabled=True, epsilon=0.03, steps=4, step_size=0.01),
        use_eot=False,
        rng=np.random.default_rng(0),
    )


def test_embedding_distance_bounds() -> None:
    vector = np.array([1.0, 0.0])
    assert embedding_distance(vector, vector) == pytest.approx(0.0)
    assert embedding_distance(vector, -vector) == pytest.approx(2.0)
    assert embedding_distance(np.zeros(2), vector) == 0.0


def test_perturbation_respects_the_epsilon_budget(
    protector: SpsaAdversarialProtector, embedder: MockFaceEmbedder, face_crop: np.ndarray
) -> None:
    cloaked = protector.protect(face_crop, [embedder], epsilon=0.03)
    difference = np.abs(cloaked.astype(int) - face_crop.astype(int))
    assert difference.max() <= 0.03 * 255 + 1


def test_output_shape_and_dtype_are_preserved(
    protector: SpsaAdversarialProtector, embedder: MockFaceEmbedder, face_crop: np.ndarray
) -> None:
    cloaked = protector.protect(face_crop, [embedder])
    assert cloaked.shape == face_crop.shape
    assert cloaked.dtype == np.uint8


def test_tighter_budget_costs_less_quality(
    embedder: MockFaceEmbedder, face_crop: np.ndarray
) -> None:
    def run(epsilon: float) -> float:
        protector = SpsaAdversarialProtector(
            AdversarialConfig(enabled=True, epsilon=epsilon, steps=4, step_size=0.01),
            use_eot=False,
            rng=np.random.default_rng(1),
        )
        return psnr(face_crop, protector.protect(face_crop, [embedder]))

    assert run(0.01) > run(0.08)


def test_search_is_reproducible_given_a_seed(
    embedder: MockFaceEmbedder, face_crop: np.ndarray
) -> None:
    def run() -> np.ndarray:
        return SpsaAdversarialProtector(
            AdversarialConfig(enabled=True, epsilon=0.03, steps=4, step_size=0.01),
            use_eot=False,
            rng=np.random.default_rng(7),
        ).protect(face_crop, [embedder])

    np.testing.assert_array_equal(run(), run())


def test_no_models_is_rejected(
    protector: SpsaAdversarialProtector, face_crop: np.ndarray
) -> None:
    with pytest.raises(ConfigurationError, match="at least one embedder"):
        protector.protect(face_crop, [])


def test_eot_path_runs(embedder: MockFaceEmbedder, face_crop: np.ndarray) -> None:
    protector = SpsaAdversarialProtector(
        AdversarialConfig(enabled=True, epsilon=0.03, steps=3, step_size=0.01),
        use_eot=True,
        rng=np.random.default_rng(2),
    )
    assert protector.protect(face_crop, [embedder]).shape == face_crop.shape


def test_evaluation_separates_the_three_measurements(
    protector: SpsaAdversarialProtector, embedder: MockFaceEmbedder, face_crop: np.ndarray
) -> None:
    """White-box success alone says nothing about a real platform, so all three are reported."""
    other = MockFaceEmbedder(FaceEmbedderConfig(embedding_dimension=32, model_name="other"))
    cloaked = protector.protect(face_crop, [embedder])
    report = protector.evaluate(face_crop, cloaked, embedder, transfer_models=[other])

    assert "white_box_displacement" in report
    assert "cross_model_displacement" in report
    assert "post_transformation_displacement" in report
    assert set(report["post_transformation_displacement"]) == {
        "jpeg70",
        "resize50",
        "screenshot",
    }
    assert "caveat" in report


def test_ensemble_optimisation_accepts_several_models(
    protector: SpsaAdversarialProtector, embedder: MockFaceEmbedder, face_crop: np.ndarray
) -> None:
    second = MockFaceEmbedder(FaceEmbedderConfig(embedding_dimension=32, model_name="second"))
    assert protector.protect(face_crop, [embedder, second]).shape == face_crop.shape
