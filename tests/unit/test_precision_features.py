"""Precision-oriented additions: TTA, ensembling and precision-targeted thresholds."""

from __future__ import annotations

import numpy as np
import pytest

from deepshield.config import FaceEmbedderConfig
from deepshield.exceptions import ModelNotAvailableError
from deepshield.face.embedder import (
    EnsembleEmbedder,
    FlipTtaEmbedder,
    MockFaceEmbedder,
    build_embedder,
)
from deepshield.risk.calibration import precision_recall_at, threshold_for_precision


@pytest.fixture
def inner() -> MockFaceEmbedder:
    return MockFaceEmbedder(FaceEmbedderConfig(embedding_dimension=64))


def test_flip_tta_preserves_dimension_and_norm(inner, face_crop: np.ndarray) -> None:
    embedding = FlipTtaEmbedder(inner).embed(face_crop)
    assert embedding.dimension == inner.dimension
    assert float(np.linalg.norm(embedding.vector)) == pytest.approx(1.0, abs=1e-5)


def test_flip_tta_is_deterministic(inner, face_crop: np.ndarray) -> None:
    embedder = FlipTtaEmbedder(inner)
    np.testing.assert_allclose(
        embedder.embed(face_crop).vector, embedder.embed(face_crop.copy()).vector
    )


def test_flip_tta_is_mirror_invariant(inner, face_crop: np.ndarray) -> None:
    """Averaging a crop with its mirror makes the result independent of which one arrived."""
    embedder = FlipTtaEmbedder(inner)
    mirrored = np.ascontiguousarray(face_crop[:, ::-1, :])
    np.testing.assert_allclose(
        embedder.embed(face_crop).vector, embedder.embed(mirrored).vector, atol=1e-6
    )


def test_flip_tta_records_itself_in_model_metadata(inner) -> None:
    """A vector produced with augmentation must not be mistaken for a plain one."""
    assert FlipTtaEmbedder(inner).model_info.name.endswith("+flip_tta")


def test_ensemble_dimension_is_the_sum(inner) -> None:
    second = MockFaceEmbedder(FaceEmbedderConfig(embedding_dimension=32, model_name="second"))
    assert EnsembleEmbedder([inner, second]).dimension == 96


def test_ensemble_output_is_unit_norm(inner, face_crop: np.ndarray) -> None:
    second = MockFaceEmbedder(FaceEmbedderConfig(embedding_dimension=32, model_name="second"))
    vector = EnsembleEmbedder([inner, second]).embed(face_crop).vector
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


def test_ensemble_cosine_is_the_mean_of_member_cosines(face_crop: np.ndarray) -> None:
    """Scaling by 1/sqrt(n) before concatenation makes fusion a score-level average."""
    first = MockFaceEmbedder(FaceEmbedderConfig(embedding_dimension=16, model_name="a"))
    second = MockFaceEmbedder(FaceEmbedderConfig(embedding_dimension=16, model_name="b"))
    other_crop = np.roll(face_crop, 17, axis=0)

    fused = EnsembleEmbedder([first, second])
    fused_similarity = float(
        np.dot(fused.embed(face_crop).vector, fused.embed(other_crop).vector)
    )
    member_similarities = [
        float(np.dot(member.embed(face_crop).vector, member.embed(other_crop).vector))
        for member in (first, second)
    ]
    assert fused_similarity == pytest.approx(float(np.mean(member_similarities)), abs=1e-5)


def test_ensemble_names_every_member(inner) -> None:
    second = MockFaceEmbedder(FaceEmbedderConfig(embedding_dimension=32, model_name="second"))
    assert "second" in EnsembleEmbedder([inner, second]).model_info.name


def test_ensemble_rejects_a_single_member(inner) -> None:
    with pytest.raises(ModelNotAvailableError, match="at least two"):
        EnsembleEmbedder([inner])


def test_builder_applies_tta_and_ensembling_from_config(face_crop: np.ndarray) -> None:
    plain = build_embedder(FaceEmbedderConfig(backend="mock", embedding_dimension=64))
    tta = build_embedder(
        FaceEmbedderConfig(backend="mock", embedding_dimension=64, flip_tta=True)
    )
    fused = build_embedder(
        FaceEmbedderConfig(backend="mock", embedding_dimension=64, ensemble=["mock", "mock"])
    )
    assert isinstance(plain, MockFaceEmbedder)
    assert isinstance(tta, FlipTtaEmbedder)
    assert isinstance(fused, EnsembleEmbedder)
    assert fused.embed(face_crop).dimension == 128


def test_builder_applies_tta_inside_the_ensemble(face_crop: np.ndarray) -> None:
    """TTA must wrap each member, not the fused vector, or members would blend."""
    fused = build_embedder(
        FaceEmbedderConfig(
            backend="mock", embedding_dimension=64, ensemble=["mock", "mock"], flip_tta=True
        )
    )
    assert isinstance(fused, EnsembleEmbedder)
    assert all(isinstance(member, FlipTtaEmbedder) for member in fused.members)


def test_precision_recall_counts_are_consistent() -> None:
    scores = np.array([0.9, 0.8, 0.4, 0.2])
    labels = np.array([1, 0, 1, 0])
    point = precision_recall_at(scores, labels, 0.5)
    assert point["true_positives"] == 1
    assert point["false_positives"] == 1
    assert point["false_negatives"] == 1
    assert point["precision"] == pytest.approx(0.5)
    assert point["recall"] == pytest.approx(0.5)
    assert point["f1"] == pytest.approx(0.5)


def test_threshold_for_precision_meets_its_target() -> None:
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0.8, 0.15, 200), rng.normal(0.3, 0.15, 800)])
    labels = np.concatenate([np.ones(200), np.zeros(800)])
    _, point = threshold_for_precision(scores, labels, 0.99)
    assert point["precision"] >= 0.99


def test_higher_precision_targets_cost_recall() -> None:
    """The trade-off must be visible rather than hidden behind a single number."""
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(0.8, 0.15, 200), rng.normal(0.3, 0.15, 800)])
    labels = np.concatenate([np.ones(200), np.zeros(800)])
    _, lenient = threshold_for_precision(scores, labels, 0.90)
    _, strict = threshold_for_precision(scores, labels, 1.00)
    assert strict["threshold"] > lenient["threshold"]
    assert strict["recall"] <= lenient["recall"]


def test_unreachable_precision_falls_back_without_claiming_success() -> None:
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 0, 1, 0])
    threshold, point = threshold_for_precision(scores, labels, 0.99)
    assert threshold == pytest.approx(0.5)
    assert point["precision"] < 0.99
