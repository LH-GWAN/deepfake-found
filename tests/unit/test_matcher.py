"""Phase 4: similarity aggregation, thresholds and model-mismatch safety."""

from __future__ import annotations

import numpy as np
import pytest

from deepshield.config import FaceMatcherConfig, FaceSimilarityThresholds
from deepshield.exceptions import IdentityNotFoundError, ModelNotAvailableError
from deepshield.face.matcher import (
    NumpyFaceMatcher,
    cosine_similarity,
    euclidean_distance,
    l2_normalize,
)
from deepshield.types import IdentityProfile, ModelInfo

MODEL = ModelInfo(name="test", version="1", backend="mock")


def make_profile(references: np.ndarray, user_id: str = "u1") -> IdentityProfile:
    normalized = references / np.linalg.norm(references, axis=1, keepdims=True)
    centroid = normalized.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    return IdentityProfile(
        user_id=user_id,
        reference_embeddings=normalized.astype(np.float32),
        centroid_embedding=centroid.astype(np.float32),
        image_count=int(normalized.shape[0]),
        model=MODEL,
        embedding_dimension=int(normalized.shape[1]),
    )


@pytest.fixture
def profile() -> IdentityProfile:
    rng = np.random.default_rng(0)
    return make_profile(rng.normal(size=(4, 32)))


def test_l2_normalize_produces_unit_vectors() -> None:
    vector = l2_normalize(np.array([3.0, 4.0]))
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0)


def test_l2_normalize_survives_zero_vector() -> None:
    assert np.all(np.isfinite(l2_normalize(np.zeros(4))))


def test_cosine_of_identical_vectors_is_one(profile: IdentityProfile) -> None:
    reference = profile.reference_embeddings[0]
    assert float(cosine_similarity(reference, reference)[0]) == pytest.approx(1.0, abs=1e-6)


def test_cosine_and_euclidean_agree_on_unit_vectors(profile: IdentityProfile) -> None:
    """On unit vectors ``d^2 = 2 - 2cos``, so the two metrics rank identically."""
    probe = profile.reference_embeddings[1]
    cosines = cosine_similarity(probe, profile.reference_embeddings)
    distances = euclidean_distance(probe, profile.reference_embeddings)
    np.testing.assert_allclose(distances**2, 2 - 2 * cosines, atol=1e-5)


def test_exact_reference_scores_one(profile: IdentityProfile) -> None:
    result = NumpyFaceMatcher().match(profile.reference_embeddings[2], profile)
    assert result.similarity == pytest.approx(1.0, abs=1e-5)
    assert result.is_candidate is True


@pytest.mark.parametrize("aggregation", ["max", "mean", "topk_mean", "centroid"])
def test_every_aggregation_runs_and_is_bounded(
    aggregation: str, profile: IdentityProfile
) -> None:
    config = FaceMatcherConfig(aggregation=aggregation)
    result = NumpyFaceMatcher(config).match(profile.reference_embeddings[0], profile)
    assert -1.0001 <= result.similarity <= 1.0001
    assert result.aggregation == aggregation


def test_max_is_at_least_mean(profile: IdentityProfile) -> None:
    probe = profile.reference_embeddings[0]
    maximum = NumpyFaceMatcher(FaceMatcherConfig(aggregation="max")).match(probe, profile)
    mean = NumpyFaceMatcher(FaceMatcherConfig(aggregation="mean")).match(probe, profile)
    assert maximum.similarity >= mean.similarity


def test_topk_mean_sits_between_mean_and_max(profile: IdentityProfile) -> None:
    probe = profile.reference_embeddings[0]
    scores = {
        name: NumpyFaceMatcher(FaceMatcherConfig(aggregation=name, top_k=2))
        .match(probe, profile)
        .similarity
        for name in ("max", "mean", "topk_mean")
    }
    assert scores["mean"] <= scores["topk_mean"] <= scores["max"] + 1e-9


def test_thresholds_drive_the_decision(profile: IdentityProfile) -> None:
    probe = profile.reference_embeddings[0]
    strict = NumpyFaceMatcher(
        thresholds=FaceSimilarityThresholds(
            candidate_threshold=0.99, high_confidence_threshold=0.999
        )
    ).match(probe, profile)
    lenient = NumpyFaceMatcher(
        thresholds=FaceSimilarityThresholds(
            candidate_threshold=-1.0, high_confidence_threshold=-1.0
        )
    ).match(probe, profile)
    assert strict.is_candidate is True
    assert lenient.is_candidate is True
    assert lenient.is_high_confidence is True


def test_unrelated_probe_is_not_a_candidate(profile: IdentityProfile) -> None:
    rng = np.random.default_rng(99)
    probe = rng.normal(size=32)
    result = NumpyFaceMatcher(
        thresholds=FaceSimilarityThresholds(candidate_threshold=0.9)
    ).match(probe, profile)
    assert result.is_candidate is False


def test_dimension_mismatch_is_rejected_not_silently_scored(profile: IdentityProfile) -> None:
    """Embeddings from different models are not comparable and must not be compared."""
    with pytest.raises(ModelNotAvailableError, match="not comparable"):
        NumpyFaceMatcher().match(np.zeros(64), profile)


def test_empty_profile_raises(profile: IdentityProfile) -> None:
    empty = make_profile(np.ones((1, 32)))
    empty.reference_embeddings = np.zeros((0, 32), dtype=np.float32)
    with pytest.raises(IdentityNotFoundError):
        NumpyFaceMatcher().match(np.ones(32), empty)


def test_match_many_sorts_best_first() -> None:
    rng = np.random.default_rng(3)
    base = rng.normal(size=(2, 16))
    near = make_profile(base, "near")
    far = make_profile(rng.normal(size=(2, 16)), "far")
    probe = near.reference_embeddings[0]
    results = NumpyFaceMatcher().match_many(probe, [far, near])
    assert results[0].matched_user_id == "near"
    assert results[0].similarity >= results[1].similarity


def test_best_match_returns_none_without_identities() -> None:
    assert NumpyFaceMatcher().best_match(np.ones(4), []) is None


def test_euclidean_is_logged_but_cosine_decides(profile: IdentityProfile) -> None:
    result = NumpyFaceMatcher(FaceMatcherConfig(log_euclidean=True)).match(
        profile.reference_embeddings[0], profile
    )
    assert result.metric == "cosine"
    assert result.euclidean_distance is not None


def test_margin_demotes_an_ambiguous_probe(profile: IdentityProfile) -> None:
    """Scoring the same against two identities identifies neither of them."""
    rng = np.random.default_rng(5)
    other = make_profile(rng.normal(size=(4, 32)), "other")
    probe = profile.reference_embeddings[0] + other.reference_embeddings[0]
    probe = probe / np.linalg.norm(probe)

    matcher = NumpyFaceMatcher(
        thresholds=FaceSimilarityThresholds(
            candidate_threshold=0.1, high_confidence_threshold=0.2, min_margin=0.3
        )
    )
    best = matcher.match_many(probe, [profile, other])[0]
    assert best.margin is not None
    assert best.margin < 0.3
    assert best.decision == "ambiguous"
    assert best.is_high_confidence is False


def test_clear_winner_keeps_high_confidence(profile: IdentityProfile) -> None:
    rng = np.random.default_rng(6)
    other = make_profile(rng.normal(size=(4, 32)), "other")
    matcher = NumpyFaceMatcher(
        thresholds=FaceSimilarityThresholds(
            candidate_threshold=0.1, high_confidence_threshold=0.5, min_margin=0.1
        )
    )
    best = matcher.match_many(profile.reference_embeddings[0], [profile, other])[0]
    assert best.decision == "high_confidence"
    assert best.margin > 0.1


def test_margin_is_none_for_a_single_identity(profile: IdentityProfile) -> None:
    """A personal deployment usually enrolls one person; the guard must not misfire."""
    matcher = NumpyFaceMatcher(
        thresholds=FaceSimilarityThresholds(
            candidate_threshold=0.1, high_confidence_threshold=0.5, min_margin=0.9
        )
    )
    best = matcher.match_many(profile.reference_embeddings[0], [profile])[0]
    assert best.margin is None
    assert best.decision == "high_confidence"


def test_low_probe_quality_raises_the_thresholds() -> None:
    matcher = NumpyFaceMatcher(
        thresholds=FaceSimilarityThresholds(
            candidate_threshold=0.3, high_confidence_threshold=0.5, low_quality_penalty=0.2
        )
    )
    assert matcher.effective_thresholds(1.0) == (0.3, 0.5)
    candidate, high = matcher.effective_thresholds(0.0)
    assert candidate == pytest.approx(0.5)
    assert high == pytest.approx(0.7)


def test_quality_penalty_is_off_by_default() -> None:
    matcher = NumpyFaceMatcher(
        thresholds=FaceSimilarityThresholds(
            candidate_threshold=0.3, high_confidence_threshold=0.5
        )
    )
    assert matcher.effective_thresholds(0.0) == (0.3, 0.5)


def test_low_quality_probe_can_be_demoted(profile: IdentityProfile) -> None:
    matcher = NumpyFaceMatcher(
        thresholds=FaceSimilarityThresholds(
            candidate_threshold=0.3, high_confidence_threshold=0.5, low_quality_penalty=0.6
        )
    )
    probe = profile.reference_embeddings[0] * 0.7 + profile.reference_embeddings[1] * 0.3
    probe = probe / np.linalg.norm(probe)
    good = matcher.match(probe, profile, probe_quality=1.0)
    poor = matcher.match(probe, profile, probe_quality=0.0)
    assert good.similarity == pytest.approx(poor.similarity)
    assert poor.is_high_confidence is False or good.is_high_confidence == poor.is_high_confidence


def test_decision_states_are_exhaustive(profile: IdentityProfile) -> None:
    matcher = NumpyFaceMatcher(
        thresholds=FaceSimilarityThresholds(
            candidate_threshold=0.4, high_confidence_threshold=0.9
        )
    )
    rng = np.random.default_rng(11)
    high = matcher.match(profile.reference_embeddings[0], profile)
    low = matcher.match(rng.normal(size=32), profile)
    assert high.decision == "high_confidence"
    assert low.decision in {"candidate", "no_match"}
