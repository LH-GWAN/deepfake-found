"""Phase 10: risk features, scoring policy and calibration metrics."""

from __future__ import annotations

import numpy as np
import pytest

from deepshield.config import (
    FaceSimilarityThresholds,
    RiskThresholds,
    Thresholds,
)
from deepshield.exceptions import ConfigurationError
from deepshield.risk.calibration import ThresholdCalibrator, pair_scores, roc_curve
from deepshield.risk.features import DefaultRiskFeatureBuilder
from deepshield.risk.scorer import WeightedRiskScorer
from deepshield.types import RiskLevel


def calibrated_thresholds() -> Thresholds:
    base = Thresholds()
    return base.model_copy(
        update={
            "face_similarity": FaceSimilarityThresholds(
                calibrated=True, calibration_source="test"
            ),
            "deepfake": base.deepfake.model_copy(update={"calibrated": True}),
        }
    )


def test_missing_signals_stay_none_not_zero() -> None:
    """A failed detector must never be scored as evidence of safety."""
    features = DefaultRiskFeatureBuilder().build({}).features
    assert features.face_similarity is None
    assert features.deepfake_score is None


def test_absent_watermark_is_neutral_not_scored() -> None:
    builder = DefaultRiskFeatureBuilder()
    absent = builder.build({"watermark_detected": False, "watermark_confidence": 0.9})
    present = builder.build({"watermark_detected": True, "watermark_confidence": 0.9})
    assert absent.features.watermark_confidence is None
    assert present.features.watermark_confidence == 0.9


def test_negative_similarity_is_clamped_not_negative() -> None:
    features = DefaultRiskFeatureBuilder().build({"face_similarity": -0.4}).features
    assert features.face_similarity == 0.0


def test_uncalibrated_signals_are_flagged() -> None:
    feature_set = DefaultRiskFeatureBuilder().build(
        {"face_similarity": 0.9, "deepfake_score": 0.8}
    )
    assert set(feature_set.uncalibrated_names()) == {"face_similarity", "deepfake_score"}


def test_uncalibrated_signals_are_excluded_from_the_score() -> None:
    thresholds = Thresholds()
    feature_set = DefaultRiskFeatureBuilder(thresholds).build({"face_similarity": 0.99})
    assessment = WeightedRiskScorer(thresholds.risk).score(feature_set)
    assert assessment.risk_score == 0
    assert "face_similarity" in assessment.signals["excluded"]
    assert any("not calibrated" in line for line in assessment.limitations)


def test_calibrated_signals_are_scored() -> None:
    thresholds = calibrated_thresholds()
    feature_set = DefaultRiskFeatureBuilder(thresholds).build({"face_similarity": 0.99})
    assessment = WeightedRiskScorer(thresholds.risk).score(feature_set)
    assert assessment.risk_score == 99
    assert assessment.risk_level == RiskLevel.CRITICAL


def test_trust_uncalibrated_can_be_enabled_explicitly() -> None:
    thresholds = Thresholds()
    feature_set = DefaultRiskFeatureBuilder(thresholds).build({"face_similarity": 0.8})
    assessment = WeightedRiskScorer(thresholds.risk, trust_uncalibrated=True).score(feature_set)
    assert assessment.risk_score == 80


def test_weights_are_renormalised_over_present_signals() -> None:
    """A signal that could not be computed lowers coverage, not the score."""
    thresholds = calibrated_thresholds()
    builder = DefaultRiskFeatureBuilder(thresholds)
    scorer = WeightedRiskScorer(thresholds.risk)
    only_face = scorer.score(builder.build({"face_similarity": 1.0}))
    assert only_face.risk_score == 100
    assert only_face.signals["coverage"] < 1.0


def test_coverage_shortfall_is_reported() -> None:
    thresholds = calibrated_thresholds()
    assessment = WeightedRiskScorer(thresholds.risk).score(
        DefaultRiskFeatureBuilder(thresholds).build({"face_similarity": 0.5})
    )
    assert any("signal weight was available" in line for line in assessment.limitations)


def test_no_signals_yields_an_undefined_score() -> None:
    assessment = WeightedRiskScorer().score(DefaultRiskFeatureBuilder().build({}))
    assert assessment.risk_score == 0
    assert assessment.risk_level == RiskLevel.LOW
    assert any("undefined" in line for line in assessment.limitations)


@pytest.mark.parametrize(
    ("score", "level"),
    [(0, RiskLevel.LOW), (39, RiskLevel.LOW), (40, RiskLevel.MEDIUM),
     (70, RiskLevel.HIGH), (85, RiskLevel.CRITICAL), (100, RiskLevel.CRITICAL)],
)
def test_level_bands(score: int, level: RiskLevel) -> None:
    assert WeightedRiskScorer(RiskThresholds()).level_for(score) == level


def test_explanation_names_every_contribution() -> None:
    thresholds = calibrated_thresholds()
    feature_set = DefaultRiskFeatureBuilder(thresholds).build(
        {
            "face_similarity": 0.9,
            "deepfake_score": 0.8,
            "watermark_detected": True,
            "watermark_confidence": 1.0,
        }
    )
    assessment = WeightedRiskScorer(thresholds.risk).score(feature_set)
    text = " ".join(assessment.explanation)
    assert "face_similarity" in text
    assert "deepfake_score" in text
    assert "watermark_confidence" in text


def test_base_limitations_are_always_present() -> None:
    assessment = WeightedRiskScorer().score(DefaultRiskFeatureBuilder().build({}))
    assert any("training data" in line for line in assessment.limitations)


def test_roc_is_perfect_on_separable_data() -> None:
    scores = np.concatenate([np.full(20, 0.9), np.full(20, 0.1)])
    labels = np.concatenate([np.ones(20), np.zeros(20)])
    curve = roc_curve(scores, labels)
    assert curve.auc == pytest.approx(1.0)
    assert curve.eer == pytest.approx(0.0)


def test_roc_is_chance_on_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=400)
    labels = np.array([1, 0] * 200)
    assert 0.4 < roc_curve(scores, labels).auc < 0.6


def test_roc_needs_both_classes() -> None:
    with pytest.raises(ConfigurationError, match="both genuine and impostor"):
        roc_curve(np.array([0.5, 0.6]), np.array([1, 1]))


def test_roc_rejects_mismatched_lengths() -> None:
    with pytest.raises(ConfigurationError, match="same length"):
        roc_curve(np.array([0.5]), np.array([1, 0]))


def test_max_margin_sits_between_the_distributions() -> None:
    scores = np.concatenate([np.full(10, 0.8), np.full(10, 0.2)])
    labels = np.concatenate([np.ones(10), np.zeros(10)])
    result = ThresholdCalibrator("max_margin").calibrate(scores, labels)
    assert result.threshold == pytest.approx(0.5)
    assert any("separable" in note for note in result.notes)


def test_max_margin_falls_back_when_classes_overlap() -> None:
    rng = np.random.default_rng(1)
    scores = np.concatenate([rng.normal(0.6, 0.2, 50), rng.normal(0.4, 0.2, 50)])
    labels = np.concatenate([np.ones(50), np.zeros(50)])
    result = ThresholdCalibrator("max_margin").calibrate(scores, labels)
    assert any("overlap" in note for note in result.notes)


def test_max_fpr_respects_its_budget() -> None:
    rng = np.random.default_rng(2)
    scores = np.concatenate([rng.normal(0.8, 0.1, 200), rng.normal(0.2, 0.1, 800)])
    labels = np.concatenate([np.ones(200), np.zeros(800)])
    result = ThresholdCalibrator("max_fpr", target_fpr=0.01).calibrate(scores, labels)
    assert result.false_positive_rate <= 0.02


def test_small_and_dependent_samples_are_flagged() -> None:
    scores = np.array([0.9, 0.85, 0.2, 0.15])
    labels = np.array([1, 1, 0, 0])
    result = ThresholdCalibrator("eer").calibrate(scores, labels, independent_pairs=2)
    assert any("provisional" in note for note in result.notes)
    assert any("independent" in note for note in result.notes)


def test_unknown_criterion_is_rejected() -> None:
    with pytest.raises(ConfigurationError, match="unknown calibration criterion"):
        ThresholdCalibrator("magic")


def test_pair_scores_labels_genuine_and_impostor() -> None:
    embeddings = {
        "a1": np.array([1.0, 0.0]),
        "a2": np.array([0.99, 0.14]),
        "b1": np.array([0.0, 1.0]),
    }
    identity = {"a1": "a", "a2": "a", "b1": "b"}
    scores, labels = pair_scores(embeddings, identity)
    assert labels.sum() == 1
    assert scores[labels == 1][0] > scores[labels == 0].max()
