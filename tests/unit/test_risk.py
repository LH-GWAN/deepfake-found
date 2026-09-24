"""Phase 10: the verdict engine and the calibration metrics behind it."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from deepshield.config import Thresholds
from deepshield.exceptions import ConfigurationError
from deepshield.risk.calibration import ThresholdCalibrator, pair_scores, roc_curve
from deepshield.risk.scorer import VerdictRiskScorer
from deepshield.types import RiskAssessment, RiskEvidence, RiskLevel, Verdict


def calibrated_detector() -> Thresholds:
    base = Thresholds()
    return base.model_copy(
        update={"deepfake": base.deepfake.model_copy(update={"calibrated": True})}
    )


def evidence(**overrides: Any) -> RiskEvidence:
    fields: dict[str, Any] = {
        "subject_user_id": "u1",
        "identities_compared": True,
        "faces_detected": 1,
    }
    fields.update(overrides)
    return RiskEvidence(**fields)


def own_asset(**overrides: Any) -> RiskEvidence:
    return evidence(
        asset_id="a1",
        asset_owner="u1",
        asset_match_basis="watermark code",
        distribution_id="instagram",
        **overrides,
    )


def assess(item: RiskEvidence, thresholds: Thresholds | None = None) -> RiskAssessment:
    return VerdictRiskScorer(thresholds).assess(item)


def test_own_photo_reposted_is_a_copy_not_a_threat() -> None:
    """The regression that motivated verdicts: the user's own photo once scored CRITICAL."""
    result = assess(own_asset(identity_decision="high_confidence", identity_similarity=0.83))
    assert result.verdict is Verdict.OWN_COPY
    assert result.risk_level is RiskLevel.LOW
    assert any("instagram" in line for line in result.explanation)


def test_a_face_swap_of_the_user_outranks_their_own_repost() -> None:
    repost = assess(own_asset(identity_decision="high_confidence", identity_similarity=0.83))
    swap = assess(evidence(identity_decision="high_confidence", identity_similarity=0.66))
    assert swap.verdict is Verdict.IDENTITY_MATCH
    order = list(RiskLevel)
    assert order.index(swap.risk_level) > order.index(repost.risk_level)


def test_an_identity_match_admits_it_cannot_tell_real_from_synthetic() -> None:
    result = assess(evidence(identity_decision="high_confidence", deepfake_score=0.97))
    assert result.verdict is Verdict.IDENTITY_MATCH
    assert result.risk_level is RiskLevel.MEDIUM
    assert any("cannot be determined" in line for line in result.explanation)
    assert any("did not affect the verdict" in line for line in result.explanation)
    assert any("uncalibrated" in line for line in result.limitations)


@pytest.mark.parametrize(
    ("score", "level"), [(0.6, RiskLevel.HIGH), (0.9, RiskLevel.CRITICAL)]
)
def test_a_calibrated_detector_raises_an_identity_match(score: float, level: RiskLevel) -> None:
    result = assess(
        evidence(
            identity_decision="high_confidence", deepfake_score=score, deepfake_calibrated=True
        ),
        calibrated_detector(),
    )
    assert result.verdict is Verdict.SYNTHETIC_SUSPECTED
    assert result.risk_level is level


def test_a_calibrated_detector_below_threshold_leaves_the_match() -> None:
    result = assess(
        evidence(
            identity_decision="high_confidence", deepfake_score=0.1, deepfake_calibrated=True
        ),
        calibrated_detector(),
    )
    assert result.verdict is Verdict.IDENTITY_MATCH


def test_own_photo_with_another_face_is_an_alteration() -> None:
    """The target direction: someone else's face placed on the user's protected photo."""
    result = assess(own_asset(identity_decision="no_match", owner_face_in_original=True))
    assert result.verdict is Verdict.OWN_ALTERED
    assert result.risk_level is RiskLevel.HIGH


def test_own_photo_that_never_showed_the_user_is_a_copy() -> None:
    result = assess(own_asset(identity_decision="no_match", owner_face_in_original=False))
    assert result.verdict is Verdict.OWN_COPY


def test_alteration_needs_the_original_to_be_checkable() -> None:
    result = assess(own_asset(identity_decision="no_match", owner_face_in_original=None))
    assert result.verdict is Verdict.OWN_UNVERIFIED
    assert result.risk_level is RiskLevel.MEDIUM
    assert any("missing or unreadable" in line for line in result.limitations)


@pytest.mark.parametrize("decision", ["candidate", "ambiguous"])
def test_a_weak_resemblance_is_not_called_an_alteration(decision: str) -> None:
    result = assess(own_asset(identity_decision=decision, owner_face_in_original=True))
    assert result.verdict is Verdict.OWN_UNVERIFIED


def test_a_copy_with_no_detectable_face_is_not_called_an_alteration() -> None:
    result = assess(
        own_asset(faces_detected=0, identity_decision=None, owner_face_in_original=True)
    )
    assert result.verdict is Verdict.OWN_UNVERIFIED
    assert any("cropped out" in line for line in result.explanation)


@pytest.mark.parametrize(("pixels", "scale"), [(54.0, 0.55), (79.0, 0.85), (54.0, None)])
def test_a_shrunk_face_on_the_users_copy_is_not_called_an_alteration(
    pixels: float, scale: float | None
) -> None:
    """A shrunk, recompressed repost can miss its owner without anyone swapping the face."""
    result = assess(
        own_asset(
            identity_decision="no_match",
            owner_face_in_original=True,
            probe_quality=0.33,
            probe_face_pixels=pixels,
            copy_scale=scale,
        )
    )
    assert result.verdict is Verdict.OWN_UNVERIFIED
    assert result.risk_level is RiskLevel.MEDIUM
    assert any("too small" in line for line in result.explanation)


def test_a_small_face_at_the_registered_size_is_still_an_alteration() -> None:
    """The face was this small in the original, where it matched; now it does not."""
    result = assess(
        own_asset(
            identity_decision="no_match",
            owner_face_in_original=True,
            probe_face_pixels=70.0,
            copy_scale=1.0,
        )
    )
    assert result.verdict is Verdict.OWN_ALTERED


@pytest.mark.parametrize("quality", [0.9, 0.3])
def test_a_full_size_face_that_does_not_match_is_still_an_alteration(quality: float) -> None:
    """A swapped face is softer than the photograph around it; softness excuses nothing."""
    result = assess(
        own_asset(
            identity_decision="no_match",
            owner_face_in_original=True,
            probe_quality=quality,
            probe_face_pixels=96.0,
        )
    )
    assert result.verdict is Verdict.OWN_ALTERED


def test_a_byte_identical_copy_is_a_copy_whatever_the_faces() -> None:
    result = assess(
        evidence(
            asset_id="a1",
            asset_owner="u1",
            asset_match_basis="exact file hash",
            identity_decision="no_match",
        )
    )
    assert result.verdict is Verdict.OWN_COPY


def test_an_unenrolled_owner_cannot_be_checked_for_face_changes() -> None:
    result = assess(own_asset(identities_compared=False, identity_decision=None))
    assert result.verdict is Verdict.OWN_UNVERIFIED
    assert any("Enroll the asset owner" in line for line in result.limitations)


def test_a_regenerated_face_on_the_users_photo_is_an_alteration() -> None:
    result = assess(
        own_asset(
            identity_decision="high_confidence", deepfake_score=0.9, deepfake_calibrated=True
        ),
        calibrated_detector(),
    )
    assert result.verdict is Verdict.OWN_ALTERED


def test_another_users_asset_is_judged_by_identity_and_named() -> None:
    result = assess(
        evidence(
            asset_id="a9",
            asset_owner="u2",
            asset_match_basis="watermark code",
            identity_decision="high_confidence",
        )
    )
    assert result.verdict is Verdict.IDENTITY_MATCH
    assert any("belongs to 'u2'" in line for line in result.explanation)


@pytest.mark.parametrize("decision", ["candidate", "ambiguous"])
def test_borderline_identity_is_a_review(decision: str) -> None:
    result = assess(evidence(identity_decision=decision, identity_similarity=0.38))
    assert result.verdict is Verdict.REVIEW
    assert result.risk_level is RiskLevel.LOW


def test_no_resemblance_is_unrelated() -> None:
    assert assess(evidence(identity_decision="no_match")).verdict is Verdict.UNRELATED


def test_a_miss_on_a_small_face_does_not_rule_the_user_out() -> None:
    """Heavy compression drops a small genuine face below the review threshold."""
    result = assess(
        evidence(
            identity_decision="no_match",
            identity_similarity=0.30,
            probe_quality=0.33,
            probe_face_pixels=54.0,
        )
    )
    assert result.verdict is Verdict.UNRELATED
    assert not any(line.startswith("No face resembles you") for line in result.explanation)
    assert any("rule you out" in line for line in result.explanation)
    assert any("80 pixels" in line for line in result.limitations)


def test_a_miss_on_a_full_size_face_is_stated_plainly() -> None:
    result = assess(
        evidence(
            identity_decision="no_match",
            identity_similarity=0.05,
            probe_quality=0.3,
            probe_face_pixels=140.0,
        )
    )
    assert any(line.startswith("No face resembles you") for line in result.explanation)
    assert not any("80 pixels" in line for line in result.limitations)


def test_no_face_is_unrelated_and_says_so() -> None:
    result = assess(evidence(faces_detected=0))
    assert result.verdict is Verdict.UNRELATED
    assert any("No face was detected" in line for line in result.explanation)


def test_nothing_to_compare_is_inconclusive_not_unrelated() -> None:
    result = assess(RiskEvidence())
    assert result.verdict is Verdict.INCONCLUSIVE


def test_uncalibrated_detector_never_changes_a_verdict() -> None:
    for score in (0.0, 0.5, 1.0):
        result = assess(evidence(identity_decision="high_confidence", deepfake_score=score))
        assert result.verdict is Verdict.IDENTITY_MATCH


def test_base_limitations_are_always_present() -> None:
    result = assess(RiskEvidence())
    assert any("training data" in line for line in result.limitations)


def test_assessment_serialises_verdict_and_signals() -> None:
    payload = assess(own_asset(identity_decision="high_confidence")).to_dict()
    assert payload["verdict"] == "own_copy"
    assert payload["risk_level"] == "LOW"
    assert payload["subject_user_id"] == "u1"
    assert payload["signals"]["asset_match_basis"] == "watermark code"
    assert "risk_score" not in payload


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
