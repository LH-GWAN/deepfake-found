"""Risk scoring: fusing independent signals into an explainable assessment.

The score starts as a deterministic weighted rule rather than a learned model.
A rule can be read, argued with and audited, which matters more at this stage
than accuracy, and there is no labelled data to fit anything on yet. The
interface leaves room for a calibrated classifier once that data exists.

Two design decisions carry most of the honesty of the system:

Renormalisation over present signals
    Missing signals are not scored as zero. The weights of the signals that were
    actually computed are renormalised, so a watermark that could not be checked
    lowers confidence rather than lowering risk.

Uncalibrated signals are reported but not scored
    A detector whose threshold has never been fitted produces a number with no
    defined operating point. Including it would launder a guess into a
    percentage. Such signals appear in the explanation and in the limitations,
    and are excluded from the arithmetic unless configuration says otherwise.

The output is never a bare number: every assessment carries the evidence that
moved it and the limits that qualify it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from deepshield.config import RiskThresholds
from deepshield.risk.features import RiskFeatureSet
from deepshield.types import RiskAssessment, RiskFeatures, RiskLevel

BASE_LIMITATIONS = [
    "Face similarity does not prove that an image was used as training data.",
    "Deepfake detectors may fail on generator families absent from their training data.",
    "A watermark can be removed by cropping or regeneration, so its absence proves nothing.",
    "High face similarity alone is expected for genuine photographs of the user.",
]


@dataclass(frozen=True)
class ScoredSignal:
    """One signal's contribution to the final score."""

    name: str
    value: float
    weight: float
    contribution: float
    counted: bool
    note: str


class RiskScorer(ABC):
    """Contract for turning risk features into an explainable assessment."""

    @abstractmethod
    def score(self, features: RiskFeatures | RiskFeatureSet) -> RiskAssessment:
        """Return the risk score, level, per-signal explanation and limitations."""


class WeightedRiskScorer(RiskScorer):
    """Deterministic weighted-sum scorer with renormalisation and calibration gating."""

    def __init__(
        self,
        thresholds: RiskThresholds | None = None,
        trust_uncalibrated: bool = False,
    ) -> None:
        """Store the weights, level bands and calibration policy."""
        self.thresholds = thresholds or RiskThresholds()
        self.trust_uncalibrated = trust_uncalibrated

    def level_for(self, score: int) -> RiskLevel:
        """Map a numeric score onto its qualitative band."""
        levels = self.thresholds.levels
        if score >= levels.critical:
            return RiskLevel.CRITICAL
        if score >= levels.high:
            return RiskLevel.HIGH
        if score >= levels.medium:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _weight_of(self, name: str) -> float:
        """Return the configured weight of one signal."""
        return float(getattr(self.thresholds.weights, name, 0.0))

    def score(self, features: RiskFeatures | RiskFeatureSet) -> RiskAssessment:
        """Fuse the present signals into a score with a full audit trail."""
        feature_set = (
            features
            if isinstance(features, RiskFeatureSet)
            else RiskFeatureSet(features=features)
        )
        values = feature_set.features
        calibration = {p.name: p.calibrated for p in feature_set.provenance}

        scored: list[ScoredSignal] = []
        for name in (
            "face_similarity",
            "deepfake_score",
            "watermark_confidence",
            "fingerprint_similarity",
            "provenance_confidence",
        ):
            value = getattr(values, name)
            if value is None:
                continue
            calibrated = calibration.get(name, True)
            counted = calibrated or self.trust_uncalibrated
            weight = self._weight_of(name)
            scored.append(
                ScoredSignal(
                    name=name,
                    value=float(value),
                    weight=weight,
                    contribution=0.0,
                    counted=counted,
                    note="" if counted else "excluded: threshold not calibrated",
                )
            )

        contributing = [signal for signal in scored if signal.counted and signal.weight > 0]
        total_weight = sum(signal.weight for signal in contributing)

        excluded_names = [signal.name for signal in scored if not signal.counted]

        if total_weight <= 0.0:
            return RiskAssessment(
                risk_score=0,
                risk_level=RiskLevel.LOW,
                signals={
                    "scored": [],
                    "excluded": excluded_names,
                    "available": [signal.name for signal in scored],
                    "coverage": 0.0,
                },
                explanation=[
                    "No calibrated signal was available, so no risk score was computed.",
                    *(
                        f"{signal.name}: {signal.value:.3f} reported but {signal.note}"
                        for signal in scored
                        if not signal.counted
                    ),
                ],
                limitations=BASE_LIMITATIONS
                + [
                    "Risk score is undefined because no calibrated signal was available.",
                    *(
                        [
                            "Excluded from the score because their thresholds are not "
                            "calibrated: " + ", ".join(excluded_names)
                        ]
                        if excluded_names
                        else []
                    ),
                ],
            )

        weighted = 0.0
        details: list[ScoredSignal] = []
        for signal in contributing:
            share = signal.weight / total_weight
            contribution = signal.value * share
            weighted += contribution
            details.append(
                ScoredSignal(
                    name=signal.name,
                    value=signal.value,
                    weight=share,
                    contribution=contribution,
                    counted=True,
                    note="",
                )
            )

        raw_score = int(round(max(0.0, min(1.0, weighted)) * 100))
        level = self.level_for(raw_score)
        coverage = total_weight / max(
            sum(
                self._weight_of(name)
                for name in (
                    "face_similarity",
                    "deepfake_score",
                    "watermark_confidence",
                    "fingerprint_similarity",
                    "provenance_confidence",
                )
            ),
            1e-9,
        )

        explanation = self._explain(details, scored, values)
        limitations = list(BASE_LIMITATIONS)
        excluded = excluded_names
        if excluded:
            limitations.append(
                "Excluded from the score because their thresholds are not calibrated: "
                + ", ".join(excluded)
            )
        if coverage < 0.999:
            limitations.append(
                f"Only {coverage:.0%} of the intended signal weight was available; "
                "the score is computed over the signals that were present."
            )

        return RiskAssessment(
            risk_score=raw_score,
            risk_level=level,
            signals={
                "scored": [
                    {
                        "name": s.name,
                        "value": round(s.value, 4),
                        "effective_weight": round(s.weight, 4),
                        "contribution": round(s.contribution, 4),
                    }
                    for s in details
                ],
                "excluded": excluded,
                "coverage": round(coverage, 4),
            },
            explanation=explanation,
            limitations=limitations,
        )

    def _explain(
        self,
        details: list[ScoredSignal],
        all_signals: list[ScoredSignal],
        values: RiskFeatures,
    ) -> list[str]:
        """Render one human-readable line per signal, present or absent."""
        lines: list[str] = []
        for signal in sorted(details, key=lambda s: s.contribution, reverse=True):
            lines.append(
                f"{signal.name}: {signal.value:.3f} "
                f"(weight {signal.weight:.0%}, contributes {signal.contribution * 100:.1f} points)"
            )
        for signal in all_signals:
            if not signal.counted:
                lines.append(f"{signal.name}: {signal.value:.3f} reported but {signal.note}")
        if values.watermark_confidence is None:
            lines.append("watermark: not detected, treated as neutral rather than exculpatory")
        if values.face_similarity is None:
            lines.append("face similarity: unavailable, no identity conclusion drawn")
        return lines


def build_risk_scorer(thresholds: RiskThresholds, trust_uncalibrated: bool = False) -> RiskScorer:
    """Instantiate the configured risk scorer."""
    return WeightedRiskScorer(thresholds, trust_uncalibrated=trust_uncalibrated)
