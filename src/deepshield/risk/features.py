"""Risk feature assembly.

Turns raw per-component results into the normalised signal set the scorer
consumes. Three rules are enforced here rather than left to the scorer, because
they are the difference between a defensible score and a misleading one:

Missing is not zero
    A signal that could not be computed stays ``None``. Collapsing it to zero
    would silently read a failed detector as evidence of safety.

Positive evidence only, for both attribution signals
    An attacker can strip a watermark, so only a positive detection contributes.
    The same applies to perceptual fingerprints: two unrelated images agree on
    roughly half their hash bits by chance, so a similarity below the evidence
    threshold is noise and is reported without being scored.

Uncalibrated signals are marked
    A threshold that has never been fitted on real data produces a number with
    no defined meaning. Such signals are carried through and reported, but the
    scorer is told not to trust them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from deepshield.config import Thresholds
from deepshield.types import RiskFeatures


@dataclass
class SignalProvenance:
    """Where one risk signal came from and whether it can be trusted numerically."""

    name: str
    value: float | None
    source: str
    calibrated: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "name": self.name,
            "value": self.value,
            "source": self.source,
            "calibrated": self.calibrated,
            "detail": self.detail,
        }


@dataclass
class RiskFeatureSet:
    """Normalised features plus the provenance of every signal."""

    features: RiskFeatures
    provenance: list[SignalProvenance] = field(default_factory=list)

    def uncalibrated_names(self) -> list[str]:
        """Return the names of present-but-uncalibrated signals."""
        return [p.name for p in self.provenance if p.value is not None and not p.calibrated]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "features": self.features.to_dict(),
            "provenance": [p.to_dict() for p in self.provenance],
        }


class RiskFeatureBuilder(ABC):
    """Contract for converting component outputs into normalised risk features."""

    @abstractmethod
    def build(self, evidence: dict[str, Any]) -> RiskFeatureSet:
        """Return the normalised feature set for one analysed asset."""


class DefaultRiskFeatureBuilder(RiskFeatureBuilder):
    """Maps analysis-pipeline output onto the normalised risk feature set."""

    def __init__(self, thresholds: Thresholds | None = None) -> None:
        """Store the threshold set used to judge calibration status."""
        self.thresholds = thresholds or Thresholds()

    @staticmethod
    def _normalise_similarity(value: float | None) -> float | None:
        """Map a cosine similarity in ``[-1, 1]`` onto ``[0, 1]``.

        Negative similarity is not "less than no evidence"; it is simply a
        mismatch, so the lower half of the range is clamped rather than allowed
        to push the score below its floor.
        """
        if value is None:
            return None
        return float(max(0.0, min(1.0, value)))

    def build(self, evidence: dict[str, Any]) -> RiskFeatureSet:
        """Assemble features and record where each one came from."""
        face = self._normalise_similarity(evidence.get("face_similarity"))
        deepfake = evidence.get("deepfake_score")
        watermark_detected = evidence.get("watermark_detected")
        watermark_confidence = evidence.get("watermark_confidence")
        fingerprint = evidence.get("fingerprint_similarity")
        provenance_confidence = evidence.get("provenance_confidence")
        manipulation = evidence.get("manipulation_score")
        source_risk = evidence.get("source_risk")

        watermark_signal = (
            float(watermark_confidence)
            if watermark_detected and watermark_confidence is not None
            else None
        )

        features = RiskFeatures(
            face_similarity=face,
            deepfake_score=None if deepfake is None else float(deepfake),
            watermark_confidence=watermark_signal,
            fingerprint_similarity=None if fingerprint is None else float(fingerprint),
            provenance_confidence=(
                None if provenance_confidence is None else float(provenance_confidence)
            ),
            manipulation_score=None if manipulation is None else float(manipulation),
            source_risk=None if source_risk is None else float(source_risk),
        )

        provenance = [
            SignalProvenance(
                name="face_similarity",
                value=features.face_similarity,
                source=str(evidence.get("face_model", "unknown")),
                calibrated=self.thresholds.face_similarity.calibrated,
                detail="cosine similarity to the enrolled identity template",
            ),
            SignalProvenance(
                name="deepfake_score",
                value=features.deepfake_score,
                source=str(evidence.get("deepfake_model", "unknown")),
                calibrated=self.thresholds.deepfake.calibrated,
                detail="synthetic-media likelihood, not a verdict",
            ),
            SignalProvenance(
                name="watermark_confidence",
                value=features.watermark_confidence,
                source=str(evidence.get("watermark_backend", "unknown")),
                calibrated=True,
                detail=(
                    "positive evidence only; absence of a watermark is treated as neutral"
                ),
            ),
            SignalProvenance(
                name="fingerprint_similarity",
                value=features.fingerprint_similarity,
                source="perceptual hash",
                calibrated=True,
                detail=(
                    "positive evidence only; similarity below the evidence threshold "
                    "is chance agreement and is not scored"
                ),
            ),
            SignalProvenance(
                name="provenance_confidence",
                value=features.provenance_confidence,
                source="local provenance log",
                calibrated=True,
                detail="self-asserted lineage recorded by this system",
            ),
        ]
        return RiskFeatureSet(features=features, provenance=provenance)
