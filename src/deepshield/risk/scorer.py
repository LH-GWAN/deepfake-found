"""Verdict engine: turning independent evidence into one explainable conclusion.

An earlier version fused every signal into one weighted 0-100 score. Measured
end to end, it ranked the user's own protected photograph, reposted unchanged,
above a GAN face swap of the user: the watermark, fingerprint and provenance
matches that prove a file is the user's registered original were added to the
risk, and a cosine similarity of 0.66, far above the calibrated match threshold,
contributed 66 points as if it were a probability.

The signals do not measure one quantity. Registered-origin evidence says where
a file came from, identity evidence says whose face it shows, and a synthetic
media score says how it was made. So the engine answers with a verdict, a
category naming the situation, and attaches an urgency to that category instead
of computing one:

``own_copy``
    The content descends from one of the subject's registered assets and still
    shows their face, or is byte-identical to it. A leak or a repost, traceable
    to its distribution channel. Low urgency.
``own_altered``
    The content descends from the subject's registered asset, the registered
    original showed their face, and the content now shows someone else's face
    where theirs was, or a calibrated detector flags their face as synthetic.
    This is the target direction of a face swap. High urgency.
``own_unverified``
    The content descends from the subject's registered asset, but their face
    could not be confirmed in it and the original could not settle why.
``identity_match``
    The subject's face is present at high confidence in content that is not one
    of their registered assets. A genuine photograph and a face swap both land
    here while no calibrated synthetic-media detector is available, and the
    engine says so rather than guessing. Medium urgency.
``synthetic_suspected``
    As above, and a calibrated detector scores the face as synthetic.
``review``
    The best face cleared the candidate threshold but not the confidence
    threshold, or resembled two enrolled identities equally.
``unrelated`` / ``inconclusive``
    No face resembling the subject, or nothing to compare against. When the only
    faces are too small to match reliably, ``unrelated`` says it cannot rule the
    subject out, and a registered copy shrunk until its face is that small is
    ``own_unverified`` rather than ``own_altered``.

Registered-origin evidence is taken first because it is the most specific: a
matching file hash or watermark code names one asset, while a face match names
only a person. Uncalibrated signals never change a verdict; they are reported
in the explanation and listed in the limitations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from deepshield.config import Thresholds
from deepshield.quality import is_small_probe_face
from deepshield.types import RiskAssessment, RiskEvidence, RiskLevel, Verdict

BASE_LIMITATIONS = [
    "Face similarity does not prove that an image was used as training data.",
    "Deepfake detectors may fail on generator families absent from their training data.",
    "A watermark can be removed by cropping or regeneration, so its absence proves nothing.",
    "High face similarity alone is expected for genuine photographs of the user.",
]

SMALL_FACE_LIMITATION = (
    "A face under 80 pixels can fail to match its own owner under heavy compression, so "
    "finding no match on it does not show that it is someone else's face."
)
SHRUNK_COPY_SCALE = 0.9

EXACT_MATCH_BASIS = "exact file hash"


def _face_detail(evidence: RiskEvidence) -> str:
    """Return the size and quality of the decisive face, for the explanation."""
    parts = []
    if evidence.probe_face_pixels is not None:
        parts.append(f"{evidence.probe_face_pixels:.0f} pixels")
    if evidence.probe_quality is not None:
        parts.append(f"quality {evidence.probe_quality:.2f}")
    return f" ({', '.join(parts)})" if parts else ""


VERDICT_LEVELS: dict[Verdict, RiskLevel] = {
    Verdict.INCONCLUSIVE: RiskLevel.LOW,
    Verdict.UNRELATED: RiskLevel.LOW,
    Verdict.REVIEW: RiskLevel.LOW,
    Verdict.OWN_COPY: RiskLevel.LOW,
    Verdict.IDENTITY_MATCH: RiskLevel.MEDIUM,
    Verdict.OWN_UNVERIFIED: RiskLevel.MEDIUM,
    Verdict.OWN_ALTERED: RiskLevel.HIGH,
    Verdict.SYNTHETIC_SUSPECTED: RiskLevel.HIGH,
}


class RiskScorer(ABC):
    """Contract for turning evidence about one subject into an explainable verdict."""

    @abstractmethod
    def assess(self, evidence: RiskEvidence) -> RiskAssessment:
        """Return the verdict, its urgency, the reasons and the limitations."""


class VerdictRiskScorer(RiskScorer):
    """Deterministic rules over registered origin, identity and calibrated synthesis."""

    def __init__(self, thresholds: Thresholds | None = None) -> None:
        """Store the thresholds that decide when the synthetic-media score may count."""
        self.thresholds = thresholds or Thresholds()

    def assess(self, evidence: RiskEvidence) -> RiskAssessment:
        """Classify the evidence and return the verdict with its reasons."""
        reasons: list[str] = []
        limitations = list(BASE_LIMITATIONS)
        level: RiskLevel | None = None

        owns_asset = (
            evidence.asset_id is not None
            and evidence.subject_user_id is not None
            and evidence.asset_owner == evidence.subject_user_id
        )
        if owns_asset:
            verdict = self._own_asset(evidence, reasons, limitations)
        else:
            if evidence.asset_id is not None and evidence.asset_owner is not None:
                reasons.append(
                    f"The content descends from registered asset {evidence.asset_id}, which "
                    f"belongs to '{evidence.asset_owner}', not to the subject of this analysis."
                )
            verdict, level = self._identity_only(evidence, reasons, limitations)

        self._report_synthesis(evidence, reasons, limitations)
        return RiskAssessment(
            verdict=verdict,
            risk_level=level or VERDICT_LEVELS[verdict],
            subject_user_id=evidence.subject_user_id,
            signals=evidence.to_dict(),
            explanation=reasons,
            limitations=limitations,
        )

    def _synthetic_level(self, evidence: RiskEvidence) -> RiskLevel | None:
        """Return the urgency a calibrated synthetic score demands, or ``None``."""
        if evidence.deepfake_score is None or not evidence.deepfake_calibrated:
            return None
        limits = self.thresholds.deepfake
        if evidence.deepfake_score >= limits.high_confidence_threshold:
            return RiskLevel.CRITICAL
        if evidence.deepfake_score >= limits.suspicious_threshold:
            return RiskLevel.HIGH
        return None

    def _own_asset(
        self, evidence: RiskEvidence, reasons: list[str], limitations: list[str]
    ) -> Verdict:
        """Decide between a copy, an alteration and an unverifiable case."""
        channel = (
            f" published as '{evidence.distribution_id}'" if evidence.distribution_id else ""
        )
        reasons.append(
            f"The content descends from your registered asset {evidence.asset_id}{channel}, "
            f"matched by {evidence.asset_match_basis}."
        )
        exact = evidence.asset_match_basis == EXACT_MATCH_BASIS
        decision = evidence.identity_decision

        if exact:
            reasons.append(
                "It is byte-identical to the registered file, so nothing in it was changed."
            )
            return Verdict.OWN_COPY

        if decision == "high_confidence":
            if self._synthetic_level(evidence) is not None:
                reasons.append(
                    "Your face is present, but a calibrated detector scores it as synthetic, "
                    "so the face appears to have been regenerated."
                )
                return Verdict.OWN_ALTERED
            reasons.append("Your face is still present at high confidence.")
            return Verdict.OWN_COPY

        if evidence.asset_shielded:
            return self._shielded_asset(evidence, reasons, limitations)

        if not evidence.identities_compared:
            reasons.append(
                "No enrolled identity was compared, so whether your face was replaced cannot "
                "be checked."
            )
            limitations.append(
                "Enroll the asset owner to check registered copies for face changes."
            )
            return Verdict.OWN_UNVERIFIED

        original = evidence.owner_face_in_original
        if original is False:
            reasons.append(
                "The registered original does not show your face either, so no face of "
                "yours was removed."
            )
            return Verdict.OWN_COPY

        if evidence.faces_detected == 0:
            reasons.append(
                "No face is detectable in the content: it may have been cropped out, degraded "
                "beyond detection, or removed."
            )
        elif decision in ("candidate", "ambiguous"):
            reasons.append(
                "A face resembles yours only weakly, which a heavily degraded copy can also "
                "produce."
            )
        elif is_small_probe_face(evidence.probe_face_pixels) and (
            evidence.copy_scale is None or evidence.copy_scale < SHRUNK_COPY_SCALE
        ):
            shrunk = (
                ""
                if evidence.copy_scale is None
                else f" The copy is {evidence.copy_scale:.0%} of the registered original's size."
            )
            reasons.append(
                f"The face in this copy is too small{_face_detail(evidence)} to tell whether "
                "it is still yours: a genuine face this small can fail to match, so a replaced "
                f"face cannot be told apart from a shrunk one.{shrunk}"
            )
            limitations.append(SMALL_FACE_LIMITATION)
        elif original is True:
            reasons.append(
                "The registered original shows your face, but the face in this copy does not "
                "match you: another face appears where yours was."
            )
            return Verdict.OWN_ALTERED
        else:
            reasons.append("The faces in the content do not match you.")

        if original is None:
            reasons.append(
                "The registered original could not be re-checked, so whether it showed your "
                "face is unknown."
            )
            limitations.append(
                "The protected file recorded for this asset is missing or unreadable; keep it "
                "to let alterations be confirmed."
            )
        return Verdict.OWN_UNVERIFIED

    def _shielded_asset(
        self, evidence: RiskEvidence, reasons: list[str], limitations: list[str]
    ) -> Verdict:
        """Decide for a copy of a photo published with the swap shield.

        The shield keeps face recognisers from matching the photo to you, so a
        missed match with your enrollment says nothing here. The copy's face is
        compared with the face in the registered file instead.
        """
        reasons.append(
            "This photo was published with the swap shield, which keeps face recognisers, "
            "this one included, from matching it to you; its face is compared with the face "
            "in the registered file instead."
        )
        registered = evidence.face_in_registered_file
        if evidence.faces_detected == 0:
            reasons.append(
                "No face is detectable in the content: it may have been cropped out, degraded "
                "beyond detection, or removed."
            )
            return Verdict.OWN_UNVERIFIED
        if registered is True:
            reasons.append("The face is the one in the registered file, unchanged.")
            return Verdict.OWN_COPY
        if registered is False:
            if is_small_probe_face(evidence.probe_face_pixels) and (
                evidence.copy_scale is None or evidence.copy_scale < SHRUNK_COPY_SCALE
            ):
                reasons.append(
                    f"The face in this copy is too small{_face_detail(evidence)} to tell "
                    "whether it is still the one that was published."
                )
                limitations.append(SMALL_FACE_LIMITATION)
                return Verdict.OWN_UNVERIFIED
            reasons.append(
                "The face in this copy does not match the face in the registered file: "
                "another face appears where it was."
            )
            return Verdict.OWN_ALTERED
        reasons.append(
            "The registered file could not be compared with this copy, so whether its face "
            "was replaced is unknown."
        )
        limitations.append(
            "Keep the shielded file recorded for this asset; it is what copies are checked "
            "against for face changes."
        )
        return Verdict.OWN_UNVERIFIED

    def _identity_only(
        self, evidence: RiskEvidence, reasons: list[str], limitations: list[str]
    ) -> tuple[Verdict, RiskLevel | None]:
        """Decide from identity and calibrated synthesis alone."""
        if not evidence.identities_compared:
            reasons.append("No enrolled identity was available to compare against.")
            return Verdict.INCONCLUSIVE, None
        if evidence.faces_detected == 0:
            reasons.append("No face was detected in the content.")
            return Verdict.UNRELATED, None

        similarity = (
            "" if evidence.identity_similarity is None
            else f" (similarity {evidence.identity_similarity:.3f})"
        )
        decision = evidence.identity_decision
        if decision == "high_confidence":
            reasons.append(f"Your face appears at high confidence{similarity}.")
            synthetic = self._synthetic_level(evidence)
            if synthetic is not None:
                reasons.append("A calibrated detector scores the face as synthetic.")
                return Verdict.SYNTHETIC_SUSPECTED, synthetic
            reasons.append(
                "The content is not one of your registered assets, and whether it is a genuine "
                "photograph or a synthetic image of your face cannot be determined from the "
                "available signals."
            )
            return Verdict.IDENTITY_MATCH, None
        if decision == "ambiguous":
            reasons.append(
                f"A face resembles you and another enrolled identity almost equally{similarity}."
            )
            return Verdict.REVIEW, None
        if decision == "candidate":
            reasons.append(
                f"A face cleared the review threshold but not the confidence threshold"
                f"{similarity}."
            )
            return Verdict.REVIEW, None
        if is_small_probe_face(evidence.probe_face_pixels):
            reasons.append(
                f"No face cleared the review threshold{similarity}, but the best face is too "
                f"small{_face_detail(evidence)} to rule you out: a genuine photograph of you "
                "can score this low at that size."
            )
            limitations.append(SMALL_FACE_LIMITATION)
            return Verdict.UNRELATED, None
        reasons.append(f"No face resembles you{similarity}.")
        return Verdict.UNRELATED, None

    def _report_synthesis(
        self, evidence: RiskEvidence, reasons: list[str], limitations: list[str]
    ) -> None:
        """Describe the synthetic-media score without letting an uncalibrated one count."""
        if evidence.deepfake_score is None:
            return
        if evidence.deepfake_calibrated:
            reasons.append(f"Synthetic-media score {evidence.deepfake_score:.3f} (calibrated).")
            return
        reasons.append(
            f"Synthetic-media score {evidence.deepfake_score:.3f} was reported but did not "
            "affect the verdict: its threshold has not been calibrated."
        )
        limitations.append(
            "The synthetic-media detector is uncalibrated, so genuine photographs and "
            "synthetic images of the same face cannot be told apart."
        )


def build_risk_scorer(thresholds: Thresholds) -> RiskScorer:
    """Instantiate the verdict engine."""
    return VerdictRiskScorer(thresholds)
