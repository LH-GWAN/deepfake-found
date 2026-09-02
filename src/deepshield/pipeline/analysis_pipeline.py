"""Image analysis pipeline: from a suspect file to an evidence record.

The stage order is the point of this module, not the individual components.

Cheap, always-on signals run first: file hash, perceptual fingerprints,
watermark extraction and a provenance lookup. Then faces are detected, aligned,
embedded and compared against every enrolled identity. Only faces whose
similarity clears the candidate threshold reach the expensive detector.

That gate is what makes the system affordable. Running a deepfake model on every
face in every submitted image would dominate the cost while saying nothing about
whether the user's own identity appears. Gating also keeps the false positive
rate down: a synthetic image of a stranger is not this user's problem, and
scoring it would only add noise to their report.

Identity reporting uses two thresholds, not one. Clearing the candidate
threshold is enough to justify spending an expensive detector on a face; it is
not enough to tell a user their face was found. Only a high-confidence decision,
with an unambiguous margin over the runner-up identity and an adequate probe
quality, populates ``matched_user_id``. A borderline face is reported as worth
reviewing, in those words.

The output is one :class:`~deepshield.types.EvidenceRecord` carrying every
signal, the model versions that produced them, an explainable risk assessment,
and the limitations that qualify all of it. The record never claims that content
was generated from the user's photographs; it reports identity similarity and
synthetic-media likelihood as separate, independently fallible measurements.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from deepshield.config import DeepShieldConfig
from deepshield.detection.deepfake import DeepfakeDetector, build_deepfake_detector
from deepshield.exceptions import InvalidMediaError
from deepshield.face.aligner import FaceAligner, build_aligner
from deepshield.face.detector import FaceDetector, build_detector
from deepshield.face.embedder import FaceEmbedder, build_embedder
from deepshield.face.matcher import FaceMatcher, build_matcher
from deepshield.logging_utils import get_logger, safe_embedding_repr
from deepshield.media import is_video_path, load_image, sha256_file, validate_rgb
from deepshield.protection.fingerprint import DefaultFingerprinter, hash_similarity
from deepshield.protection.watermark import Watermarker, build_watermarker
from deepshield.quality import face_quality_score
from deepshield.risk.features import DefaultRiskFeatureBuilder
from deepshield.risk.scorer import WeightedRiskScorer
from deepshield.storage.repository import (
    FileAssetRepository,
    FileProvenanceStore,
    IdentityRepository,
    build_asset_repository,
    build_identity_repository,
    build_provenance_store,
)
from deepshield.types import (
    DetectedFace,
    EvidenceRecord,
    IdentityProfile,
    MediaType,
    SimilarityResult,
)

logger = get_logger(__name__)

DEEPFAKE_CROP_MARGIN = 0.25
EXACT_MATCH_PROVENANCE_CONFIDENCE = 1.0
WATERMARK_PROVENANCE_CONFIDENCE = 0.8
PERCEPTUAL_PROVENANCE_CONFIDENCE = 0.4


class AnalysisPipeline(ABC):
    """Contract for analysing suspect media against enrolled identities."""

    @abstractmethod
    def analyze_image(self, image_path: Path, user_id: str | None = None) -> EvidenceRecord:
        """Analyse one image and return its evidence record."""

    @abstractmethod
    def analyze_video(self, video_path: Path, user_id: str | None = None) -> EvidenceRecord:
        """Analyse one video and return its aggregated evidence record."""


def crop_with_margin(image: np.ndarray, face: DetectedFace, margin: float) -> np.ndarray:
    """Return the face region expanded by ``margin`` on each side.

    Deepfake detectors are usually trained on loose face crops that include the
    blending boundary at the hairline and jaw, which is where face-swap artefacts
    concentrate. A tight box would cut away the most informative region.
    """
    height, width = image.shape[:2]
    box = face.bbox
    pad_x, pad_y = box.width * margin, box.height * margin
    x1 = int(max(0, round(box.x1 - pad_x)))
    y1 = int(max(0, round(box.y1 - pad_y)))
    x2 = int(min(width, round(box.x2 + pad_x)))
    y2 = int(min(height, round(box.y2 + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


class DefaultAnalysisPipeline(AnalysisPipeline):
    """Wires every detection component into the gated analysis order."""

    def __init__(
        self,
        config: DeepShieldConfig,
        detector: FaceDetector | None = None,
        aligner: FaceAligner | None = None,
        embedder: FaceEmbedder | None = None,
        matcher: FaceMatcher | None = None,
        deepfake_detector: DeepfakeDetector | None = None,
        watermarker: Watermarker | None = None,
        identity_repository: IdentityRepository | None = None,
        asset_repository: FileAssetRepository | None = None,
        provenance_store: FileProvenanceStore | None = None,
    ) -> None:
        """Build or accept every component the pipeline needs."""
        self.config = config
        self.detector = detector or build_detector(config.face.detector)
        self.aligner = aligner or build_aligner(config.face.aligner)
        self.embedder = embedder or build_embedder(config.face.embedder)
        self.matcher = matcher or build_matcher(
            config.face.matcher, config.thresholds.face_similarity
        )
        self.deepfake_detector = deepfake_detector or build_deepfake_detector(
            config.detection.deepfake
        )
        self.watermarker = watermarker or build_watermarker(config.protection.watermark)
        self.fingerprinter = DefaultFingerprinter(config.protection.fingerprint)
        self.identities = identity_repository or build_identity_repository(config)
        self.assets = asset_repository or build_asset_repository(config)
        self.provenance = provenance_store or build_provenance_store(config)
        self.feature_builder = DefaultRiskFeatureBuilder(config.thresholds)
        self.scorer = WeightedRiskScorer(config.thresholds.risk)

    def _profiles(self, user_id: str | None) -> list[IdentityProfile]:
        """Return the identity templates this analysis should compare against."""
        if user_id is not None:
            return [self.identities.require(user_id)]
        return self.identities.load_all()

    def _analyse_faces(
        self, image: np.ndarray, profiles: list[IdentityProfile]
    ) -> tuple[list[dict[str, Any]], SimilarityResult | None, DetectedFace | None]:
        """Detect, embed and match every face, returning the best identity hit."""
        faces = self.detector.detect(image)
        records: list[dict[str, Any]] = []
        best_result: SimilarityResult | None = None
        best_face: DetectedFace | None = None

        for index, face in enumerate(faces):
            entry: dict[str, Any] = {
                "index": index,
                "bbox": face.bbox.to_list(),
                "detection_confidence": face.detection_confidence,
                "similarity": None,
                "matched_user_id": None,
                "candidate": False,
                "decision": "no_match",
                "probe_quality": None,
            }
            if not profiles:
                records.append(entry)
                continue

            aligned = self.aligner.align(image, face)
            embedding = self.embedder.embed(aligned.image)
            quality = face_quality_score(
                min(face.bbox.width, face.bbox.height), aligned.image
            )
            entry["probe_quality"] = round(quality, 4)
            logger.debug(
                "probe embedding %s", safe_embedding_repr(embedding.vector.tolist())
            )

            comparable = [
                profile
                for profile in profiles
                if profile.embedding_dimension == embedding.dimension
            ]
            skipped = len(profiles) - len(comparable)
            if skipped:
                entry["skipped_incomparable_identities"] = skipped
            if not comparable:
                records.append(entry)
                continue

            result = self.matcher.match_many(embedding.vector, comparable, quality)[0]
            entry.update(
                {
                    "similarity": round(result.similarity, 6),
                    "matched_user_id": result.matched_user_id,
                    "candidate": result.is_candidate,
                    "high_confidence": result.is_high_confidence,
                    "decision": result.decision,
                    "margin": None if result.margin is None else round(result.margin, 6),
                    "runner_up_similarity": (
                        None
                        if result.runner_up_similarity is None
                        else round(result.runner_up_similarity, 6)
                    ),
                    "euclidean_distance": result.euclidean_distance,
                }
            )
            records.append(entry)

            if best_result is None or result.similarity > best_result.similarity:
                best_result, best_face = result, face

        return records, best_result, best_face

    def _source_attribution(
        self, image: np.ndarray, file_digest: str, watermark_code: str | None
    ) -> dict[str, Any]:
        """Match the file against registered assets by hash, watermark and pHash."""
        fingerprint = self.fingerprinter.fingerprint_image(image, "probe")
        registered = self.assets.list_assets()

        exact = next(
            (a for a in registered if a.fingerprint.sha256 == file_digest), None
        )
        by_watermark = (
            next((a for a in registered if a.watermark_code == watermark_code), None)
            if watermark_code
            else None
        )

        best_asset = None
        best_similarity = 0.0
        for asset in registered:
            if len(asset.fingerprint.phash) != len(fingerprint.phash):
                continue
            similarity = hash_similarity(asset.fingerprint.phash, fingerprint.phash)
            if similarity > best_similarity:
                best_asset, best_similarity = asset, similarity

        matched = exact or by_watermark or best_asset
        if exact is not None:
            confidence = EXACT_MATCH_PROVENANCE_CONFIDENCE
            basis = "exact file hash"
        elif by_watermark is not None:
            confidence = WATERMARK_PROVENANCE_CONFIDENCE
            basis = "watermark code"
        elif best_asset is not None and best_similarity >= 0.9:
            confidence = PERCEPTUAL_PROVENANCE_CONFIDENCE
            basis = "perceptual hash"
        else:
            confidence = None
            basis = "no registered asset matched"

        evidence_floor = self.config.thresholds.fingerprint.evidence_similarity_threshold
        is_evidence = bool(registered) and best_similarity >= evidence_floor

        return {
            "fingerprint": fingerprint,
            "matched_asset_id": matched.asset_id if matched and confidence else None,
            "distribution_id": matched.distribution_id if matched and confidence else None,
            "perceptual_similarity": best_similarity if registered else None,
            "fingerprint_evidence": best_similarity if is_evidence else None,
            "fingerprint_evidence_threshold": evidence_floor,
            "provenance_confidence": confidence,
            "provenance_basis": basis,
            "registered_assets": len(registered),
        }

    def analyze_image(self, image_path: Path, user_id: str | None = None) -> EvidenceRecord:
        """Analyse one image and return its evidence record.

        Raises:
            InvalidMediaError: If the file cannot be decoded as an image.
            IdentityNotFoundError: If ``user_id`` names an unenrolled user.

        """
        started = time.perf_counter()
        path = Path(image_path)
        if is_video_path(path):
            raise InvalidMediaError(f"{path} looks like a video; use analyze_video")

        image = validate_rgb(load_image(path))
        file_digest = sha256_file(path)
        profiles = self._profiles(user_id)

        watermark = self.watermarker.detect(image)
        attribution = self._source_attribution(image, file_digest, watermark.watermark_code)
        face_records, best_result, best_face = self._analyse_faces(image, profiles)

        deepfake_score: float | None = None
        deepfake_notes: list[str] = []
        gated = True
        if best_result is not None and best_result.is_candidate and best_face is not None:
            crop = crop_with_margin(image, best_face, DEEPFAKE_CROP_MARGIN)
            outcome = self.deepfake_detector.predict_image(crop)
            deepfake_score = outcome.score
            deepfake_notes = list(outcome.notes)
            gated = False

        evidence_inputs = {
            "face_similarity": best_result.similarity if best_result else None,
            "face_model": self.embedder.model_info.name,
            "deepfake_score": deepfake_score,
            "deepfake_model": self.deepfake_detector.model_info.name,
            "watermark_detected": watermark.detected,
            "watermark_confidence": watermark.confidence,
            "watermark_backend": watermark.backend,
            "fingerprint_similarity": attribution["fingerprint_evidence"],
            "provenance_confidence": attribution["provenance_confidence"],
        }
        feature_set = self.feature_builder.build(evidence_inputs)
        risk = self.scorer.score(feature_set)

        limitations = list(risk.limitations)
        if (
            attribution["perceptual_similarity"] is not None
            and attribution["fingerprint_evidence"] is None
        ):
            limitations.append(
                f"The closest registered asset matched at "
                f"{attribution['perceptual_similarity']:.3f} perceptual similarity, below the "
                f"{attribution['fingerprint_evidence_threshold']:.3f} evidence threshold. "
                "Unrelated images score around 0.5 by chance, so this was reported but not "
                "counted toward the risk score."
            )
        if gated:
            limitations.append(
                "The synthetic-media detector was not run: no face passed the identity "
                "candidate threshold, so there was nothing to attribute to this user."
            )
        limitations.extend(deepfake_notes)
        if best_result is not None and best_result.decision == "candidate":
            limitations.append(
                f"The best identity score ({best_result.similarity:.3f}) cleared the "
                "candidate threshold but not the high-confidence threshold, so no identity "
                "match is asserted."
            )
        if best_result is not None and best_result.decision == "ambiguous":
            limitations.append(
                f"The probe scored {best_result.similarity:.3f} against "
                f"'{best_result.matched_user_id}' and "
                f"{best_result.runner_up_similarity:.3f} against another enrolled identity. "
                "That margin is too small to identify either of them."
            )
        if (
            best_result is not None
            and best_result.probe_quality is not None
            and best_result.probe_quality < 0.5
        ):
            limitations.append(
                f"Probe face quality was {best_result.probe_quality:.2f} of the reference "
                "resolution and sharpness, so the similarity threshold was raised for this "
                "comparison."
            )
        if not profiles:
            limitations.append(
                "No identity is enrolled, so no identity comparison was possible."
            )

        record = EvidenceRecord(
            source_id=path.name,
            media_type=MediaType.IMAGE,
            media_sha256=file_digest,
            face_detection_confidence=(
                best_face.detection_confidence if best_face is not None else None
            ),
            face_similarity=best_result.similarity if best_result else None,
            matched_user_id=(
                best_result.matched_user_id
                if best_result and best_result.is_high_confidence
                else None
            ),
            identity_decision=best_result.decision if best_result else "no_match",
            identity_margin=best_result.margin if best_result else None,
            probe_quality=best_result.probe_quality if best_result else None,
            deepfake_score=deepfake_score,
            watermark_detected=watermark.detected,
            watermark_confidence=watermark.confidence,
            watermark_code=watermark.watermark_code,
            perceptual_similarity=attribution["perceptual_similarity"],
            matched_asset_id=attribution["matched_asset_id"],
            provenance_confidence=attribution["provenance_confidence"],
            risk=risk,
            faces=face_records,
            detector_versions={
                "face_embedder": self.embedder.model_info,
                "deepfake_detector": self.deepfake_detector.model_info,
            },
            limitations=limitations,
            processing_seconds=round(time.perf_counter() - started, 4),
        )
        record.summary = summarize(record)
        return record

    def analyze_video(self, video_path: Path, user_id: str | None = None) -> EvidenceRecord:
        """Analyse one video by delegating to the Phase 6 video processor."""
        from deepshield.video.processor import DefaultVideoProcessor

        processor = DefaultVideoProcessor(self.config, analysis=self)
        return processor.analyze(Path(video_path), user_id)


def summarize(record: EvidenceRecord) -> str:
    """Render the one-line finding, phrased to match what was actually measured.

    The wording is load-bearing. The system observes identity similarity and a
    synthetic-media likelihood; it never observes that content was produced from
    the user's photographs, so it never says so. It also distinguishes a
    confirmed match from a borderline one, because reporting the second as the
    first is the cheapest way to make a system untrustworthy.
    """
    if record.matched_user_id is None:
        if record.face_similarity is None:
            return "No face was compared against an enrolled identity."
        if record.identity_decision == "ambiguous":
            return (
                f"A face scored {record.face_similarity:.3f} against two enrolled identities "
                "with too small a gap between them to identify either."
            )
        if record.identity_decision == "candidate":
            return (
                f"A face reached {record.face_similarity:.3f} similarity to an enrolled "
                "identity, above the review threshold but below the confidence threshold. "
                "This is worth reviewing, not a match."
            )
        return (
            "No enrolled identity reached the similarity threshold in this content "
            f"(best similarity {record.face_similarity:.3f})."
        )

    parts = [
        f"Content containing a face with high similarity ({record.face_similarity:.3f}) "
        f"to the enrolled identity '{record.matched_user_id}' was found."
    ]
    if record.deepfake_score is not None:
        parts.append(
            f"The synthetic-media detector scored it {record.deepfake_score:.3f}; "
            "this is a likelihood, not a determination."
        )
    if record.watermark_detected:
        parts.append(
            f"A DeepShield watermark ({record.watermark_code}) was recovered, "
            "indicating the file descends from a registered protected asset."
        )
    else:
        parts.append("No watermark was recovered, which is inconclusive on its own.")
    return " ".join(parts)
