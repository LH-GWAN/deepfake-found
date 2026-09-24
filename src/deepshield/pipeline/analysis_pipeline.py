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
signal, the model versions that produced them, a verdict from the risk engine,
and the limitations that qualify all of it. The record never claims that content
was generated from the user's photographs; it reports identity similarity and
synthetic-media likelihood as separate, independently fallible measurements.

The verdict concerns one subject: the requested user, otherwise the owner of a
registered asset the file descends from, otherwise the best-matching enrolled
identity. When a registered asset of the subject matches but their face cannot
be confirmed in the content, the protected file recorded for that asset is
re-analysed to learn whether it showed their face in the first place. Nothing
new is stored for this; the file is the one the protection pipeline already
wrote.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from deepshield.config import DeepShieldConfig
from deepshield.detection.deepfake import DeepfakeDetector, build_deepfake_detector
from deepshield.exceptions import InvalidMediaError, ModelNotAvailableError
from deepshield.face.aligner import FaceAligner, build_aligner
from deepshield.face.detector import FaceDetector, build_detector
from deepshield.face.embedder import FaceEmbedder, build_embedder
from deepshield.face.matcher import FaceMatcher, build_matcher
from deepshield.logging_utils import get_logger, safe_embedding_repr
from deepshield.media import (
    image_size,
    is_video_path,
    load_image,
    resize_image,
    sha256_file,
    validate_rgb,
)
from deepshield.protection.fingerprint import DefaultFingerprinter, hash_similarity
from deepshield.protection.watermark import Watermarker, build_watermarker
from deepshield.provenance.c2pa_adapter import build_c2pa_adapter
from deepshield.quality import face_quality_score
from deepshield.risk.scorer import EXACT_MATCH_BASIS, build_risk_scorer
from deepshield.storage.repository import (
    FileAssetRepository,
    FileProvenanceStore,
    IdentityRepository,
    build_asset_repository,
    build_identity_repository,
    build_provenance_store,
)
from deepshield.types import (
    AssetFingerprint,
    AssetRecord,
    DetectedFace,
    EvidenceRecord,
    IdentityProfile,
    MediaType,
    RiskEvidence,
    SimilarityResult,
    Verdict,
    WatermarkDetectionResult,
)

logger = get_logger(__name__)

DEEPFAKE_CROP_MARGIN = 0.25
EXACT_MATCH_PROVENANCE_CONFIDENCE = 1.0
WATERMARK_PROVENANCE_CONFIDENCE = 0.8
PERCEPTUAL_PROVENANCE_CONFIDENCE = 0.4
DECISION_RANK = {"high_confidence": 3, "candidate": 2, "ambiguous": 1, "no_match": 0}
# A copy within this fraction of the registered size is read as it is; the
# grid search already covers that much. Aspect ratios further apart than
# ASPECT_TOLERANCE mean a crop, which restoring the size would only distort.
SIZE_TOLERANCE = 0.01
ASPECT_TOLERANCE = 0.02
MAX_RESTORED_SIZES = 3


def _restorable(copy: tuple[int, int], original: tuple[int, int]) -> bool:
    """Return whether ``copy`` looks like ``original`` scaled, and not already at its size."""
    (width, height), (target_width, target_height) = copy, original
    same = (
        abs(width - target_width) <= SIZE_TOLERANCE * target_width
        and abs(height - target_height) <= SIZE_TOLERANCE * target_height
    )
    aspect, target_aspect = width / height, target_width / target_height
    return not same and abs(aspect - target_aspect) <= ASPECT_TOLERANCE * target_aspect


def subject_result(
    face_matches: list[list[SimilarityResult]], subject: str
) -> SimilarityResult | None:
    """Return the subject's strongest result across every face in the content.

    A face whose best-matching enrolled identity is someone else does not count
    as the subject's face even when the subject also clears a threshold on it;
    that is reported as ``ambiguous`` at most, never as a confident match.
    """
    best: SimilarityResult | None = None
    for ranked in face_matches:
        own = next((r for r in ranked if r.matched_user_id == subject), None)
        if own is None:
            continue
        if ranked[0].matched_user_id != subject and own.decision != "no_match":
            own = replace(own, decision="ambiguous", is_high_confidence=False)
        if best is None or (DECISION_RANK[own.decision], own.similarity) > (
            DECISION_RANK[best.decision],
            best.similarity,
        ):
            best = own
    return best


def describe_credentials(credentials: dict[str, Any]) -> str:
    """Return the one-line reading of a C2PA verification result.

    Credentials are reported, never scored. A valid signature from an unknown
    signer proves that the file has not changed since someone signed it, and
    nothing about who that someone is; the absence of credentials proves
    nothing at all, since almost no photograph carries them yet.
    """
    if not credentials.get("supported"):
        return (
            "C2PA content credentials were not checked: the 'provenance' extra is not "
            "installed. Absence of a check is not absence of credentials."
        )
    if credentials.get("present") is False:
        return (
            "The file carries no C2PA content credentials. Most photographs do not, so "
            "this says nothing about authenticity."
        )
    if credentials.get("present") is None:
        return f"C2PA content credentials could not be read: {credentials.get('reason')}"
    signer = credentials.get("issuer") or "an unnamed signer"
    generator = credentials.get("claim_generator") or "an unnamed tool"
    if credentials.get("trusted"):
        return (
            f"C2PA content credentials verified and trusted: signed by {signer} using "
            f"{generator}. The bytes have not changed since signing."
        )
    if credentials.get("verified"):
        return (
            f"C2PA content credentials verified but not trusted: signed by {signer} using "
            f"{generator}, and the bytes have not changed since signing, but the signing "
            "certificate is not on a configured trust list, so the signer's identity is "
            "asserted rather than vouched for."
        )
    codes = ", ".join(str(issue.get("code")) for issue in credentials.get("issues") or [])
    return (
        f"C2PA content credentials are present but did not validate ({codes}). The file "
        "was altered after signing or the manifest is malformed; treat the credential as "
        "absent, not as proof of tampering by any particular party."
    )


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
        self.c2pa = build_c2pa_adapter(config.provenance.c2pa_backend)
        self.scorer = build_risk_scorer(config.thresholds)

    def _profiles(self, user_id: str | None) -> list[IdentityProfile]:
        """Return the identity templates this analysis should compare against."""
        if user_id is not None:
            return [self.identities.require(user_id)]
        return self.identities.load_all()

    def _analyse_faces(
        self, image: np.ndarray, profiles: list[IdentityProfile]
    ) -> tuple[
        list[dict[str, Any]],
        SimilarityResult | None,
        DetectedFace | None,
        list[list[SimilarityResult]],
    ]:
        """Detect, embed and match every face.

        Returns the per-face report rows, the best identity hit with its face,
        and every face's full ranking over the compared identities, from which
        the decision for any one subject can be read.
        """
        faces = self.detector.detect(image)
        records: list[dict[str, Any]] = []
        best_result: SimilarityResult | None = None
        best_face: DetectedFace | None = None
        face_matches: list[list[SimilarityResult]] = []

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
            face_pixels = float(min(face.bbox.width, face.bbox.height))
            quality = face_quality_score(face_pixels, aligned.image)
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

            ranked = [
                replace(result, probe_face_pixels=face_pixels)
                for result in self.matcher.match_many(embedding.vector, comparable, quality)
            ]
            face_matches.append(ranked)
            result = ranked[0]
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

        return records, best_result, best_face, face_matches

    def _owner_face_in_original(self, asset: AssetRecord) -> bool | None:
        """Re-analyse an asset's protected file to learn whether it showed its owner.

        Returns ``True`` when a face matches the owner at high confidence,
        ``False`` when no face reaches even the candidate threshold, and
        ``None`` when the file is gone, unreadable, the owner is not enrolled,
        or the only resemblance is borderline.
        """
        profile = self.identities.get(asset.user_id)
        if profile is None or not asset.protected_path:
            return None
        try:
            original = validate_rgb(load_image(Path(asset.protected_path)))
            _, _, _, face_matches = self._analyse_faces(original, [profile])
        except (InvalidMediaError, ModelNotAvailableError) as exc:
            logger.info("could not re-check asset %s: %s", asset.asset_id, exc)
            return None
        best = subject_result(face_matches, asset.user_id)
        if best is None or best.decision == "no_match":
            return False
        if best.decision == "high_confidence":
            return True
        return None

    def registered_size(self, asset: AssetRecord) -> tuple[int, int] | None:
        """Return the ``(width, height)`` a registered asset was protected at, if known."""
        fingerprint = asset.fingerprint
        if fingerprint.width and fingerprint.height:
            return fingerprint.width, fingerprint.height
        return image_size(asset.protected_path) if asset.protected_path else None

    def resembling(
        self, phash: str, registered: list[AssetRecord]
    ) -> list[tuple[float, AssetRecord]]:
        """Return registered assets at or above the perceptual evidence floor, closest first."""
        floor = self.config.thresholds.fingerprint.evidence_similarity_threshold
        scored = [
            (hash_similarity(asset.fingerprint.phash, phash), asset)
            for asset in registered
            if len(asset.fingerprint.phash) == len(phash)
        ]
        return sorted(
            (item for item in scored if item[0] >= floor), key=lambda item: item[0], reverse=True
        )

    def read_watermark(
        self, image: np.ndarray, candidates: list[AssetRecord]
    ) -> tuple[WatermarkDetectionResult, tuple[int, int] | None]:
        """Detect the watermark, first at the size of each registered photo the image resembles.

        A copy that was scaled down keeps its watermark but not the 8x8 grid it
        is read on, and the grid search only undoes enlargement. Restoring the
        copy to the size its registered original was protected at puts the grid
        back: a 70% copy, the size a phone upload or a vertical video frame
        gets, then decodes, while at 50% the mark is too weakened either way.
        The checksum still decides; a restoration that is wrong yields no code,
        not a wrong one.

        Returns the detection and the ``(width, height)`` it was read at when a
        restoration was what found it.
        """
        height, width = image.shape[:2]
        tried: set[tuple[int, int]] = set()
        for asset in candidates:
            size = self.registered_size(asset)
            if size is None or size in tried or not _restorable((width, height), size):
                continue
            if len(tried) >= MAX_RESTORED_SIZES:
                break
            tried.add(size)
            restored = self.watermarker.detect(resize_image(image, *size))
            if restored.detected:
                return restored, size
        return self.watermarker.detect(image), None

    def _source_attribution(
        self,
        fingerprint: AssetFingerprint,
        registered: list[AssetRecord],
        file_digest: str,
        watermark_code: str | None,
    ) -> dict[str, Any]:
        """Match the file against registered assets by hash, watermark and pHash."""
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

        evidence_floor = self.config.thresholds.fingerprint.evidence_similarity_threshold
        matched = exact or by_watermark or best_asset
        if exact is not None:
            confidence = EXACT_MATCH_PROVENANCE_CONFIDENCE
            basis = EXACT_MATCH_BASIS
        elif by_watermark is not None:
            confidence = WATERMARK_PROVENANCE_CONFIDENCE
            basis = "watermark code"
        elif best_asset is not None and best_similarity >= evidence_floor:
            confidence = PERCEPTUAL_PROVENANCE_CONFIDENCE
            basis = "perceptual hash"
        else:
            confidence = None
            basis = "no registered asset matched"

        return {
            "fingerprint": fingerprint,
            "asset": matched if matched and confidence else None,
            "matched_asset_id": matched.asset_id if matched and confidence else None,
            "distribution_id": matched.distribution_id if matched and confidence else None,
            "perceptual_similarity": best_similarity if registered else None,
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

        fingerprint = self.fingerprinter.fingerprint_image(image, "probe")
        registered = self.assets.list_assets()
        watermark, restored_size = self.read_watermark(
            image, [asset for _, asset in self.resembling(fingerprint.phash, registered)]
        )
        attribution = self._source_attribution(
            fingerprint, registered, file_digest, watermark.watermark_code
        )
        credentials = self.c2pa.verify(path)
        face_records, best_result, best_face, face_matches = self._analyse_faces(
            image, profiles
        )

        deepfake_score: float | None = None
        deepfake_notes: list[str] = []
        gated = True
        if best_result is not None and best_result.is_candidate and best_face is not None:
            crop = crop_with_margin(image, best_face, DEEPFAKE_CROP_MARGIN)
            outcome = self.deepfake_detector.predict_image(crop)
            deepfake_score = outcome.score
            deepfake_notes = list(outcome.notes)
            gated = False

        asset: AssetRecord | None = attribution["asset"]
        subject = (
            user_id
            or (asset.user_id if asset is not None else None)
            or (
                best_result.matched_user_id
                if best_result is not None and best_result.decision != "no_match"
                else None
            )
        )
        subject_match = subject_result(face_matches, subject) if subject else None
        compared = (
            any(profile.user_id == subject for profile in profiles) if subject else bool(profiles)
        ) and (bool(face_matches) or not face_records)
        owner_face_in_original: bool | None = None
        if (
            asset is not None
            and asset.user_id == subject
            and attribution["provenance_basis"] != EXACT_MATCH_BASIS
            and (subject_match is None or subject_match.decision != "high_confidence")
        ):
            owner_face_in_original = self._owner_face_in_original(asset)
        registered_at = self.registered_size(asset) if asset is not None else None
        copy_scale = (
            round(min(image.shape[1] / registered_at[0], image.shape[0] / registered_at[1]), 4)
            if registered_at
            else None
        )

        risk = self.scorer.assess(
            RiskEvidence(
                subject_user_id=subject,
                identities_compared=compared,
                faces_detected=len(face_records),
                identity_decision=(
                    subject_match.decision
                    if subject_match is not None
                    else "no_match" if compared and face_matches else None
                ),
                identity_similarity=(
                    subject_match.similarity if subject_match is not None else None
                ),
                probe_quality=(
                    subject_match.probe_quality if subject_match is not None else None
                ),
                probe_face_pixels=(
                    subject_match.probe_face_pixels if subject_match is not None else None
                ),
                copy_scale=copy_scale,
                asset_id=asset.asset_id if asset is not None else None,
                asset_owner=asset.user_id if asset is not None else None,
                asset_match_basis=(
                    attribution["provenance_basis"] if asset is not None else None
                ),
                distribution_id=attribution["distribution_id"],
                owner_face_in_original=owner_face_in_original,
                deepfake_score=(
                    deepfake_score
                    if best_result is not None and best_result.matched_user_id == subject
                    else None
                ),
                deepfake_calibrated=self.config.thresholds.deepfake.calibrated,
            )
        )

        limitations = list(risk.limitations)
        limitations.append(describe_credentials(credentials))
        if restored_size is not None:
            limitations.append(
                f"This {image.shape[1]}x{image.shape[0]} file is a resized copy; its watermark "
                f"was read after restoring it to the {restored_size[0]}x{restored_size[1]} "
                "size the registered original was protected at."
            )
        if attribution["perceptual_similarity"] is not None and asset is None:
            limitations.append(
                f"The closest registered asset matched at "
                f"{attribution['perceptual_similarity']:.3f} perceptual similarity, below the "
                f"{attribution['fingerprint_evidence_threshold']:.3f} evidence threshold. "
                "Unrelated images score around 0.5 by chance, so the file was not treated "
                "as descending from it."
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
        if best_result is not None and best_result.probe_quality is not None:
            raised = self.config.thresholds.face_similarity.low_quality_penalty * max(
                0.0, 1.0 - best_result.probe_quality
            )
            if raised >= 0.001:
                limitations.append(
                    f"Probe face quality was {best_result.probe_quality:.2f} of the reference "
                    "resolution and sharpness, so both similarity thresholds were raised by "
                    f"{raised:.3f} for this comparison."
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
            content_credentials=credentials,
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


def verdict_headline(record: EvidenceRecord) -> str | None:
    """Return the sentence naming the verdict, or ``None`` when there is none."""
    if record.risk is None:
        return None
    subject = record.risk.subject_user_id
    who = f"'{subject}'" if subject else "an enrolled identity"
    if record.media_type is MediaType.VIDEO:
        shown = {
            Verdict.OWN_COPY: f"This video shows a photograph registered by {who}.",
            Verdict.OWN_ALTERED: (
                f"This video shows a photograph registered by {who} with a face that no "
                "longer matches theirs."
            ),
            Verdict.OWN_UNVERIFIED: (
                f"This video shows a photograph registered by {who}, but their face could not "
                "be confirmed in it."
            ),
        }
        if record.risk.verdict in shown:
            return shown[record.risk.verdict]
    headlines = {
        Verdict.OWN_COPY: f"This is a copy of a photograph registered by {who}.",
        Verdict.OWN_ALTERED: (
            f"A photograph registered by {who} appears to have been altered: the face in it "
            "no longer matches theirs."
        ),
        Verdict.OWN_UNVERIFIED: (
            f"This descends from a photograph registered by {who}, but their face could not "
            "be confirmed in it."
        ),
        Verdict.IDENTITY_MATCH: (
            f"A face highly similar to {who} appears in content that is not one of their "
            "registered photographs; whether it is genuine or synthetic is undetermined."
        ),
        Verdict.SYNTHETIC_SUSPECTED: (
            f"Suspected synthetic content showing a face highly similar to {who} was found."
        ),
        Verdict.REVIEW: (
            "A face resembles an enrolled identity, but not strongly enough to identify anyone."
        ),
        Verdict.UNRELATED: f"No face resembling {who} was found.",
        Verdict.INCONCLUSIVE: (
            "No enrolled identity or registered asset was available to compare against."
        ),
    }
    return headlines[record.risk.verdict]


def summarize(record: EvidenceRecord) -> str:
    """Render the finding, phrased to match what was actually measured.

    The verdict comes first, then the identity measurement behind it. The
    wording is load-bearing. The system observes identity similarity, registered
    origin and a synthetic-media likelihood; it never observes that content was
    produced from the user's photographs, so it never says so. It also
    distinguishes a confirmed match from a borderline one, because reporting the
    second as the first is the cheapest way to make a system untrustworthy.
    """
    headline = verdict_headline(record)
    parts = [headline] if headline else []

    if record.face_similarity is None:
        parts.append("No face was compared against an enrolled identity.")
    elif record.matched_user_id is not None:
        parts.append(
            f"Best face similarity {record.face_similarity:.3f} to "
            f"'{record.matched_user_id}', above the confidence threshold."
        )
    elif record.identity_decision == "ambiguous":
        parts.append(
            f"A face scored {record.face_similarity:.3f} against two enrolled identities "
            "with too small a gap between them to identify either."
        )
    elif record.identity_decision == "candidate":
        parts.append(
            f"A face reached {record.face_similarity:.3f} similarity to an enrolled "
            "identity, above the review threshold but below the confidence threshold. "
            "This is worth reviewing, not a match."
        )
    else:
        parts.append(
            "No enrolled identity reached the similarity threshold in this content "
            f"(best similarity {record.face_similarity:.3f})."
        )

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
    elif record.watermark_detected is False:
        parts.append("No watermark was recovered, which is inconclusive on its own.")
    return " ".join(parts)
