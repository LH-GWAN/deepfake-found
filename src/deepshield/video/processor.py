"""Video analysis: sampling, tracking and per-track evidence aggregation.

Chains the video-specific stages onto the same detection components the image
pipeline uses, so a change to the face model or the risk engine applies to both
media types automatically.

Order: sample frames, detect faces in each, group them into tracks, embed and
match every sampled face of every track, then run the expensive detector only
on tracks where some face cleared the candidate threshold, on those faces first.

Every sampled face is embedded rather than one representative per track,
because a track follows a position and an appearance, not an identity. A
spliced deepfake keeps the head, the framing and the colours of the genuine
frames around it, so the tracker joins the swapped span to the genuine track;
with one embedding per track, a genuine frame was picked as the representative
and the swapped span was never compared. Sampling already bounds the work, so
embedding each sampled face costs a constant factor, not a new scaling.

Aggregation across tracks takes the maximum identity similarity, because a
person appearing in one second of a ten-minute video is exactly the case the
user cares about and any averaging would bury it. Frame-level deepfake scores
are combined with a trimmed mean instead, because there a single outlier frame
is far more likely to be a detector error than a finding.

Registered photographs are looked for in every sampled frame. The black bars a
photo is padded with are trimmed and the rest is compared with every registered
asset by perceptual hash, which costs milliseconds per frame. Only the frame
that resembles an asset most is kept, and only there is the watermark read,
restored first to the size the photo was protected at: a still shown at 70% of
its size, as a vertical video shows it, keeps a readable mark through H.264,
while one shrunk to half its size does not.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from deepshield.config import DeepShieldConfig
from deepshield.detection.deepfake_backends import aggregate_frame_scores
from deepshield.exceptions import InvalidMediaError
from deepshield.face.detector import FaceDetector
from deepshield.logging_utils import get_logger
from deepshield.media import content_region, sha256_file
from deepshield.protection.fingerprint import perceptual_hash
from deepshield.quality import face_quality_score
from deepshield.risk.scorer import build_risk_scorer
from deepshield.types import (
    AssetRecord,
    DetectedFace,
    EvidenceRecord,
    MediaType,
    RiskEvidence,
    SimilarityResult,
)
from deepshield.video.sampler import FrameSampler, build_sampler
from deepshield.video.tracker import FaceTracker, appearance_descriptor, build_tracker, crop_of

logger = get_logger(__name__)

MAX_DEEPFAKE_FRAMES_PER_TRACK = 8
CROP_PADDING = 0.5
MAX_CROP_SIDE = 384
# A padded region smaller than this is a thumbnail or a sliver, not a photo on screen.
MIN_STILL_SIDE = 64
MAX_ASSETS_READ = 3


@dataclass
class StillSighting:
    """The sampled frame that looked most like one registered photograph."""

    asset: AssetRecord
    similarity: float
    timestamp_seconds: float
    region: np.ndarray


@dataclass(frozen=True)
class FaceCrop:
    """The pixels around one detection, kept in place of the whole frame.

    ``face`` is the same detection in the crop's own coordinates, so the
    aligner and the synthetic-media crop work on it exactly as they would on
    the frame.
    """

    image: np.ndarray
    face: DetectedFace


def face_crop(image: np.ndarray, face: DetectedFace) -> FaceCrop:
    """Cut a detection out of its frame with room for alignment and blending seams.

    The padding covers the aligner's margin and the synthetic-media detector's
    wider one. Crops larger than ``MAX_CROP_SIDE`` are downscaled: both
    consumers resample to at most a few hundred pixels anyway.
    """
    height, width = image.shape[:2]
    box = face.bbox
    pad_x, pad_y = box.width * CROP_PADDING, box.height * CROP_PADDING
    x1 = int(max(0, min(width - 1, math.floor(box.x1 - pad_x))))
    y1 = int(max(0, min(height - 1, math.floor(box.y1 - pad_y))))
    x2 = int(max(x1 + 1, min(width, math.ceil(box.x2 + pad_x))))
    y2 = int(max(y1 + 1, min(height, math.ceil(box.y2 + pad_y))))
    region = np.ascontiguousarray(image[y1:y2, x1:x2])
    local = FaceDetector.shift_face(face, -x1, -y1)
    region, scale = FaceDetector.downscale_for_detection(region, MAX_CROP_SIDE)
    if scale != 1.0:
        local = FaceDetector.rescale_face(local, 1.0 / scale)
    return FaceCrop(image=np.ascontiguousarray(region), face=local)


class VideoProcessor(ABC):
    """Contract for end-to-end video analysis."""

    @abstractmethod
    def analyze(self, video_path: Path, user_id: str | None = None) -> EvidenceRecord:
        """Analyse a video and return per-track evidence plus an aggregate result."""


class DefaultVideoProcessor(VideoProcessor):
    """Sampling, tracking and gated per-track analysis."""

    def __init__(
        self,
        config: DeepShieldConfig,
        analysis: Any = None,
        sampler: FrameSampler | None = None,
        tracker: FaceTracker | None = None,
    ) -> None:
        """Build or accept the video stages and reuse the image components."""
        from deepshield.pipeline.analysis_pipeline import DefaultAnalysisPipeline

        self.config = config
        self.analysis = analysis or DefaultAnalysisPipeline(config)
        self.sampler = sampler or build_sampler(config.video.sampling)
        self.tracker = tracker or build_tracker(
            config.video.tracking, config.video.representative_frame
        )
        self.scorer = build_risk_scorer(config.thresholds)

    def _scan(
        self, path: Path, registered: list[AssetRecord]
    ) -> tuple[
        list[float],
        list[list[DetectedFace]],
        dict[int, FaceCrop],
        list[list[Any]],
        dict[str, StillSighting],
    ]:
        """Detect faces frame by frame, keeping only a crop of each.

        A frame is released as soon as its faces are cut out, so memory grows
        with the number of faces found rather than with the number of frames
        sampled times their resolution. The same pass compares each frame with
        the registered photographs and keeps, per photograph, the one frame
        region that resembled it most.

        Returns the sampled timestamps, the detections per frame in frame
        coordinates, each detection's crop keyed by ``id`` of the detection,
        each detection's appearance descriptor for the tracker, and the
        sightings of registered photographs keyed by asset id.
        """
        timestamps: list[float] = []
        detections: list[list[DetectedFace]] = []
        crops: dict[int, FaceCrop] = {}
        descriptors: list[list[Any]] = []
        sightings: dict[str, StillSighting] = {}
        hash_size = self.analysis.fingerprinter.hash_size
        for frame in self.sampler.iterate(path):
            if registered:
                region = content_region(frame.image)
                if min(region.shape[:2]) >= MIN_STILL_SIDE:
                    phash = perceptual_hash(region, hash_size)
                    for similarity, asset in self.analysis.resembling(phash, registered):
                        seen = sightings.get(asset.asset_id)
                        if seen is None or similarity > seen.similarity:
                            sightings[asset.asset_id] = StillSighting(
                                asset, similarity, frame.timestamp_seconds, region.copy()
                            )
            faces = [
                replace(
                    face,
                    frame_index=frame.frame_number,
                    timestamp_seconds=frame.timestamp_seconds,
                )
                for face in self.analysis.detector.detect(frame.image)
            ]
            frame_descriptors = []
            for face in faces:
                cut = face_crop(frame.image, face)
                crops[id(face)] = cut
                frame_descriptors.append(appearance_descriptor(crop_of(cut.image, cut.face.bbox)))
            timestamps.append(frame.timestamp_seconds)
            detections.append(faces)
            descriptors.append(frame_descriptors)
        return timestamps, detections, crops, descriptors, sightings

    def _still_evidence(
        self, sightings: dict[str, StillSighting], registered: list[AssetRecord]
    ) -> tuple[StillSighting | None, AssetRecord | None, str | None, Any]:
        """Read the watermark where a registered photograph was seen and pick the asset.

        Returns the sighting used, the asset the video is attributed to, the
        basis of that attribution, and the watermark detection (``None`` when
        no registered photograph was seen). A watermark code outranks the
        perceptual match, and names its own asset, which may be a different
        copy of the same photograph than the closest hash.
        """
        ranked = sorted(sightings.values(), key=lambda seen: seen.similarity, reverse=True)
        if not ranked:
            return None, None, None, None
        best_detection = None
        for seen in ranked[:MAX_ASSETS_READ]:
            detection, _ = self.analysis.read_watermark(seen.region, [seen.asset])
            if best_detection is None or (detection.detected, detection.confidence) > (
                best_detection.detected,
                best_detection.confidence,
            ):
                best_detection = detection
            if detection.detected:
                marked = next(
                    (a for a in registered if a.watermark_code == detection.watermark_code), None
                )
                if marked is not None:
                    return seen, marked, "watermark code", detection
        return ranked[0], ranked[0].asset, "perceptual hash", best_detection

    def analyze(self, video_path: Path, user_id: str | None = None) -> EvidenceRecord:
        """Analyse a video and return one aggregated evidence record.

        Raises:
            InvalidMediaError: If the video cannot be opened or decoded.

        """
        started = time.perf_counter()
        path = Path(video_path)
        metadata = self.sampler.probe(path)
        registered = self.analysis.assets.list_assets()
        timestamps, detections, crops, descriptors, sightings = self._scan(path, registered)
        if not timestamps:
            raise InvalidMediaError(f"no frames sampled from {path}")
        still, asset, asset_basis, watermark = self._still_evidence(sightings, registered)

        tracks = self.tracker.track(detections, descriptors=descriptors)
        profiles = self.analysis._profiles(user_id)

        track_records: list[dict[str, Any]] = []
        best_result: SimilarityResult | None = None
        best_confidence: float | None = None
        track_matches: list[list[SimilarityResult]] = []
        subject_scores: dict[str | None, list[float]] = {}
        gated_tracks = 0

        for track in tracks:
            face = track.representative
            entry: dict[str, Any] = {
                "track_id": track.track_id,
                "frames": len(track.faces),
                "first_timestamp": track.faces[0].timestamp_seconds,
                "last_timestamp": track.faces[-1].timestamp_seconds,
                "representative_frame": None if face is None else face.frame_index,
                "detection_confidence": None if face is None else face.detection_confidence,
                "similarity": None,
                "identity_frame": None,
                "matched_user_id": None,
                "candidate": False,
                "deepfake_score": None,
            }
            if face is None or not profiles:
                track_records.append(entry)
                continue

            per_face: list[tuple[int, Any, list[SimilarityResult]]] = []
            for step, detected in zip(track.frame_indices, track.faces, strict=False):
                cut = crops[id(detected)]
                aligned = self.analysis.aligner.align(cut.image, cut.face)
                embedding = self.analysis.embedder.embed(aligned.image)
                comparable = [
                    p for p in profiles if p.embedding_dimension == embedding.dimension
                ]
                if not comparable:
                    continue
                face_pixels = float(min(cut.face.bbox.width, cut.face.bbox.height))
                quality = face_quality_score(face_pixels, aligned.image)
                ranked = [
                    replace(result, probe_face_pixels=face_pixels)
                    for result in self.analysis.matcher.match_many(
                        embedding.vector, comparable, quality
                    )
                ]
                track_matches.append(ranked)
                per_face.append((step, detected, ranked))
            if not per_face:
                track_records.append(entry)
                continue

            _, strongest_face, strongest = max(per_face, key=lambda item: item[2][0].similarity)
            result = strongest[0]
            entry.update(
                {
                    "similarity": round(result.similarity, 6),
                    "identity_frame": strongest_face.frame_index,
                    "matched_user_id": result.matched_user_id,
                    "candidate": result.is_candidate,
                    "decision": result.decision,
                    "probe_quality": (
                        None if result.probe_quality is None else round(result.probe_quality, 4)
                    ),
                    "face_pixels": result.probe_face_pixels,
                }
            )

            if best_result is None or result.similarity > best_result.similarity:
                best_result = result
                best_confidence = strongest_face.detection_confidence

            if result.is_candidate:
                from deepshield.pipeline.analysis_pipeline import (
                    DEEPFAKE_CROP_MARGIN,
                    crop_with_margin,
                )

                matching = [item for item in per_face if item[2][0].is_candidate]
                others = [item for item in per_face if not item[2][0].is_candidate]
                chosen = (matching + others)[:MAX_DEEPFAKE_FRAMES_PER_TRACK]
                seams = [
                    crop_with_margin(
                        crops[id(detected)].image, crops[id(detected)].face, DEEPFAKE_CROP_MARGIN
                    )
                    for _, detected, _ in chosen
                ]
                outcome = self.analysis.deepfake_detector.predict_video(seams)
                entry["deepfake_score"] = round(outcome.score, 6)
                subject_scores.setdefault(result.matched_user_id, []).extend(
                    outcome.per_frame_scores or [outcome.score]
                )
            else:
                gated_tracks += 1

            track_records.append(entry)

        from deepshield.pipeline.analysis_pipeline import (
            PERCEPTUAL_PROVENANCE_CONFIDENCE,
            WATERMARK_PROVENANCE_CONFIDENCE,
            subject_result,
        )

        subject = (
            user_id
            or (asset.user_id if asset is not None else None)
            or (
                best_result.matched_user_id
                if best_result is not None and best_result.decision != "no_match"
                else None
            )
        )
        subject_match = subject_result(track_matches, subject) if subject else None
        compared = (
            any(profile.user_id == subject for profile in profiles) if subject else bool(profiles)
        ) and (bool(track_matches) or not tracks)
        owner_face_in_original: bool | None = None
        if (
            asset is not None
            and asset.user_id == subject
            and (subject_match is None or subject_match.decision != "high_confidence")
        ):
            owner_face_in_original = self.analysis._owner_face_in_original(asset)
        deepfake_scores = subject_scores.get(subject, []) if subject else []
        video_deepfake = (
            aggregate_frame_scores(
                deepfake_scores, self.config.detection.deepfake.frame_aggregation
            )
            if deepfake_scores
            else None
        )
        risk = self.scorer.assess(
            RiskEvidence(
                subject_user_id=subject,
                identities_compared=compared,
                faces_detected=len(tracks),
                identity_decision=(
                    subject_match.decision
                    if subject_match is not None
                    else "no_match" if compared and track_matches else None
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
                asset_id=asset.asset_id if asset is not None else None,
                asset_owner=asset.user_id if asset is not None else None,
                asset_match_basis=asset_basis,
                distribution_id=asset.distribution_id if asset is not None else None,
                owner_face_in_original=owner_face_in_original,
                deepfake_score=video_deepfake,
                deepfake_calibrated=self.config.thresholds.deepfake.calibrated,
            )
        )

        limitations = list(risk.limitations)
        interval = (
            timestamps[1] - timestamps[0] if len(timestamps) > 1 else None
        )
        limitations.append(
            f"Only {len(timestamps)} frames were sampled from {metadata.get('frame_count')} total"
            + (
                f", one every {interval:.2f} s; a face shown for less than that can fall "
                "between samples and go unexamined."
                if interval
                else "; content between samples was not examined."
            )
        )
        duration = metadata.get("duration_seconds")
        covered = timestamps[-1] + (interval or 0.0)
        if duration and covered < duration:
            limitations.append(
                f"Sampling stopped at {timestamps[-1]:.1f} s of a {duration:.1f} s "
                "video because the frame budget ran out; everything after that was not examined."
            )
        if gated_tracks:
            limitations.append(
                f"{gated_tracks} of {len(tracks)} tracks did not reach the identity "
                "candidate threshold, so no synthetic-media scoring was run on them."
            )
        if still is not None:
            limitations.append(
                f"A frame at {still.timestamp_seconds:.1f} s resembles registered asset "
                f"{still.asset.asset_id} (perceptual similarity {still.similarity:.3f}); the "
                "watermark was read only in the closest such frame per registered photograph."
            )
        elif registered:
            limitations.append(
                "No sampled frame resembled a registered photograph. Frames are compared "
                "whole, after trimming black bars, so a photograph shown cropped, zoomed or "
                "inside a larger composition is not recognised, and a photograph shown at "
                "half its size or less keeps no readable watermark."
            )

        record = EvidenceRecord(
            source_id=path.name,
            media_type=MediaType.VIDEO,
            media_sha256=sha256_file(path),
            face_detection_confidence=best_confidence,
            face_similarity=best_result.similarity if best_result else None,
            matched_user_id=(
                best_result.matched_user_id
                if best_result is not None and best_result.is_high_confidence
                else None
            ),
            identity_decision=best_result.decision if best_result else "no_match",
            identity_margin=best_result.margin if best_result else None,
            deepfake_score=video_deepfake,
            watermark_detected=watermark.detected if watermark is not None else None,
            watermark_confidence=watermark.confidence if watermark is not None else None,
            watermark_code=watermark.watermark_code if watermark is not None else None,
            perceptual_similarity=still.similarity if still is not None else None,
            matched_asset_id=asset.asset_id if asset is not None else None,
            provenance_confidence=(
                None
                if asset is None
                else WATERMARK_PROVENANCE_CONFIDENCE
                if asset_basis == "watermark code"
                else PERCEPTUAL_PROVENANCE_CONFIDENCE
            ),
            risk=risk,
            faces=track_records,
            detector_versions={
                "face_embedder": self.analysis.embedder.model_info,
                "deepfake_detector": self.analysis.deepfake_detector.model_info,
            },
            limitations=limitations,
            processing_seconds=round(time.perf_counter() - started, 4),
        )

        from deepshield.pipeline.analysis_pipeline import summarize

        record.summary = summarize(record)
        record.faces = track_records
        logger.info(
            "analysed %s: %d frames, %d tracks, best similarity %s",
            path.name,
            len(timestamps),
            len(tracks),
            "none" if best_result is None else f"{best_result.similarity:.3f}",
        )
        return record
