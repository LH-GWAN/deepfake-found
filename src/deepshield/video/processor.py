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
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from deepshield.config import DeepShieldConfig
from deepshield.detection.deepfake_backends import aggregate_frame_scores
from deepshield.exceptions import InvalidMediaError
from deepshield.logging_utils import get_logger
from deepshield.media import sha256_file
from deepshield.risk.scorer import build_risk_scorer
from deepshield.types import EvidenceRecord, MediaType, RiskEvidence, SimilarityResult
from deepshield.video.sampler import FrameSampler, SampledFrame, build_sampler
from deepshield.video.tracker import FaceTracker, build_tracker

logger = get_logger(__name__)

MAX_DEEPFAKE_FRAMES_PER_TRACK = 8


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

    def _detect_all(self, frames: list[SampledFrame]) -> list[list[Any]]:
        """Detect faces in every sampled frame, tagging each with its position."""
        from dataclasses import replace

        per_frame = []
        for frame in frames:
            faces = self.analysis.detector.detect(frame.image)
            per_frame.append(
                [
                    replace(
                        face,
                        frame_index=frame.frame_number,
                        timestamp_seconds=frame.timestamp_seconds,
                    )
                    for face in faces
                ]
            )
        return per_frame

    def analyze(self, video_path: Path, user_id: str | None = None) -> EvidenceRecord:
        """Analyse a video and return one aggregated evidence record.

        Raises:
            InvalidMediaError: If the video cannot be opened or decoded.

        """
        started = time.perf_counter()
        path = Path(video_path)
        metadata = self.sampler.probe(path)
        frames = self.sampler.sample(path)
        if not frames:
            raise InvalidMediaError(f"no frames sampled from {path}")

        frame_lookup = {index: frame.image for index, frame in enumerate(frames)}
        detections = self._detect_all(frames)
        tracks = self.tracker.track(detections, [frame.image for frame in frames])
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
                aligned = self.analysis.aligner.align(frame_lookup[step], detected)
                embedding = self.analysis.embedder.embed(aligned.image)
                comparable = [
                    p for p in profiles if p.embedding_dimension == embedding.dimension
                ]
                if not comparable:
                    continue
                ranked = self.analysis.matcher.match_many(embedding.vector, comparable)
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
                crops = [
                    crop_with_margin(frame_lookup[step], detected, DEEPFAKE_CROP_MARGIN)
                    for step, detected, _ in chosen
                ]
                outcome = self.analysis.deepfake_detector.predict_video(crops)
                entry["deepfake_score"] = round(outcome.score, 6)
                subject_scores.setdefault(result.matched_user_id, []).extend(
                    outcome.per_frame_scores or [outcome.score]
                )
            else:
                gated_tracks += 1

            track_records.append(entry)

        from deepshield.pipeline.analysis_pipeline import subject_result

        subject = user_id or (
            best_result.matched_user_id
            if best_result is not None and best_result.decision != "no_match"
            else None
        )
        subject_match = subject_result(track_matches, subject) if subject else None
        compared = (
            any(profile.user_id == subject for profile in profiles) if subject else bool(profiles)
        ) and (bool(track_matches) or not tracks)
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
                deepfake_score=video_deepfake,
                deepfake_calibrated=self.config.thresholds.deepfake.calibrated,
            )
        )

        limitations = list(risk.limitations)
        interval = (
            frames[1].timestamp_seconds - frames[0].timestamp_seconds if len(frames) > 1 else None
        )
        limitations.append(
            f"Only {len(frames)} frames were sampled from {metadata.get('frame_count')} total"
            + (
                f", one every {interval:.2f} s; a face shown for less than that can fall "
                "between samples and go unexamined."
                if interval
                else "; content between samples was not examined."
            )
        )
        duration = metadata.get("duration_seconds")
        covered = frames[-1].timestamp_seconds + (interval or 0.0)
        if duration and covered < duration:
            limitations.append(
                f"Sampling stopped at {frames[-1].timestamp_seconds:.1f} s of a {duration:.1f} s "
                "video because the frame budget ran out; everything after that was not examined."
            )
        if gated_tracks:
            limitations.append(
                f"{gated_tracks} of {len(tracks)} tracks did not reach the identity "
                "candidate threshold, so no synthetic-media scoring was run on them."
            )
        limitations.append(
            "Watermark extraction is not run per frame on video; re-encoding a video "
            "destroys frame-level marks and the cost would not be justified."
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
            watermark_detected=None,
            watermark_confidence=None,
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
            len(frames),
            len(tracks),
            "none" if best_result is None else f"{best_result.similarity:.3f}",
        )
        return record
