"""Video analysis: sampling, tracking and per-track evidence aggregation.

Chains the video-specific stages onto the same detection components the image
pipeline uses, so a change to the face model or the risk engine applies to both
media types automatically.

Order: sample frames, detect faces in each, group them into tracks, pick one
representative frame per track, embed and match only those, then run the
expensive detector only on tracks whose identity similarity cleared the
candidate threshold.

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
from deepshield.risk.features import DefaultRiskFeatureBuilder
from deepshield.risk.scorer import WeightedRiskScorer
from deepshield.types import EvidenceRecord, MediaType
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
        self.feature_builder = DefaultRiskFeatureBuilder(config.thresholds)
        self.scorer = WeightedRiskScorer(config.thresholds.risk)

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
        best_similarity: float | None = None
        best_user: str | None = None
        best_confidence: float | None = None
        deepfake_scores: list[float] = []
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
                "matched_user_id": None,
                "candidate": False,
                "deepfake_score": None,
            }
            if face is None or not profiles:
                track_records.append(entry)
                continue

            step = track.frame_indices[track.representative_index or 0]
            image = frame_lookup[step]
            aligned = self.analysis.aligner.align(image, face)
            embedding = self.analysis.embedder.embed(aligned.image)

            comparable = [
                p for p in profiles if p.embedding_dimension == embedding.dimension
            ]
            if not comparable:
                track_records.append(entry)
                continue

            result = self.analysis.matcher.match_many(embedding.vector, comparable)[0]
            entry.update(
                {
                    "similarity": round(result.similarity, 6),
                    "matched_user_id": result.matched_user_id,
                    "candidate": result.is_candidate,
                }
            )

            if best_similarity is None or result.similarity > best_similarity:
                best_similarity = result.similarity
                best_user = result.matched_user_id if result.is_candidate else None
                best_confidence = face.detection_confidence

            if result.is_candidate:
                from deepshield.pipeline.analysis_pipeline import (
                    DEEPFAKE_CROP_MARGIN,
                    crop_with_margin,
                )

                chosen = track.frame_indices[:MAX_DEEPFAKE_FRAMES_PER_TRACK]
                crops = [
                    crop_with_margin(frame_lookup[i], f, DEEPFAKE_CROP_MARGIN)
                    for i, f in zip(chosen, track.faces, strict=False)
                ]
                outcome = self.analysis.deepfake_detector.predict_video(crops)
                entry["deepfake_score"] = round(outcome.score, 6)
                deepfake_scores.extend(outcome.per_frame_scores or [outcome.score])
            else:
                gated_tracks += 1

            track_records.append(entry)

        video_deepfake = (
            aggregate_frame_scores(
                deepfake_scores, self.config.detection.deepfake.frame_aggregation
            )
            if deepfake_scores
            else None
        )

        feature_set = self.feature_builder.build(
            {
                "face_similarity": best_similarity,
                "face_model": self.analysis.embedder.model_info.name,
                "deepfake_score": video_deepfake,
                "deepfake_model": self.analysis.deepfake_detector.model_info.name,
                "watermark_detected": None,
                "watermark_confidence": None,
            }
        )
        risk = self.scorer.score(feature_set)

        limitations = list(risk.limitations)
        limitations.append(
            f"Only {len(frames)} frames were sampled at {self.config.video.sampling.fps} fps "
            f"from {metadata.get('frame_count')} total; content between samples was not examined."
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
            face_similarity=best_similarity,
            matched_user_id=best_user,
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
            "none" if best_similarity is None else f"{best_similarity:.3f}",
        )
        return record
