"""Phase 6 end-to-end: sampling, tracking, gating and aggregation on a real file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepshield.config import DeepShieldConfig, default_config
from deepshield.exceptions import InvalidMediaError
from deepshield.video.processor import DefaultVideoProcessor

pytestmark = pytest.mark.integration
cv2 = pytest.importorskip("cv2")


@pytest.fixture
def config(tmp_path: Path) -> DeepShieldConfig:
    base = default_config()
    return base.model_copy(
        update={
            "runtime": base.runtime.model_copy(
                update={
                    "data_dir": tmp_path / "data",
                    "results_dir": tmp_path / "data" / "results",
                    "model_dir": tmp_path / "models",
                }
            ),
            "storage": base.storage.model_copy(
                update={"embedding_store_dir": tmp_path / "data" / "embeddings"}
            ),
            "face": base.face.model_copy(
                update={
                    "detector": base.face.detector.model_copy(update={"backend": "mock"}),
                    "aligner": base.face.aligner.model_copy(update={"backend": "mock"}),
                    "embedder": base.face.embedder.model_copy(update={"backend": "mock"}),
                }
            ),
            "detection": base.detection.model_copy(
                update={"deepfake": base.detection.deepfake.model_copy(update={"backend": "mock"})}
            ),
        }
    )


@pytest.fixture
def two_scene_clip(tmp_path: Path) -> Path:
    """Return a clip whose two halves look clearly different, as after a cut."""
    path = tmp_path / "clip.mp4"
    size, frames, fps = 200, 100, 25.0
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    rng = np.random.default_rng(0)
    for index in range(frames):
        base = 210 if index < frames // 2 else 40
        frame = np.clip(
            np.full((size, size, 3), base, dtype=np.int16) + rng.integers(-15, 15, (size, size, 3)),
            0,
            255,
        ).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_video_analysis_produces_an_evidence_record(config, two_scene_clip: Path) -> None:
    record = DefaultVideoProcessor(config).analyze(two_scene_clip)
    assert record.media_type.value == "video"
    assert record.media_sha256
    assert record.faces
    assert record.risk is not None


def test_sampling_reduces_work_and_says_so(config, two_scene_clip: Path) -> None:
    record = DefaultVideoProcessor(config).analyze(two_scene_clip)
    assert any("frames were sampled" in line for line in record.limitations)


def test_a_cut_produces_separate_tracks(config, two_scene_clip: Path) -> None:
    """Merging two people into one track would silently drop one from the report."""
    record = DefaultVideoProcessor(config).analyze(two_scene_clip)
    assert len(record.faces) >= 2


def test_tracks_carry_timestamps(config, two_scene_clip: Path) -> None:
    record = DefaultVideoProcessor(config).analyze(two_scene_clip)
    for track in record.faces:
        assert track["first_timestamp"] is not None
        assert track["last_timestamp"] >= track["first_timestamp"]


def test_detector_is_gated_without_an_identity(config, two_scene_clip: Path) -> None:
    record = DefaultVideoProcessor(config).analyze(two_scene_clip)
    assert record.deepfake_score is None
    assert all(track["deepfake_score"] is None for track in record.faces)


def test_no_watermark_is_read_without_a_registered_photo(config, two_scene_clip: Path) -> None:
    record = DefaultVideoProcessor(config).analyze(two_scene_clip)
    assert record.watermark_detected is None
    assert record.matched_asset_id is None


def test_corrupt_video_is_rejected(config, tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(InvalidMediaError):
        DefaultVideoProcessor(config).analyze(broken)


def test_analysis_pipeline_delegates_video(config, two_scene_clip: Path) -> None:
    from deepshield.pipeline.analysis_pipeline import DefaultAnalysisPipeline

    record = DefaultAnalysisPipeline(config).analyze_video(two_scene_clip)
    assert record.media_type.value == "video"


def test_a_swap_inside_a_genuine_track_is_found(config, tmp_path: Path) -> None:
    """A spliced deepfake joins the genuine track around it; it must still be compared.

    The track's representative is a genuine frame, as the detection-confidence
    rule picks it on real footage. Only the third sampled frame carries the
    user's face.
    """
    from deepshield.pipeline.analysis_pipeline import DefaultAnalysisPipeline
    from deepshield.storage import build_identity_repository
    from deepshield.types import IdentityProfile, Verdict
    from deepshield.video.sampler import FrameSampler, SampledFrame
    from deepshield.video.tracker import FaceTrack, FaceTracker
    from tests.conftest import synthetic_photo
    from tests.integration.test_verdicts import CoarseLayoutEmbedder, replace_face

    stranger = synthetic_photo(seed=200, size=256)
    user = synthetic_photo(seed=5, size=256)
    images = [stranger, stranger, replace_face(stranger, user), stranger]

    class StillFrames(FrameSampler):
        def probe(self, video_path: Path) -> dict:
            return {"frame_count": len(images), "fps": 1.0}

        def sample(self, video_path: Path) -> list[SampledFrame]:
            return [
                SampledFrame(image=image, frame_number=n, timestamp_seconds=float(n))
                for n, image in enumerate(images)
            ]

    class OneTrack(FaceTracker):
        def track(self, detections_per_frame, frames=None, descriptors=None) -> list[FaceTrack]:
            faces = [face for per_frame in detections_per_frame for face in per_frame]
            return [
                FaceTrack(
                    track_id=0,
                    faces=faces,
                    frame_indices=list(range(len(faces))),
                    representative_index=0,
                )
            ]

        def select_representative(self, track, frames=None) -> FaceTrack:
            return track

    analysis = DefaultAnalysisPipeline(config, embedder=CoarseLayoutEmbedder())
    face = analysis.detector.detect(user)[0]
    vector = analysis.embedder.embed(analysis.aligner.align(user, face).image).vector
    build_identity_repository(config).save(
        IdentityProfile(
            user_id="u1",
            reference_embeddings=vector[None, :],
            centroid_embedding=vector,
            image_count=1,
            model=analysis.embedder.model_info,
            embedding_dimension=analysis.embedder.dimension,
        )
    )
    placeholder = tmp_path / "clip.mp4"
    placeholder.write_bytes(b"frames are supplied by the sampler")

    record = DefaultVideoProcessor(
        config, analysis=analysis, sampler=StillFrames(), tracker=OneTrack()
    ).analyze(placeholder, "u1")

    assert record.risk is not None
    assert record.risk.verdict is Verdict.IDENTITY_MATCH
    assert record.faces[0]["representative_frame"] == 0
    assert record.faces[0]["identity_frame"] == 2
    assert record.faces[0]["face_pixels"] == 128.0
    assert 0.0 <= record.faces[0]["probe_quality"] <= 1.0
    assert record.risk.signals["probe_face_pixels"] == 128.0
