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


def test_watermark_scope_is_stated_for_video(config, two_scene_clip: Path) -> None:
    record = DefaultVideoProcessor(config).analyze(two_scene_clip)
    assert record.watermark_detected is None
    assert any("not run per frame on video" in line for line in record.limitations)


def test_corrupt_video_is_rejected(config, tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    with pytest.raises(InvalidMediaError):
        DefaultVideoProcessor(config).analyze(broken)


def test_analysis_pipeline_delegates_video(config, two_scene_clip: Path) -> None:
    from deepshield.pipeline.analysis_pipeline import DefaultAnalysisPipeline

    record = DefaultAnalysisPipeline(config).analyze_video(two_scene_clip)
    assert record.media_type.value == "video"
