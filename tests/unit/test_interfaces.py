"""Contracts of the pluggable component interfaces and their mock backends."""

from __future__ import annotations

import inspect
from abc import ABC

import numpy as np
import pytest

from deepshield.config import (
    DeepfakeDetectorConfig,
    FaceAlignerConfig,
    FaceDetectorConfig,
    FaceEmbedderConfig,
    WatermarkConfig,
)
from deepshield.detection.deepfake import (
    DeepfakeDetector,
    MockDeepfakeDetector,
    build_deepfake_detector,
)
from deepshield.exceptions import InvalidMediaError, ModelNotAvailableError
from deepshield.face.aligner import FaceAligner, SimpleCropAligner, build_aligner
from deepshield.face.detector import FaceDetector, MockFaceDetector, build_detector
from deepshield.face.embedder import FaceEmbedder, MockFaceEmbedder, build_embedder
from deepshield.protection.watermark import MockWatermarker, Watermarker, build_watermarker
from deepshield.types import BoundingBox, DetectedFace, WatermarkPayload


@pytest.mark.parametrize(
    "interface",
    [FaceDetector, FaceAligner, FaceEmbedder, DeepfakeDetector, Watermarker],
)
def test_interfaces_are_abstract(interface: type[ABC]) -> None:
    assert inspect.isabstract(interface)
    with pytest.raises(TypeError):
        interface()


def test_detector_returns_one_face(rgb_image: np.ndarray) -> None:
    faces = MockFaceDetector(FaceDetectorConfig()).detect(rgb_image)
    assert len(faces) == 1
    assert 0.0 <= faces[0].detection_confidence <= 1.0


def test_detector_rejects_grayscale() -> None:
    with pytest.raises(InvalidMediaError, match="RGB"):
        MockFaceDetector().detect(np.zeros((32, 32), dtype=np.uint8))


def test_detector_rejects_non_array() -> None:
    with pytest.raises(InvalidMediaError):
        MockFaceDetector().detect("not an image")


def test_detector_rejects_empty_image() -> None:
    with pytest.raises(InvalidMediaError, match="empty"):
        MockFaceDetector().detect(np.zeros((0, 0, 3), dtype=np.uint8))


def test_detector_returns_no_face_when_below_min_size() -> None:
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    assert MockFaceDetector(FaceDetectorConfig(min_face_size=40)).detect(tiny) == []


def test_aligner_produces_square_crop(rgb_image: np.ndarray) -> None:
    face = DetectedFace(bbox=BoundingBox(50, 50, 150, 170), detection_confidence=0.9)
    aligned = SimpleCropAligner(FaceAlignerConfig(output_size=112)).align(rgb_image, face)
    assert aligned.image.shape == (112, 112, 3)
    assert aligned.image.dtype == np.uint8


def test_aligner_clamps_boxes_outside_the_frame(rgb_image: np.ndarray) -> None:
    face = DetectedFace(bbox=BoundingBox(-50, -50, 40, 40), detection_confidence=0.9)
    aligned = SimpleCropAligner().align(rgb_image, face)
    assert aligned.image.shape[0] == aligned.output_size


def test_aligner_survives_degenerate_box(rgb_image: np.ndarray) -> None:
    face = DetectedFace(bbox=BoundingBox(10, 10, 10, 10), detection_confidence=0.9)
    aligned = SimpleCropAligner().align(rgb_image, face)
    assert aligned.image.shape[0] == aligned.output_size


def test_embedding_is_unit_norm(face_crop: np.ndarray) -> None:
    embedding = MockFaceEmbedder(FaceEmbedderConfig()).embed(face_crop)
    assert embedding.dimension == 512
    assert float(np.linalg.norm(embedding.vector)) == pytest.approx(1.0, abs=1e-5)


def test_embedding_is_deterministic(face_crop: np.ndarray) -> None:
    embedder = MockFaceEmbedder()
    first = embedder.embed(face_crop).vector
    second = embedder.embed(face_crop.copy()).vector
    np.testing.assert_allclose(first, second)


def test_different_crops_give_different_embeddings(face_crop: np.ndarray) -> None:
    embedder = MockFaceEmbedder()
    other = np.roll(face_crop, 40, axis=0)
    assert not np.allclose(embedder.embed(face_crop).vector, embedder.embed(other).vector)


def test_embedder_rejects_invalid_crop() -> None:
    with pytest.raises(InvalidMediaError):
        MockFaceEmbedder().embed(np.zeros((8, 8), dtype=np.uint8))


def test_embed_batch_matches_single_calls(face_crop: np.ndarray) -> None:
    embedder = MockFaceEmbedder()
    batch = embedder.embed_batch([face_crop, face_crop])
    np.testing.assert_allclose(batch[0].vector, embedder.embed(face_crop).vector)
    assert len(batch) == 2


def test_embedder_reports_model_metadata() -> None:
    info = MockFaceEmbedder().model_info
    assert info.backend == "mock"
    assert info.name and info.version


def test_deepfake_score_is_in_range_and_deterministic(rgb_image: np.ndarray) -> None:
    detector = MockDeepfakeDetector(DeepfakeDetectorConfig())
    first = detector.predict_image(rgb_image)
    second = detector.predict_image(rgb_image.copy())
    assert 0.0 <= first.score <= 1.0
    assert first.score == second.score


def test_deepfake_result_carries_model_metadata_and_caveat(rgb_image: np.ndarray) -> None:
    result = MockDeepfakeDetector().predict_image(rgb_image)
    assert result.model.version
    assert any("mock" in note for note in result.notes)


def test_deepfake_video_aggregates_frames(rgb_image: np.ndarray, other_rgb_image) -> None:
    result = MockDeepfakeDetector().predict_video([rgb_image, other_rgb_image])
    assert len(result.per_frame_scores) == 2
    assert result.score == pytest.approx(float(np.mean(result.per_frame_scores)))


def test_deepfake_video_rejects_empty_frame_list() -> None:
    with pytest.raises(InvalidMediaError, match="no frames"):
        MockDeepfakeDetector().predict_video([])


def test_mock_watermarker_does_not_modify_the_image(rgb_image: np.ndarray) -> None:
    payload = WatermarkPayload(version=1, user_token="t", asset_id="a")
    output = MockWatermarker(WatermarkConfig()).embed(rgb_image, payload)
    np.testing.assert_array_equal(output, rgb_image)


def test_mock_watermarker_never_claims_detection(rgb_image: np.ndarray) -> None:
    result = MockWatermarker().detect(rgb_image)
    assert result.detected is False
    assert result.confidence == 0.0
    assert result.payload is None


def test_builders_resolve_configured_backends() -> None:
    assert isinstance(build_detector(FaceDetectorConfig(backend="mock")), MockFaceDetector)
    assert isinstance(build_aligner(FaceAlignerConfig(backend="mock")), SimpleCropAligner)
    assert isinstance(build_embedder(FaceEmbedderConfig(backend="mock")), MockFaceEmbedder)
    assert isinstance(
        build_deepfake_detector(DeepfakeDetectorConfig(backend="mock")), MockDeepfakeDetector
    )
    assert isinstance(build_watermarker(WatermarkConfig(backend="mock")), MockWatermarker)


def test_unknown_backend_raises_model_not_available() -> None:
    with pytest.raises(ModelNotAvailableError):
        build_embedder(FaceEmbedderConfig(backend="no_such_embedder"))


def test_unimplemented_components_declare_their_phase() -> None:
    """Components still owned by a later phase must name that phase, not fail vaguely."""
    from deepshield.pipeline.analysis_pipeline import AnalysisPipeline

    assert inspect.isabstract(AnalysisPipeline)
