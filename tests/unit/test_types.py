"""Core data structures and their serialisation contracts."""

from __future__ import annotations

import numpy as np

from deepshield.types import (
    AssetFingerprint,
    BoundingBox,
    DetectedFace,
    EvidenceRecord,
    FaceEmbedding,
    MediaType,
    ModelInfo,
    RiskAssessment,
    RiskFeatures,
    RiskLevel,
    WatermarkDetectionResult,
    WatermarkPayload,
)

MODEL = ModelInfo(name="m", version="1", backend="mock")


def test_bounding_box_geometry() -> None:
    box = BoundingBox(10, 20, 30, 60)
    assert box.width == 20
    assert box.height == 40
    assert box.area == 800


def test_bounding_box_iou_identical_is_one() -> None:
    box = BoundingBox(0, 0, 10, 10)
    assert box.iou(box) == 1.0


def test_bounding_box_iou_disjoint_is_zero() -> None:
    assert BoundingBox(0, 0, 10, 10).iou(BoundingBox(20, 20, 30, 30)) == 0.0


def test_bounding_box_degenerate_is_safe() -> None:
    box = BoundingBox(10, 10, 10, 10)
    assert box.area == 0.0
    assert box.iou(BoundingBox(0, 0, 5, 5)) == 0.0


def test_face_embedding_dict_omits_raw_vector() -> None:
    embedding = FaceEmbedding(vector=np.ones(8, dtype=np.float32), model=MODEL)
    payload = embedding.to_dict()
    assert payload["dimension"] == 8
    assert "vector" not in payload
    assert len(payload["digest"]) == 12


def test_detected_face_dict_omits_landmarks() -> None:
    face = DetectedFace(
        bbox=BoundingBox(0, 0, 10, 10),
        detection_confidence=0.9,
        landmarks=np.zeros((5, 2)),
    )
    payload = face.to_dict()
    assert payload["has_landmarks"] is True
    assert "landmarks" not in payload


def test_watermark_payload_carries_no_direct_identifier() -> None:
    payload = WatermarkPayload(version=1, user_token="tok", asset_id="a1")
    fields = set(payload.to_dict())
    assert fields == {"version", "user_token", "asset_id", "distribution_id", "timestamp"}


def test_watermark_negative_result_is_explicit() -> None:
    result = WatermarkDetectionResult(detected=False, confidence=0.0, backend="mock")
    payload = result.to_dict()
    assert payload["detected"] is False
    assert payload["payload"] is None


def test_asset_fingerprint_dict_omits_semantic_vector() -> None:
    fingerprint = AssetFingerprint(
        asset_id="a1",
        sha256="0" * 64,
        phash="ff",
        dhash="ee",
        semantic_embedding=np.ones(4),
    )
    payload = fingerprint.to_dict()
    assert payload["has_semantic_embedding"] is True
    assert "semantic_embedding" not in payload


def test_risk_features_default_to_none_not_zero() -> None:
    features = RiskFeatures()
    assert set(features.to_dict().values()) == {None}


def test_evidence_record_serialises_to_report_shape() -> None:
    record = EvidenceRecord(
        source_id="s1",
        media_type=MediaType.IMAGE,
        face_similarity=0.93,
        matched_user_id="u1",
        deepfake_score=0.81,
        watermark_detected=False,
        watermark_confidence=0.0,
        risk=RiskAssessment(risk_score=84, risk_level=RiskLevel.HIGH),
        detector_versions={"face_embedder": MODEL},
        limitations=["Face similarity does not prove training-data usage."],
    )
    payload = record.to_dict()
    assert payload["identity"]["similarity"] == 0.93
    assert payload["deepfake"]["score"] == 0.81
    assert payload["risk"]["risk_level"] == "HIGH"
    assert payload["detector_versions"]["face_embedder"]["backend"] == "mock"
    assert payload["limitations"]


def test_deepfake_result_field_is_named_probability() -> None:
    from deepshield.types import DeepfakeResult

    payload = DeepfakeResult(score=0.5, model=MODEL).to_dict()
    assert "synthetic_probability" in payload
    assert "is_fake" not in payload
