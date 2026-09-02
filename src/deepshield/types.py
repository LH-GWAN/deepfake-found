"""Core data structures shared across every DeepShield phase.

These are plain dataclasses rather than pydantic models because most of them
carry ``numpy`` arrays. Each one exposes ``to_dict`` so results can be serialised
into JSON evidence reports or flat experiment rows without leaking raw biometric
vectors.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np

from deepshield.logging_utils import embedding_digest


def utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


class MediaType(StrEnum):
    """Kind of media a record refers to."""

    IMAGE = "image"
    VIDEO = "video"
    FRAME = "frame"


class RiskLevel(StrEnum):
    """Qualitative risk band derived from the numeric risk score."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class ModelInfo:
    """Identity of a model used to produce a result.

    Recorded on every result so experiments stay reproducible and so a score can
    always be traced back to the exact detector that produced it.
    """

    name: str
    version: str
    backend: str
    training_dataset: str | None = None
    input_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned face box in pixel coordinates, ``x2``/``y2`` exclusive."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    def iou(self, other: BoundingBox) -> float:
        """Return the intersection-over-union overlap with another box."""
        inter_w = max(0.0, min(self.x2, other.x2) - max(self.x1, other.x1))
        inter_h = max(0.0, min(self.y2, other.y2) - max(self.y1, other.y1))
        intersection = inter_w * inter_h
        union = self.area + other.area - intersection
        return 0.0 if union <= 0.0 else intersection / union

    def to_list(self) -> list[float]:
        """Return ``[x1, y1, x2, y2]``."""
        return [self.x1, self.y1, self.x2, self.y2]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)


@dataclass(frozen=True)
class DetectedFace:
    """One face located in an image, before alignment."""

    bbox: BoundingBox
    detection_confidence: float
    landmarks: np.ndarray | None = None
    frame_index: int | None = None
    timestamp_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping without raw landmark coordinates."""
        return {
            "bbox": self.bbox.to_list(),
            "detection_confidence": self.detection_confidence,
            "has_landmarks": self.landmarks is not None,
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
        }


@dataclass(frozen=True)
class AlignedFace:
    """A face crop warped to the canonical pose expected by the embedder."""

    image: np.ndarray
    source: DetectedFace
    output_size: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping without pixel data."""
        return {
            "shape": list(self.image.shape),
            "output_size": self.output_size,
            "source": self.source.to_dict(),
        }


@dataclass(frozen=True)
class FaceEmbedding:
    """An L2-normalised face embedding plus the model that produced it."""

    vector: np.ndarray
    model: ModelInfo
    normalized: bool = True

    @property
    def dimension(self) -> int:
        """Number of embedding dimensions."""
        return int(self.vector.shape[-1])

    @property
    def digest(self) -> str:
        """Short non-reversible digest used as a log-safe identifier."""
        return embedding_digest(self.vector.tolist())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping that omits the raw vector."""
        return {
            "dimension": self.dimension,
            "normalized": self.normalized,
            "digest": self.digest,
            "model": self.model.to_dict(),
        }


@dataclass
class IdentityProfile:
    """Enrolled identity template for one user.

    Reference embeddings are kept alongside the centroid because averaging alone
    discards pose and lighting variation that a max- or top-k aggregation can use.
    """

    user_id: str
    reference_embeddings: np.ndarray
    centroid_embedding: np.ndarray
    image_count: int
    model: ModelInfo
    embedding_dimension: int
    created_at: str = field(default_factory=utc_now)
    source_image_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping that omits raw biometric vectors."""
        return {
            "user_id": self.user_id,
            "image_count": self.image_count,
            "embedding_dimension": self.embedding_dimension,
            "reference_count": int(self.reference_embeddings.shape[0]),
            "centroid_digest": embedding_digest(self.centroid_embedding.tolist()),
            "model": self.model.to_dict(),
            "created_at": self.created_at,
            "source_image_ids": list(self.source_image_ids),
        }


@dataclass(frozen=True)
class SimilarityResult:
    """Outcome of comparing one probe face against one identity profile."""

    matched_user_id: str | None
    similarity: float
    aggregation: str
    metric: str
    per_reference_similarity: list[float] = field(default_factory=list)
    euclidean_distance: float | None = None
    is_candidate: bool = False
    is_high_confidence: bool = False
    runner_up_similarity: float | None = None
    margin: float | None = None
    probe_quality: float | None = None
    decision: str = "no_match"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)


@dataclass(frozen=True)
class DeepfakeResult:
    """Synthetic-media likelihood for one image, crop or video.

    ``score`` is a detector output, never a verdict. A high score means the
    detector found patterns it associates with synthetic media, and detectors are
    known to generalise poorly to unseen generators.
    """

    score: float
    model: ModelInfo
    per_frame_scores: list[float] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "synthetic_probability": self.score,
            "model": self.model.to_dict(),
            "per_frame_scores": list(self.per_frame_scores),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class WatermarkPayload:
    """Watermark contents.

    Carries opaque identifiers only. Names, e-mail addresses and any other direct
    identifier stay in the database, keyed by ``asset_id``.
    """

    version: int
    user_token: str
    asset_id: str
    distribution_id: str | None = None
    timestamp: str = field(default_factory=utc_now)

    def code(self, bits: int = 32) -> int:
        """Return a stable opaque code identifying this payload.

        Only this code is carried inside the image. It is derived from the
        opaque identifiers, so it reveals nothing on its own; the database maps
        it back to the full payload.
        """
        material = f"{self.version}|{self.user_token}|{self.asset_id}|{self.distribution_id}"
        digest = hashlib.blake2s(material.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % (1 << bits)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)


@dataclass(frozen=True)
class WatermarkDetectionResult:
    """Outcome of a watermark extraction attempt.

    A negative result is weak evidence: an attacker can strip a watermark, so
    ``detected=False`` must be treated as neutral rather than exculpatory.
    """

    detected: bool
    confidence: float
    payload: WatermarkPayload | None = None
    watermark_code: str | None = None
    bit_accuracy: float | None = None
    backend: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "detected": self.detected,
            "confidence": self.confidence,
            "payload": self.payload.to_dict() if self.payload else None,
            "watermark_code": self.watermark_code,
            "bit_accuracy": self.bit_accuracy,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class AssetFingerprint:
    """Three fingerprints of one asset, each answering a different question.

    ``sha256`` answers exact file identity, the perceptual hashes answer
    "is this a re-encoded or lightly edited copy", and the semantic embedding
    answers "does this depict the same scene".
    """

    asset_id: str
    sha256: str
    phash: str
    dhash: str
    semantic_embedding: np.ndarray | None = None
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping without the raw semantic vector."""
        return {
            "asset_id": self.asset_id,
            "sha256": self.sha256,
            "phash": self.phash,
            "dhash": self.dhash,
            "has_semantic_embedding": self.semantic_embedding is not None,
            "created_at": self.created_at,
        }


@dataclass
class AssetRecord:
    """A protected asset the user published, and how it was marked.

    This is the registry the analysis side matches suspect content against: it
    links an opaque watermark code and a set of fingerprints back to the user,
    the distribution channel and the protection settings used.
    """

    asset_id: str
    user_id: str
    fingerprint: AssetFingerprint
    watermark_code: str | None = None
    distribution_id: str | None = None
    protected_path: str | None = None
    source_path: str | None = None
    protection_version: str | None = None
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "asset_id": self.asset_id,
            "user_id": self.user_id,
            "fingerprint": self.fingerprint.to_dict(),
            "watermark_code": self.watermark_code,
            "distribution_id": self.distribution_id,
            "protected_path": self.protected_path,
            "source_path": self.source_path,
            "protection_version": self.protection_version,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class ProvenanceRecord:
    """One node of the asset provenance graph."""

    asset_id: str
    sha256: str
    created_at: str
    parent_asset: str | None = None
    watermark_id: str | None = None
    protection_version: str | None = None
    software_version: str | None = None
    protection_config_digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)


@dataclass(frozen=True)
class RiskFeatures:
    """Normalised signals fed into the risk engine.

    Every field is optional: signals fail independently, and a missing signal
    must not be silently read as zero risk.
    """

    face_similarity: float | None = None
    deepfake_score: float | None = None
    watermark_confidence: float | None = None
    fingerprint_similarity: float | None = None
    provenance_confidence: float | None = None
    manipulation_score: float | None = None
    source_risk: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)


@dataclass(frozen=True)
class RiskAssessment:
    """Explainable risk output: a score, its evidence and its known limits."""

    risk_score: int
    risk_level: RiskLevel
    signals: dict[str, Any] = field(default_factory=dict)
    explanation: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "risk_score": self.risk_score,
            "risk_level": self.risk_level.value,
            "signals": dict(self.signals),
            "explanation": list(self.explanation),
            "limitations": list(self.limitations),
        }


@dataclass
class EvidenceRecord:
    """Every signal collected about one analysed asset, in one object.

    This is the unit that is persisted, reported and used to build experiment
    rows. Detector versions are mandatory so that a past analysis can be
    reproduced after models are upgraded.
    """

    source_id: str
    media_type: MediaType
    timestamp: str = field(default_factory=utc_now)

    face_detection_confidence: float | None = None
    face_similarity: float | None = None
    matched_user_id: str | None = None
    identity_decision: str = "no_match"
    identity_margin: float | None = None
    probe_quality: float | None = None

    deepfake_score: float | None = None

    watermark_detected: bool | None = None
    watermark_confidence: float | None = None
    watermark_payload: WatermarkPayload | None = None

    perceptual_similarity: float | None = None
    provenance_confidence: float | None = None

    risk: RiskAssessment | None = None
    detector_versions: dict[str, ModelInfo] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    analysis_id: str | None = None
    media_sha256: str | None = None
    faces: list[dict[str, Any]] = field(default_factory=list)
    matched_asset_id: str | None = None
    watermark_code: str | None = None
    processing_seconds: float | None = None
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON evidence-report representation of this record."""
        return {
            "analysis_id": self.analysis_id,
            "source_id": self.source_id,
            "media": {"type": self.media_type.value, "sha256": self.media_sha256},
            "timestamp": self.timestamp,
            "summary": self.summary,
            "identity": {
                "matched": self.matched_user_id is not None,
                "matched_user_id": self.matched_user_id,
                "similarity": self.face_similarity,
                "decision": self.identity_decision,
                "margin": self.identity_margin,
                "probe_quality": self.probe_quality,
                "face_detection_confidence": self.face_detection_confidence,
            },
            "deepfake": {"score": self.deepfake_score},
            "watermark": {
                "detected": self.watermark_detected,
                "confidence": self.watermark_confidence,
                "code": self.watermark_code,
                "payload": self.watermark_payload.to_dict() if self.watermark_payload else None,
            },
            "fingerprint": {
                "perceptual_similarity": self.perceptual_similarity,
                "matched_asset_id": self.matched_asset_id,
            },
            "faces": list(self.faces),
            "provenance": {"confidence": self.provenance_confidence},
            "risk": self.risk.to_dict() if self.risk else None,
            "detector_versions": {k: v.to_dict() for k, v in self.detector_versions.items()},
            "limitations": list(self.limitations),
            "processing_seconds": self.processing_seconds,
        }
