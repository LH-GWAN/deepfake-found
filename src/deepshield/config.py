"""Typed configuration loading for DeepShield.

Configuration lives in YAML under ``configs/`` and is validated into pydantic
models. Nothing in the codebase is allowed to hard-code a threshold: every
decision boundary is read from :class:`Thresholds` so it can be recalibrated on
real data without touching source files.

Backend defaults follow one rule: a component whose real implementation needs no
downloaded weights defaults to that implementation, and a component that would
need a download defaults to ``mock``. So watermarking and deepfake scoring are
real out of the box, while face detection and embedding stay mock until
``deepshield download-models`` has run. ``configs/default.yaml`` then selects the
real face backends, because by then the weights are expected to be present.

Precedence, lowest to highest:

1. defaults declared on the pydantic models
2. ``configs/default.yaml`` and ``configs/thresholds.yaml``
3. explicit overrides passed to :func:`load_config`
4. ``DEEPSHIELD_*`` environment variables
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from deepshield.exceptions import ConfigurationError

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_THRESHOLDS_PATH = Path("configs/thresholds.yaml")
ENV_PREFIX = "DEEPSHIELD_"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(_Base):
    """Static project identity recorded in every experiment and evidence record."""

    name: str = "deepshield"
    version: str = "0.1.0"
    phase: int = 0


class RuntimeConfig(_Base):
    """Execution environment: determinism, device selection and filesystem layout."""

    random_seed: int = 42
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    num_workers: int = 2
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    results_dir: Path = Path("data/results")


class LoggingConfig(_Base):
    """Logging behaviour, including biometric redaction rules."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    file: Path | None = None
    redact_embeddings: bool = True
    embedding_preview_dims: int = Field(default=4, ge=0, le=16)


class FaceDetectorConfig(_Base):
    """Face detection backend selection and filtering rules."""

    backend: str = "mock"
    min_face_size: int = Field(default=40, gt=0)
    detection_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    max_faces: int = Field(default=20, gt=0)
    model_path: Path | None = None
    nms_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    allow_download: bool = True
    max_detection_side: int = Field(default=1280, gt=0)


class FaceAlignerConfig(_Base):
    """Face alignment backend and canonical crop size."""

    backend: str = "mock"
    output_size: int = Field(default=112, gt=0)
    margin: float = Field(default=0.1, ge=0.0, le=1.0)


class FaceEmbedderConfig(_Base):
    """Embedding backend metadata, recorded with every similarity result."""

    backend: str = "mock"
    model_name: str = "mock-embedder"
    model_version: str = "0.1.0"
    embedding_dimension: int = Field(default=512, gt=0)
    normalize: bool = True
    model_path: Path | None = None
    allow_download: bool = True
    insightface_pack: str = "buffalo_l"
    flip_tta: bool = False
    ensemble: list[str] = Field(default_factory=list)


class FaceMatcherConfig(_Base):
    """How a probe embedding is scored against a set of reference embeddings."""

    metric: Literal["cosine", "euclidean"] = "cosine"
    aggregation: Literal["max", "mean", "topk_mean", "centroid"] = "max"
    top_k: int = Field(default=3, gt=0)
    log_euclidean: bool = True


class FaceConfig(_Base):
    """Container for the four face-pipeline stages."""

    detector: FaceDetectorConfig = FaceDetectorConfig()
    aligner: FaceAlignerConfig = FaceAlignerConfig()
    embedder: FaceEmbedderConfig = FaceEmbedderConfig()
    matcher: FaceMatcherConfig = FaceMatcherConfig()


class EnrollmentQualityConfig(_Base):
    """Minimum quality bar an enrollment image must clear.

    ``min_sharpness`` is an absolute floor on the Laplacian variance of the
    aligned crop; ``min_sharpness_ratio`` additionally rejects any image far
    blurrier than the median of the batch, which catches a soft photo in an
    otherwise sharp set without needing an absolute value tuned per camera.
    """

    min_detection_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    min_face_pixels: int = Field(default=80, gt=0)
    min_sharpness: float = Field(default=100.0, ge=0.0)
    min_sharpness_ratio: float = Field(default=0.15, ge=0.0, le=1.0)


class EnrollmentConfig(_Base):
    """Identity enrollment policy."""

    min_images: int = Field(default=3, gt=0)
    max_images: int = Field(default=10, gt=0)
    quality: EnrollmentQualityConfig = EnrollmentQualityConfig()
    keep_reference_embeddings: bool = True
    store_centroid: bool = True


class DeepfakeDetectorConfig(_Base):
    """Deepfake detector adapter selection and provenance metadata."""

    backend: str = "spectral"
    model_name: str = "spectral-artifact-heuristic"
    model_version: str = "0.1.0"
    input_size: int = Field(default=224, gt=0)
    batch_size: int = Field(default=8, gt=0)
    model_path: Path | None = None
    training_dataset: str | None = None
    positive_index: int = Field(default=1, ge=0)
    frame_aggregation: Literal["mean", "max", "trimmed_mean"] = "trimmed_mean"


class BackendOnlyConfig(_Base):
    """Backend selector for components with no extra Phase 0 parameters."""

    backend: str = "mock"


class DetectionConfig(_Base):
    """Detection-side adapters: synthetic media, watermark and manipulation."""

    deepfake: DeepfakeDetectorConfig = DeepfakeDetectorConfig()
    watermark: BackendOnlyConfig = BackendOnlyConfig()
    manipulation: BackendOnlyConfig = BackendOnlyConfig()


class WatermarkConfig(_Base):
    """Watermark embedding parameters."""

    backend: str = "dct"
    strength: float = Field(default=0.16, gt=0.0, le=1.0)
    payload_bits: int = Field(default=64, gt=0)
    soft_decode_bits: int = Field(default=12, ge=0, le=16)
    resync_enabled: bool = True
    resync_scales: list[float] = Field(
        default_factory=lambda: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    )
    resync_min_confidence: float = Field(default=0.15, ge=0.0, le=1.0)
    resync_candidates: int = Field(default=4, gt=0, le=32)
    resync_soft_decode_bits: int = Field(default=0, ge=0, le=16)
    resync_max_blocks: int = Field(default=2048, gt=0)


class FingerprintConfig(_Base):
    """Perceptual and semantic fingerprint parameters."""

    hash_size: int = Field(default=8, gt=0)
    semantic_embedding: bool = False


class AdversarialConfig(_Base):
    """Adversarial identity-cloaking research parameters, disabled by default."""

    enabled: bool = False
    epsilon: float = Field(default=0.03, gt=0.0, le=1.0)
    steps: int = Field(default=50, gt=0)
    step_size: float = Field(default=0.005, gt=0.0)


class ProtectionConfig(_Base):
    """Container for the three protection layers."""

    watermark: WatermarkConfig = WatermarkConfig()
    fingerprint: FingerprintConfig = FingerprintConfig()
    adversarial: AdversarialConfig = AdversarialConfig()


class VideoSamplingConfig(_Base):
    """Frame sampling policy; full-frame decoding is never the default."""

    strategy: Literal["uniform_fps", "scene_change", "adaptive"] = "uniform_fps"
    fps: float = Field(default=1.0, gt=0.0)
    max_frames: int = Field(default=600, gt=0)


class VideoTrackingConfig(_Base):
    """Face tracking parameters used to group faces across frames."""

    backend: str = "mock"
    iou_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    max_gap_frames: int = Field(default=5, ge=0)
    appearance_threshold: float = Field(default=0.75, ge=0.0, le=1.0)


class RepresentativeFrameConfig(_Base):
    """Criterion for picking one frame per track to embed."""

    criterion: Literal["detection_confidence", "frontal_pose", "sharpness"] = (
        "detection_confidence"
    )


class VideoConfig(_Base):
    """Container for the video pipeline configuration."""

    sampling: VideoSamplingConfig = VideoSamplingConfig()
    tracking: VideoTrackingConfig = VideoTrackingConfig()
    representative_frame: RepresentativeFrameConfig = RepresentativeFrameConfig()


class StorageConfig(_Base):
    """Persistence layout; biometric templates are stored apart from raw media."""

    database_url: str = "sqlite:///deepshield.db"
    separate_embedding_store: bool = True
    embedding_store_dir: Path = Path("data/embeddings")


class ApiConfig(_Base):
    """REST API binding and upload limits."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, gt=0, lt=65536)
    max_upload_mb: int = Field(default=50, gt=0)


class ExperimentsConfig(_Base):
    """Where machine-readable experiment results are written."""

    output_dir: Path = Path("data/results")
    format: Literal["csv", "parquet"] = "csv"
    record_environment: bool = True


class FaceSimilarityThresholds(_Base):
    """Identity decision boundaries; must be calibrated on real data before trust."""

    candidate_threshold: float = Field(default=0.35, ge=-1.0, le=1.0)
    high_confidence_threshold: float = Field(default=0.55, ge=-1.0, le=1.0)
    min_margin: float = Field(default=0.0, ge=0.0, le=2.0)
    low_quality_penalty: float = Field(default=0.0, ge=0.0, le=1.0)
    min_probe_face_pixels: int = Field(default=0, ge=0)
    calibrated: bool = False
    calibration_source: str | None = None


class DeepfakeThresholds(_Base):
    """Synthetic-media score boundaries."""

    suspicious_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    high_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    calibrated: bool = False


class WatermarkThresholds(_Base):
    """Watermark decision boundaries."""

    detection_confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    bit_accuracy_threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class FingerprintThresholds(_Base):
    """Perceptual hash distances and semantic similarity boundary."""

    phash_hamming_threshold: int = Field(default=12, ge=0)
    dhash_hamming_threshold: int = Field(default=12, ge=0)
    semantic_similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    evidence_similarity_threshold: float = Field(default=0.8125, ge=0.0, le=1.0)


class RiskLevels(_Base):
    """Lower bounds, in points, for each qualitative risk level."""

    low: int = 0
    medium: int = 40
    high: int = 70
    critical: int = 85


class RiskWeights(_Base):
    """Weights of the deterministic Phase 10 risk score."""

    face_similarity: float = 0.40
    deepfake_score: float = 0.30
    watermark_confidence: float = 0.15
    fingerprint_similarity: float = 0.10
    provenance_confidence: float = 0.05


class RiskThresholds(_Base):
    """Risk engine configuration."""

    weights: RiskWeights = RiskWeights()
    levels: RiskLevels = RiskLevels()


class Thresholds(_Base):
    """All decision boundaries in one object, loaded from ``thresholds.yaml``."""

    face_similarity: FaceSimilarityThresholds = FaceSimilarityThresholds()
    deepfake: DeepfakeThresholds = DeepfakeThresholds()
    watermark: WatermarkThresholds = WatermarkThresholds()
    fingerprint: FingerprintThresholds = FingerprintThresholds()
    risk: RiskThresholds = RiskThresholds()


class DeepShieldConfig(_Base):
    """Root configuration object handed to every pipeline and component factory."""

    project: ProjectConfig = ProjectConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    logging: LoggingConfig = LoggingConfig()
    face: FaceConfig = FaceConfig()
    enrollment: EnrollmentConfig = EnrollmentConfig()
    detection: DetectionConfig = DetectionConfig()
    protection: ProtectionConfig = ProtectionConfig()
    video: VideoConfig = VideoConfig()
    storage: StorageConfig = StorageConfig()
    api: ApiConfig = ApiConfig()
    experiments: ExperimentsConfig = ExperimentsConfig()
    thresholds: Thresholds = Thresholds()


ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "LOG_LEVEL": ("logging", "level"),
    "LOG_FILE": ("logging", "file"),
    "DATA_DIR": ("runtime", "data_dir"),
    "MODEL_DIR": ("runtime", "model_dir"),
    "DB_URL": ("storage", "database_url"),
    "RANDOM_SEED": ("runtime", "random_seed"),
    "FACE_EMBEDDER": ("face", "embedder", "backend"),
    "DEEPFAKE_DETECTOR": ("detection", "deepfake", "backend"),
    "WATERMARK_BACKEND": ("protection", "watermark", "backend"),
}


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML mapping, raising :class:`ConfigurationError` on any problem."""
    if not path.exists():
        raise ConfigurationError(f"configuration file not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {path}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"{path} must contain a mapping at the top level")
    return loaded


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _set_path(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Assign ``value`` at a nested key path, creating intermediate mappings."""
    node = target
    for key in path[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[path[-1]] = value


def _env_overrides(environ: dict[str, str]) -> dict[str, Any]:
    """Translate ``DEEPSHIELD_*`` variables into a nested override mapping."""
    overrides: dict[str, Any] = {}
    for suffix, path in ENV_OVERRIDES.items():
        raw = environ.get(f"{ENV_PREFIX}{suffix}")
        if raw is None or raw == "":
            continue
        _set_path(overrides, path, raw)
    return overrides


def load_config(
    config_path: Path | str | None = None,
    thresholds_path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
    environ: dict[str, str] | None = None,
) -> DeepShieldConfig:
    """Load, merge and validate the full configuration.

    Args:
        config_path: Main YAML file. Defaults to ``DEEPSHIELD_CONFIG`` or
            ``configs/default.yaml``.
        thresholds_path: Threshold YAML file merged under the ``thresholds`` key.
        overrides: Nested mapping applied after the files, before the environment.
        environ: Environment mapping, injectable for tests.

    Returns:
        A validated, immutable :class:`DeepShieldConfig`.

    Raises:
        ConfigurationError: If a file is missing, malformed, or fails validation.

    """
    env = dict(os.environ if environ is None else environ)
    if config_path is not None:
        main_path = Path(config_path)
    else:
        main_path = Path(env.get(f"{ENV_PREFIX}CONFIG", str(DEFAULT_CONFIG_PATH)))
    thr_path = Path(thresholds_path) if thresholds_path is not None else DEFAULT_THRESHOLDS_PATH

    data = _read_yaml(main_path)
    if thr_path.exists():
        data = deep_merge(data, {"thresholds": _read_yaml(thr_path)})
    if overrides:
        data = deep_merge(data, overrides)
    data = deep_merge(data, _env_overrides(env))

    try:
        return DeepShieldConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid configuration: {exc}") from exc


def default_config() -> DeepShieldConfig:
    """Return a configuration built purely from model defaults, for tests."""
    return DeepShieldConfig()
