"""Detection adapters: synthetic media, watermark extraction and manipulation."""

from deepshield.detection import blending as _blending
from deepshield.detection import deepfake_backends as _deepfake_backends
from deepshield.detection.blending import BlendingArtifactDetector, extract_features
from deepshield.detection.deepfake import (
    DEEPFAKE_REGISTRY,
    DeepfakeDetector,
    MockDeepfakeDetector,
    build_deepfake_detector,
)
from deepshield.detection.deepfake_backends import (
    OnnxDeepfakeDetector,
    SpectralArtifactDetector,
    aggregate_frame_scores,
)
from deepshield.detection.manipulation import ManipulationDetector
from deepshield.detection.watermark_detector import WatermarkDetector

__all__ = [
    "BlendingArtifactDetector",
    "_blending",
    "extract_features",
    "OnnxDeepfakeDetector",
    "SpectralArtifactDetector",
    "_deepfake_backends",
    "aggregate_frame_scores",
    "DEEPFAKE_REGISTRY",
    "DeepfakeDetector",
    "ManipulationDetector",
    "MockDeepfakeDetector",
    "WatermarkDetector",
    "build_deepfake_detector",
]
