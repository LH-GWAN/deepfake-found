"""Face pipeline: detection, alignment, embedding, matching and enrollment."""

from deepshield.face import backends as _backends
from deepshield.face.aligner import ALIGNER_REGISTRY, FaceAligner, SimpleCropAligner
from deepshield.face.detector import DETECTOR_REGISTRY, FaceDetector, MockFaceDetector
from deepshield.face.embedder import (
    EMBEDDER_REGISTRY,
    EnsembleEmbedder,
    FaceEmbedder,
    FlipTtaEmbedder,
    MockFaceEmbedder,
)
from deepshield.face.enrollment import DefaultIdentityEnroller, EnrollmentResult, IdentityEnroller
from deepshield.face.matcher import FaceMatcher, NumpyFaceMatcher, build_matcher

__all__ = [
    "_backends",
    "ALIGNER_REGISTRY",
    "DETECTOR_REGISTRY",
    "EMBEDDER_REGISTRY",
    "FaceAligner",
    "FaceDetector",
    "EnsembleEmbedder",
    "FaceEmbedder",
    "FlipTtaEmbedder",
    "FaceMatcher",
    "NumpyFaceMatcher",
    "build_matcher",
    "DefaultIdentityEnroller",
    "EnrollmentResult",
    "IdentityEnroller",
    "MockFaceDetector",
    "MockFaceEmbedder",
    "SimpleCropAligner",
]
