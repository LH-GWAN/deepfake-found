"""DeepShield: personal deepfake identity protection and multi-signal detection system."""

from deepshield.exceptions import (
    ConfigurationError,
    DeepShieldError,
    InvalidMediaError,
    ModelNotAvailableError,
    NoFaceDetectedError,
    NotImplementedInPhaseError,
)

__version__ = "0.1.0"
__all__ = [
    "ConfigurationError",
    "DeepShieldError",
    "InvalidMediaError",
    "ModelNotAvailableError",
    "NoFaceDetectedError",
    "NotImplementedInPhaseError",
    "__version__",
]
