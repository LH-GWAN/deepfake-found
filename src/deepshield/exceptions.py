"""Exception hierarchy shared by every DeepShield module.

Pipelines must never crash with an opaque traceback on bad input. Every failure
mode that a caller can reasonably act on gets a dedicated exception type so the
CLI and the REST API can map it to a clear message or status code.
"""

from __future__ import annotations


class DeepShieldError(Exception):
    """Base class for all DeepShield errors."""


class ConfigurationError(DeepShieldError):
    """Raised when configuration files are missing, malformed or inconsistent."""


class InvalidMediaError(DeepShieldError):
    """Raised when an image or video cannot be decoded or has an unsupported format."""


class NoFaceDetectedError(DeepShieldError):
    """Raised when a pipeline step requires at least one face and none was found."""


class ModelNotAvailableError(DeepShieldError):
    """Raised when a backend implementation or its weights are not installed."""


class IdentityNotFoundError(DeepShieldError):
    """Raised when an operation references an unknown user or identity profile."""


class EnrollmentError(DeepShieldError):
    """Raised when identity enrollment cannot produce a usable identity template."""


class WatermarkError(DeepShieldError):
    """Raised when watermark embedding or extraction fails."""


class NotImplementedInPhaseError(DeepShieldError):
    """Raised by interfaces whose concrete implementation belongs to a later phase.

    This exists so that the codebase never pretends a capability is implemented.
    The message states which phase owns the feature.
    """

    def __init__(self, feature: str, phase: str) -> None:
        """Store the missing feature and the phase that will deliver it."""
        self.feature = feature
        self.phase = phase
        super().__init__(f"{feature} is not implemented yet; planned for {phase}")
