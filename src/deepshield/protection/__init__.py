"""Protection layers applied to a user's images before publication."""

from deepshield.protection.adversarial import AdversarialProtector
from deepshield.protection.fingerprint import Fingerprinter
from deepshield.protection.watermark import (
    WATERMARK_REGISTRY,
    MockWatermarker,
    Watermarker,
    build_watermarker,
)

__all__ = [
    "WATERMARK_REGISTRY",
    "AdversarialProtector",
    "Fingerprinter",
    "MockWatermarker",
    "Watermarker",
    "build_watermarker",
]
