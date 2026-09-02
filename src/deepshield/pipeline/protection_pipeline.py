"""Protection pipeline: preparing an image for publication.

Runs the outbound half of the system. One image goes in; a protected file, an
asset record and a provenance node come out.

Three independent layers are applied, and their independence is the design:

Watermark
    Embeds an opaque code so a copy found later can be traced to the channel it
    was published on. Survives re-encoding, rescaling and moderate cropping; not
    rotation or heavy downscaling.
Fingerprint
    Records exact and perceptual hashes so near-duplicates can be recognised
    even when the watermark is gone.
Adversarial cloaking
    Optional research layer that perturbs pixels to disturb automated face
    embedding. Off by default.

Each can fail or be stripped on its own. None of them prevents anyone from
generating synthetic content, and none of them survives into a generative
model's output. They make published content traceable, which is a different and
achievable goal.

Publishing the same image to several channels with different distribution ids
produces different watermark codes, so a leak points at a channel.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from deepshield import __version__
from deepshield.config import DeepShieldConfig
from deepshield.exceptions import WatermarkError
from deepshield.logging_utils import get_logger
from deepshield.media import load_image, save_image, sha256_file, validate_rgb
from deepshield.protection.fingerprint import DefaultFingerprinter
from deepshield.protection.watermark import CODE_BITS, Watermarker, build_watermarker
from deepshield.quality import image_quality
from deepshield.storage.repository import (
    FileAssetRepository,
    FileProvenanceStore,
    build_asset_repository,
    build_provenance_store,
    utc_timestamp,
)
from deepshield.types import AssetRecord, ProvenanceRecord, WatermarkPayload

logger = get_logger(__name__)

PROTECTED_SUFFIX = ".png"


class ProtectionPipeline(ABC):
    """Contract for producing a protected asset from a user image."""

    @abstractmethod
    def protect(
        self,
        image_path: Path,
        user_id: str,
        distribution_id: str | None = None,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        """Protect one image and return the resulting asset record."""


class DefaultProtectionPipeline(ProtectionPipeline):
    """Applies watermarking, fingerprinting and provenance registration."""

    def __init__(
        self,
        config: DeepShieldConfig,
        watermarker: Watermarker | None = None,
        asset_repository: FileAssetRepository | None = None,
        provenance_store: FileProvenanceStore | None = None,
    ) -> None:
        """Build or accept the protection components."""
        self.config = config
        self.watermarker = watermarker or build_watermarker(config.protection.watermark)
        self.fingerprinter = DefaultFingerprinter(config.protection.fingerprint)
        self.assets = asset_repository or build_asset_repository(config)
        self.provenance = provenance_store or build_provenance_store(config)

    def _default_output(self, source: Path, asset_id: str) -> Path:
        """Return the default destination inside the configured protected directory."""
        directory = Path(self.config.runtime.data_dir) / "protected"
        return directory / f"{source.stem}_{asset_id}{PROTECTED_SUFFIX}"

    def _cloak(self, image: np.ndarray, user_token: str) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply the optional adversarial layer, reporting what it did.

        Cloaking runs before watermarking so that the watermark is embedded into
        the image the user will actually publish. The reverse order would let the
        perturbation search damage the mark it cannot see.
        """
        settings = self.config.protection.adversarial
        if not settings.enabled:
            return image, {"applied": False, "reason": "disabled in configuration"}

        from deepshield.face.embedder import build_embedder
        from deepshield.protection.adversarial import SpsaAdversarialProtector

        protector = SpsaAdversarialProtector(settings)
        embedder = build_embedder(self.config.face.embedder)
        cloaked = protector.protect(image, [embedder], settings.epsilon)
        return cloaked, {
            "applied": True,
            "epsilon": settings.epsilon,
            "steps": settings.steps,
            "attacked_model": embedder.model_info.name,
            "caveat": (
                "white-box displacement against one model; transferability to an "
                "unknown platform model is not established"
            ),
        }

    def protect(
        self,
        image_path: Path,
        user_id: str,
        distribution_id: str | None = None,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        """Protect one image, register it, and return a full report.

        Raises:
            InvalidMediaError: If the source file cannot be decoded.
            WatermarkError: If the image is too small to carry the watermark.

        """
        source = Path(image_path)
        original = validate_rgb(load_image(source))
        source_digest = sha256_file(source)

        asset_id = uuid.uuid4().hex[:16]
        user_token = uuid.uuid5(uuid.NAMESPACE_URL, f"deepshield:{user_id}").hex[:16]
        payload = WatermarkPayload(
            version=1,
            user_token=user_token,
            asset_id=asset_id,
            distribution_id=distribution_id,
        )

        cloaked, cloak_report = self._cloak(original, user_token)

        try:
            protected = self.watermarker.embed(cloaked, payload)
            watermark_code: str | None = f"{payload.code(CODE_BITS):08x}"
        except WatermarkError as exc:
            logger.warning("watermark skipped for %s: %s", source.name, exc)
            protected = cloaked
            watermark_code = None

        destination = Path(output_path) if output_path else self._default_output(source, asset_id)
        save_image(protected, destination)

        verification = self.watermarker.detect(load_image(destination))
        fingerprint = self.fingerprinter.fingerprint_file(destination, asset_id)
        quality = image_quality(original, protected)

        record = AssetRecord(
            asset_id=asset_id,
            user_id=user_id,
            fingerprint=fingerprint,
            watermark_code=watermark_code,
            distribution_id=distribution_id,
            protected_path=str(destination),
            source_path=str(source),
            protection_version=__version__,
        )
        self.assets.save(record)

        self.provenance.record(
            ProvenanceRecord(
                asset_id=f"{asset_id}:source",
                sha256=source_digest,
                created_at=utc_timestamp(),
                parent_asset=None,
                protection_version=__version__,
                software_version=f"deepshield {__version__}",
            )
        )
        self.provenance.record(
            ProvenanceRecord(
                asset_id=asset_id,
                sha256=fingerprint.sha256,
                created_at=utc_timestamp(),
                parent_asset=f"{asset_id}:source",
                watermark_id=watermark_code,
                protection_version=__version__,
                software_version=f"deepshield {__version__}",
                protection_config_digest=self._config_digest(),
            )
        )

        return {
            "asset_id": asset_id,
            "user_id": user_id,
            "distribution_id": distribution_id,
            "source_path": str(source),
            "protected_path": str(destination),
            "watermark": {
                "backend": self.watermarker.name,
                "code": watermark_code,
                "embedded": watermark_code is not None,
                "verified_after_save": verification.detected,
                "verification_confidence": verification.confidence,
            },
            "fingerprint": fingerprint.to_dict(),
            "adversarial": cloak_report,
            "quality": {
                "psnr": None if quality["psnr"] == float("inf") else round(quality["psnr"], 3),
                "ssim": round(quality["ssim"], 5),
            },
            "limitations": [
                "A watermark identifies this published file, not the identity in it.",
                "The mark survives re-encoding, rescaling, screenshots and moderate "
                "cropping; rotation and heavy downscaling destroy it, and regenerating "
                "the image through another model removes it entirely.",
                "Protection does not prevent anyone from generating synthetic content.",
            ],
        }

    def _config_digest(self) -> str:
        """Return a short digest of the protection settings used, for reproducibility."""
        import hashlib
        import json

        payload = json.dumps(
            self.config.protection.model_dump(mode="json"), sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
