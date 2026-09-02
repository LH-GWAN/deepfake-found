"""Identity enrollment: turning reference photos into one identity template.

Plain language: take a handful of photos of the user and boil them down to the
numeric description of their face that everything else compares against.

Pipeline: detect, align, embed, filter out weak crops, L2-normalise, then keep
both the individual reference embeddings and their centroid.

Both representations are kept deliberately. The centroid is compact but averages
away the pose and lighting variation that made collecting several photos worth
doing; keeping the references lets the matcher use max or top-k aggregation,
which usually recovers that variation.

Quality filtering matters more here than anywhere else in the system. A blurred
or badly posed enrollment photo does not merely fail to help - it pulls the
centroid toward a bad region of the embedding space and depresses every future
comparison. Rejected images are reported with a reason rather than dropped
silently.

Enrollment refuses to produce a template from a single usable image: one photo
gives no evidence about which variation is identity and which is lighting, and
the resulting threshold behaviour is not trustworthy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from deepshield.config import DeepShieldConfig
from deepshield.exceptions import EnrollmentError, InvalidMediaError
from deepshield.face.aligner import FaceAligner, build_aligner
from deepshield.face.detector import FaceDetector, build_detector
from deepshield.face.embedder import FaceEmbedder, build_embedder
from deepshield.logging_utils import get_logger
from deepshield.media import IMAGE_SUFFIXES, load_image
from deepshield.quality import laplacian_variance
from deepshield.types import IdentityProfile

logger = get_logger(__name__)


@dataclass
class EnrollmentImageReport:
    """Why one candidate image was accepted or rejected."""

    path: str
    accepted: bool
    reason: str = "ok"
    detection_confidence: float | None = None
    face_pixels: int | None = None
    sharpness: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return {
            "path": self.path,
            "accepted": self.accepted,
            "reason": self.reason,
            "detection_confidence": self.detection_confidence,
            "face_pixels": self.face_pixels,
            "sharpness": self.sharpness,
        }


@dataclass
class EnrollmentResult:
    """An identity template plus the audit trail that produced it."""

    profile: IdentityProfile
    reports: list[EnrollmentImageReport] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        """Number of images that survived quality filtering."""
        return sum(1 for report in self.reports if report.accepted)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping without raw biometric vectors."""
        return {
            "profile": self.profile.to_dict(),
            "accepted": self.accepted_count,
            "considered": len(self.reports),
            "images": [report.to_dict() for report in self.reports],
        }


class IdentityEnroller(ABC):
    """Contract for building an identity template from reference images."""

    @abstractmethod
    def enroll(self, user_id: str, image_paths: list[Path]) -> EnrollmentResult:
        """Build an identity template for ``user_id`` from reference images."""


class DefaultIdentityEnroller(IdentityEnroller):
    """Detector, aligner and embedder wired together with quality filtering."""

    def __init__(
        self,
        config: DeepShieldConfig,
        detector: FaceDetector | None = None,
        aligner: FaceAligner | None = None,
        embedder: FaceEmbedder | None = None,
    ) -> None:
        """Build or accept the three face-pipeline stages."""
        self.config = config
        self.detector = detector or build_detector(config.face.detector)
        self.aligner = aligner or build_aligner(config.face.aligner)
        self.embedder = embedder or build_embedder(config.face.embedder)

    @staticmethod
    def collect_images(directory: Path) -> list[Path]:
        """Return the supported image files directly inside a directory, sorted.

        Raises:
            EnrollmentError: If the directory is missing or holds no images.

        """
        folder = Path(directory)
        if not folder.is_dir():
            raise EnrollmentError(f"enrollment directory not found: {folder}")
        images = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise EnrollmentError(f"no supported images found in {folder}")
        return images

    def _analyse(self, path: Path) -> tuple[EnrollmentImageReport, np.ndarray | None]:
        """Detect, align and embed one candidate image, reporting its quality."""
        report = EnrollmentImageReport(path=str(path), accepted=False)
        try:
            image = load_image(path)
        except InvalidMediaError as exc:
            report.reason = f"unreadable: {exc}"
            return report, None

        faces = self.detector.detect(image)
        if not faces:
            report.reason = "no face detected"
            return report, None
        if len(faces) > 1:
            logger.debug("%s has %d faces, using the most confident", path.name, len(faces))

        face = faces[0]
        quality = self.config.enrollment.quality
        report.detection_confidence = face.detection_confidence
        report.face_pixels = int(min(face.bbox.width, face.bbox.height))

        if face.detection_confidence < quality.min_detection_confidence:
            report.reason = (
                f"detection confidence {face.detection_confidence:.2f} below "
                f"{quality.min_detection_confidence:.2f}"
            )
            return report, None
        if report.face_pixels < quality.min_face_pixels:
            report.reason = (
                f"face is {report.face_pixels}px, below {quality.min_face_pixels}px"
            )
            return report, None

        aligned = self.aligner.align(image, face)
        report.sharpness = laplacian_variance(aligned.image)
        if report.sharpness < quality.min_sharpness:
            report.reason = (
                f"sharpness {report.sharpness:.0f} below absolute floor "
                f"{quality.min_sharpness:.0f}"
            )
            return report, None

        embedding = self.embedder.embed(aligned.image)
        report.accepted = True
        return report, embedding.vector

    def enroll(self, user_id: str, image_paths: list[Path]) -> EnrollmentResult:
        """Build an identity template from reference images.

        Raises:
            EnrollmentError: If too few images survive quality filtering.

        """
        if not user_id:
            raise EnrollmentError("user_id must not be empty")

        settings = self.config.enrollment
        candidates = list(image_paths)[: settings.max_images]
        if not candidates:
            raise EnrollmentError("no enrollment images supplied")

        reports: list[EnrollmentImageReport] = []
        vectors: list[np.ndarray] = []
        for path in candidates:
            report, vector = self._analyse(path)
            reports.append(report)
            if vector is not None:
                vectors.append(vector)

        if vectors:
            sharpness = np.array(
                [r.sharpness or 0.0 for r in reports if r.accepted], dtype=np.float64
            )
            floor = float(np.median(sharpness)) * settings.quality.min_sharpness_ratio
            kept: list[np.ndarray] = []
            index = 0
            for report in reports:
                if not report.accepted:
                    continue
                if (report.sharpness or 0.0) < floor:
                    report.accepted = False
                    report.reason = (
                        f"sharpness {report.sharpness:.0f} far below batch median "
                        f"floor {floor:.0f}"
                    )
                else:
                    kept.append(vectors[index])
                index += 1
            vectors = kept

        if len(vectors) < settings.min_images:
            rejected = [f"{r.path}: {r.reason}" for r in reports if not r.accepted]
            raise EnrollmentError(
                f"only {len(vectors)} of {len(reports)} images passed quality filtering, "
                f"but at least {settings.min_images} are required. "
                f"Rejected: {'; '.join(rejected) if rejected else 'none'}"
            )

        stacked = np.vstack([vector.astype(np.float32) for vector in vectors])
        norms = np.linalg.norm(stacked, axis=1, keepdims=True)
        matrix = (stacked / np.maximum(norms, 1e-12)).astype(np.float32)

        mean_vector = matrix.mean(axis=0)
        centroid = (
            mean_vector / max(float(np.linalg.norm(mean_vector)), 1e-12)
        ).astype(np.float32)

        profile = IdentityProfile(
            user_id=user_id,
            reference_embeddings=matrix if settings.keep_reference_embeddings else matrix[:1],
            centroid_embedding=centroid,
            image_count=int(matrix.shape[0]),
            model=self.embedder.model_info,
            embedding_dimension=int(matrix.shape[1]),
            source_image_ids=[r.path for r in reports if r.accepted],
        )
        logger.info(
            "enrolled '%s' from %d of %d images with %s",
            user_id,
            matrix.shape[0],
            len(reports),
            self.embedder.model_info.name,
        )
        return EnrollmentResult(profile=profile, reports=reports)

    def enroll_directory(self, user_id: str, directory: Path) -> EnrollmentResult:
        """Enroll every supported image in a directory."""
        return self.enroll(user_id, self.collect_images(directory))
