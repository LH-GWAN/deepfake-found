"""FastAPI application exposing the protection and analysis pipelines.

The API is a thin transport over the same pipelines the CLI drives, so there is
one implementation of the analysis order and one set of caveats. Handlers do
three things: validate the upload, call a pipeline, and return the evidence
record unchanged.

Uploads are written to a temporary file rather than analysed from memory,
because file hashing and provenance matching are defined over the exact bytes on
disk. Temporary files are removed even when a handler raises.

FastAPI is an optional dependency. It is imported at module scope inside a
``try`` so that this module still imports without it, while the names it defines
stay resolvable: the handlers below are annotated, and with postponed annotation
evaluation FastAPI can only resolve those annotations against module globals.
Importing ``deepshield`` never reaches this module, so a minimal install is
unaffected.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from deepshield import __version__
from deepshield.config import DeepShieldConfig, load_config
from deepshield.exceptions import (
    DeepShieldError,
    EnrollmentError,
    IdentityNotFoundError,
    InvalidMediaError,
    ModelNotAvailableError,
)
from deepshield.logging_utils import get_logger

logger = get_logger(__name__)

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

DEFAULT_LIMITATIONS = [
    "Face similarity does not prove that an image was used as training data.",
    "Deepfake detectors may fail on generator families absent from their training data.",
    "Absence of a watermark is inconclusive.",
]


def create_app(config: DeepShieldConfig | None = None) -> Any:
    """Build the FastAPI application.

    Raises:
        ModelNotAvailableError: If the ``api`` extra is not installed.

    """
    if not FASTAPI_AVAILABLE:
        raise ModelNotAvailableError(
            "FastAPI is not installed; install the 'api' extra: pip install -e '.[api]'"
        )

    settings = config or load_config()
    max_upload_bytes = settings.api.max_upload_mb * 1024 * 1024

    app = FastAPI(
        title="DeepShield",
        version=__version__,
        description=(
            "Personal deepfake identity protection and multi-signal detection. "
            "This API reports identity similarity and synthetic-media likelihood. "
            "It cannot prove that an image was used as training data for a "
            "generative model, and it never claims to."
        ),
    )

    def _save_upload(upload: UploadFile, directory: Path) -> Path:
        """Stream an upload to disk, enforcing the configured size limit."""
        name = Path(upload.filename or "upload").name
        destination = directory / name
        size = 0
        with destination.open("wb") as handle:
            while chunk := upload.file.read(1 << 20):
                size += len(chunk)
                if size > max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"upload exceeds {settings.api.max_upload_mb} MB",
                    )
                handle.write(chunk)
        return destination

    @app.exception_handler(DeepShieldError)
    async def _domain_error(_: Any, exc: DeepShieldError) -> JSONResponse:
        """Map domain errors onto meaningful status codes."""
        status = {
            IdentityNotFoundError: 404,
            InvalidMediaError: 415,
            EnrollmentError: 422,
            ModelNotAvailableError: 503,
        }.get(type(exc), 400)
        return JSONResponse(status_code=status, content={"error": str(exc)})

    @app.get("/health", summary="Liveness and component availability")
    def health() -> dict[str, Any]:
        """Report the service version, selected backends and calibration status."""
        return {
            "status": "ok",
            "version": __version__,
            "backends": {
                "face_detector": settings.face.detector.backend,
                "face_embedder": settings.face.embedder.backend,
                "deepfake_detector": settings.detection.deepfake.backend,
                "watermark": settings.protection.watermark.backend,
            },
            "thresholds_calibrated": {
                "face_similarity": settings.thresholds.face_similarity.calibrated,
                "deepfake": settings.thresholds.deepfake.calibrated,
            },
        }

    @app.post("/identity/enroll", summary="Enroll a user identity from reference images")
    def enroll(
        user_id: str = Form(...), files: list[UploadFile] = File(...)
    ) -> dict[str, Any]:
        """Enroll an identity from several uploaded reference photographs."""
        from deepshield.face.enrollment import DefaultIdentityEnroller
        from deepshield.storage import build_identity_repository

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            paths = [_save_upload(upload, directory) for upload in files]
            result = DefaultIdentityEnroller(settings).enroll(user_id, paths)
            build_identity_repository(settings).save(result.profile)
            payload = result.to_dict()
        payload["note"] = (
            "Face embeddings are biometric identifiers and are stored separately "
            "from image files."
        )
        return payload

    @app.get("/identity", summary="List enrolled identities")
    def identities() -> dict[str, Any]:
        """Return every enrolled identity template, without biometric vectors."""
        from deepshield.storage import build_identity_repository

        return {
            "identities": [p.to_dict() for p in build_identity_repository(settings).load_all()]
        }

    @app.post("/protect/image", summary="Apply the protection pipeline to one image")
    def protect(
        user_id: str = Form(...),
        distribution_id: str | None = Form(None),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        """Watermark, fingerprint and register one uploaded image."""
        from deepshield.pipeline.protection_pipeline import DefaultProtectionPipeline

        with tempfile.TemporaryDirectory() as raw:
            path = _save_upload(file, Path(raw))
            return DefaultProtectionPipeline(settings).protect(
                path, user_id, distribution_id
            )

    @app.post("/analyze/image", summary="Analyse one suspect image")
    def analyze_image(
        file: UploadFile = File(...), user_id: str | None = Form(None)
    ) -> dict[str, Any]:
        """Run the gated analysis pipeline over one uploaded image."""
        from deepshield.pipeline.analysis_pipeline import DefaultAnalysisPipeline
        from deepshield.storage import build_evidence_repository

        with tempfile.TemporaryDirectory() as raw:
            path = _save_upload(file, Path(raw))
            record = DefaultAnalysisPipeline(settings).analyze_image(path, user_id)
            record.analysis_id = build_evidence_repository(settings).save(record)
            return record.to_dict()

    @app.post("/analyze/video", summary="Analyse one suspect video")
    def analyze_video(
        file: UploadFile = File(...), user_id: str | None = Form(None)
    ) -> dict[str, Any]:
        """Sample, track and analyse one uploaded video."""
        from deepshield.pipeline.analysis_pipeline import DefaultAnalysisPipeline
        from deepshield.storage import build_evidence_repository

        with tempfile.TemporaryDirectory() as raw:
            path = _save_upload(file, Path(raw))
            record = DefaultAnalysisPipeline(settings).analyze_video(path, user_id)
            record.analysis_id = build_evidence_repository(settings).save(record)
            return record.to_dict()

    @app.post("/watermark/detect", summary="Attempt watermark extraction from one image")
    def watermark_detect(file: UploadFile = File(...)) -> dict[str, Any]:
        """Try to recover a watermark code and resolve it to a registered asset."""
        from deepshield.media import load_image
        from deepshield.protection.watermark import build_watermarker
        from deepshield.storage import build_asset_repository

        with tempfile.TemporaryDirectory() as raw:
            path = _save_upload(file, Path(raw))
            result = build_watermarker(settings.protection.watermark).detect(load_image(path))

        payload = result.to_dict()
        asset = (
            build_asset_repository(settings).find_by_watermark_code(result.watermark_code)
            if result.watermark_code
            else None
        )
        payload["matched_asset"] = asset.to_dict() if asset else None
        payload["interpretation"] = (
            "A positive detection is strong attribution evidence. A negative result is "
            "inconclusive: cropping, rescaling and regeneration all remove the mark."
        )
        return payload

    @app.get("/analysis/{analysis_id}", summary="Fetch a stored evidence record")
    def get_analysis(analysis_id: str) -> dict[str, Any]:
        """Return a previously stored evidence record."""
        from deepshield.storage import build_evidence_repository

        payload = build_evidence_repository(settings).get(analysis_id)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"unknown analysis '{analysis_id}'")
        return payload

    @app.get("/limitations", summary="What this system does not establish")
    def limitations() -> dict[str, Any]:
        """State the boundaries of every result this API returns."""
        return {
            "cannot_prove": (
                "that a particular image was used as training data for a generative model"
            ),
            "limitations": DEFAULT_LIMITATIONS,
        }

    logger.info("DeepShield API created (version %s)", __version__)
    return app



