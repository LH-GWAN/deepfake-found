"""Model weight resolution and download.

Weights are never committed to the repository: they are large, they carry their
own licences, and pinning them by URL plus SHA-256 is what makes an experiment
reproducible. This module resolves a logical model name to a local file,
downloading it once into the configured model directory and verifying its digest
before any component is allowed to load it.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from deepshield.exceptions import ModelNotAvailableError
from deepshield.logging_utils import get_logger

logger = get_logger(__name__)

DOWNLOAD_CHUNK_BYTES = 1 << 16
DOWNLOAD_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class ModelAsset:
    """One downloadable model file, pinned by URL and digest."""

    key: str
    filename: str
    url: str
    sha256: str | None
    description: str
    archive_member: str | None = None

    def local_path(self, model_dir: Path) -> Path:
        """Return where this asset lives once downloaded."""
        return Path(model_dir) / self.filename


MODEL_ASSETS: dict[str, ModelAsset] = {
    "yunet": ModelAsset(
        key="yunet",
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        description="YuNet face detector, used by the opencv_yunet backend",
    ),
    "sface": ModelAsset(
        key="sface",
        filename="face_recognition_sface_2021dec.onnx",
        url=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
            "face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        description="SFace recognition embedder, used by the opencv_sface backend",
    ),
}


def sha256_of(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download_asset(asset: ModelAsset, model_dir: Path, force: bool = False) -> Path:
    """Download one model asset into ``model_dir`` and verify its digest.

    Args:
        asset: The asset to fetch.
        model_dir: Destination directory, created if missing.
        force: Re-download even when the file is already present.

    Returns:
        The local path of the verified file.

    Raises:
        ModelNotAvailableError: If the download fails or the digest mismatches.

    """
    destination = asset.local_path(model_dir)
    if destination.exists() and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    logger.info("downloading model asset %s from %s", asset.key, asset.url)
    try:
        with urllib.request.urlopen(asset.url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        temporary.unlink(missing_ok=True)
        raise ModelNotAvailableError(
            f"could not download model '{asset.key}' from {asset.url}: {exc}"
        ) from exc

    if asset.archive_member is not None:
        extracted = temporary.with_suffix(".extracted")
        try:
            with zipfile.ZipFile(temporary) as archive:
                with archive.open(asset.archive_member) as source, extracted.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
        except (zipfile.BadZipFile, KeyError) as exc:
            temporary.unlink(missing_ok=True)
            raise ModelNotAvailableError(
                f"model archive for '{asset.key}' did not contain {asset.archive_member}"
            ) from exc
        temporary.unlink(missing_ok=True)
        temporary = extracted

    if asset.sha256 is not None:
        actual = sha256_of(temporary)
        if actual != asset.sha256:
            temporary.unlink(missing_ok=True)
            raise ModelNotAvailableError(
                f"checksum mismatch for model '{asset.key}': "
                f"expected {asset.sha256}, got {actual}"
            )

    temporary.replace(destination)
    logger.info("model asset %s ready at %s", asset.key, destination)
    return destination


def resolve_model(
    key: str,
    model_dir: Path,
    explicit_path: Path | str | None = None,
    allow_download: bool = True,
) -> Path:
    """Return a usable local path for a model, downloading it when permitted.

    Args:
        key: Logical model name from :data:`MODEL_ASSETS`.
        model_dir: Directory holding downloaded weights.
        explicit_path: A path supplied in configuration, which always wins.
        allow_download: Whether fetching over the network is permitted.

    Raises:
        ModelNotAvailableError: If no usable file can be produced.

    """
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.is_file():
            raise ModelNotAvailableError(f"configured model file not found: {path}")
        return path

    asset = MODEL_ASSETS.get(key)
    if asset is None:
        available = ", ".join(sorted(MODEL_ASSETS))
        raise ModelNotAvailableError(f"unknown model '{key}'; available: {available}")

    destination = asset.local_path(Path(model_dir))
    if destination.is_file():
        return destination
    if not allow_download:
        raise ModelNotAvailableError(
            f"model '{key}' is not present at {destination} and downloading is disabled"
        )
    return download_asset(asset, Path(model_dir))


def available_models(model_dir: Path) -> dict[str, dict[str, object]]:
    """Return the download status and description of every known model asset."""
    return {
        key: {
            "description": asset.description,
            "filename": asset.filename,
            "present": asset.local_path(Path(model_dir)).is_file(),
        }
        for key, asset in MODEL_ASSETS.items()
    }
