"""Media loading, saving and hashing helpers.

Every pipeline entry point funnels through this module so that decoding errors,
unsupported formats and colour-space mistakes are handled in exactly one place.
Images are RGB ``uint8`` arrays everywhere inside DeepShield; conversion from
whatever the file contained happens here and nowhere else.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from deepshield.exceptions import InvalidMediaError

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
HASH_CHUNK_BYTES = 1 << 20


def is_image_path(path: Path) -> bool:
    """Return whether a path looks like a supported still image."""
    return path.suffix.lower() in IMAGE_SUFFIXES


def is_video_path(path: Path) -> bool:
    """Return whether a path looks like a supported video container."""
    return path.suffix.lower() in VIDEO_SUFFIXES


def load_image(path: Path | str) -> np.ndarray:
    """Decode an image file into an RGB ``uint8`` array.

    Args:
        path: File to read.

    Returns:
        ``H x W x 3`` RGB array.

    Raises:
        InvalidMediaError: If the file is missing, not an image, or corrupt.

    """
    file_path = Path(path)
    if not file_path.exists():
        raise InvalidMediaError(f"image not found: {file_path}")
    if not file_path.is_file():
        raise InvalidMediaError(f"not a file: {file_path}")
    try:
        with Image.open(file_path) as handle:
            decoded: np.ndarray = np.asarray(handle.convert("RGB"), dtype=np.uint8)
            return decoded
    except UnidentifiedImageError as exc:
        raise InvalidMediaError(f"unsupported or corrupt image: {file_path}") from exc
    except OSError as exc:
        raise InvalidMediaError(f"could not read image {file_path}: {exc}") from exc


def save_image(image: np.ndarray, path: Path | str, quality: int = 95) -> Path:
    """Write an RGB ``uint8`` array to disk, creating parent directories.

    Args:
        image: ``H x W x 3`` RGB array.
        path: Destination file; the suffix selects the format.
        quality: JPEG or WebP quality when the format supports it.

    Returns:
        The path written.

    Raises:
        InvalidMediaError: If the array is not a writable RGB image.

    """
    array = validate_rgb(image)
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image = Image.fromarray(array)
    suffix = file_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        pil_image.save(file_path, quality=quality, subsampling=0)
    elif suffix == ".webp":
        pil_image.save(file_path, quality=quality)
    else:
        pil_image.save(file_path)
    return file_path


def validate_rgb(image: np.ndarray, name: str = "image") -> np.ndarray:
    """Check that an array is an RGB ``uint8`` image and return it as ``uint8``.

    Raises:
        InvalidMediaError: If the array has the wrong rank, channels or is empty.

    """
    if not isinstance(image, np.ndarray):
        raise InvalidMediaError(f"{name} must be a numpy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise InvalidMediaError(f"{name} must be H x W x 3 RGB, got shape {image.shape}")
    if image.size == 0:
        raise InvalidMediaError(f"{name} is empty")
    if image.dtype != np.uint8:
        clipped: np.ndarray = np.clip(image, 0, 255).astype(np.uint8)
        return clipped
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to a float32 luminance plane using ITU-R BT.601."""
    array = validate_rgb(image).astype(np.float32)
    luminance: np.ndarray = (
        array[:, :, 0] * 0.299 + array[:, :, 1] * 0.587 + array[:, :, 2] * 0.114
    )
    return luminance


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 of a file's exact bytes, streamed in chunks."""
    file_path = Path(path)
    if not file_path.is_file():
        raise InvalidMediaError(f"file not found: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(image: np.ndarray) -> str:
    """Return the SHA-256 of an array's raw pixel bytes.

    This differs from :func:`sha256_file`: it identifies decoded pixels, so it is
    stable across container rewrites that do not touch pixel values, and it is
    the right hash for an image that only exists in memory.
    """
    array = validate_rgb(image)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()
