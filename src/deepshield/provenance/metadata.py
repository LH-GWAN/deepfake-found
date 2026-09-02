"""Container and EXIF metadata extraction.

Metadata is weak evidence in both directions: every major platform strips it on
upload, and anything that survives can be forged with a text editor. It is
collected as context - a camera model or a creation date can corroborate a
story - and never as proof. Absence of metadata says nothing at all.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image, UnidentifiedImageError

from deepshield.exceptions import InvalidMediaError
from deepshield.logging_utils import get_logger

logger = get_logger(__name__)

INTERESTING_TAGS = frozenset(
    {
        "Make",
        "Model",
        "Software",
        "DateTime",
        "DateTimeOriginal",
        "DateTimeDigitized",
        "Artist",
        "Copyright",
        "ImageDescription",
        "Orientation",
        "ExifImageWidth",
        "ExifImageHeight",
    }
)


class MetadataExtractor(ABC):
    """Contract for reading embedded file metadata."""

    @abstractmethod
    def extract(self, path: Path) -> dict[str, Any]:
        """Return metadata found in a media file, or an empty mapping."""


class ImageMetadataExtractor(MetadataExtractor):
    """Reads EXIF and container metadata from still images."""

    def extract(self, path: Path) -> dict[str, Any]:
        """Return container and EXIF fields, plus a note on what absence means.

        Raises:
            InvalidMediaError: If the file cannot be opened as an image.

        """
        file_path = Path(path)
        if not file_path.is_file():
            raise InvalidMediaError(f"file not found: {file_path}")

        try:
            with Image.open(file_path) as handle:
                container = {
                    "format": handle.format,
                    "mode": handle.mode,
                    "width": handle.width,
                    "height": handle.height,
                }
                raw = handle.getexif()
        except UnidentifiedImageError as exc:
            raise InvalidMediaError(f"unsupported or corrupt image: {file_path}") from exc

        exif: dict[str, Any] = {}
        for tag_id, value in dict(raw).items():
            name = ExifTags.TAGS.get(tag_id, str(tag_id))
            if name not in INTERESTING_TAGS:
                continue
            exif[name] = value.decode("utf-8", "replace") if isinstance(value, bytes) else value

        return {
            "container": container,
            "exif": exif,
            "exif_present": bool(exif),
            "interpretation": (
                "Metadata is trivially stripped by platforms and trivially forged by an "
                "attacker; treat it as context, never as proof of origin."
            ),
        }
