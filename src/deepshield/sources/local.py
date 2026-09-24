"""A folder on disk as a content source."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from deepshield.exceptions import SourceError
from deepshield.media import IMAGE_SUFFIXES, VIDEO_SUFFIXES, sha256_file
from deepshield.sources.base import ContentItem, ContentKind, ContentSource, FetchedContent


class FolderSource(ContentSource):
    """Every image and video under a directory, recursively.

    Files are read in place rather than copied: they are already local, and the
    evidence must hash the file the user pointed at.
    """

    name = "folder"

    def discover(self, query: str) -> Iterator[ContentItem]:
        """Yield every supported image and video under the directory ``query``.

        Raises:
            SourceError: If ``query`` is not a directory.

        """
        root = Path(query)
        if not root.is_dir():
            raise SourceError(f"not a directory: {root}")
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            suffix = path.suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                kind = ContentKind.IMAGE
            elif suffix in VIDEO_SUFFIXES:
                kind = ContentKind.VIDEO
            else:
                continue
            yield ContentItem(uri=str(path), kind=kind, source=self.name, found_on=str(root))

    def fetch(self, item: ContentItem, directory: Path) -> FetchedContent:
        """Return the file itself; ``directory`` is unused for local files.

        Raises:
            SourceError: If the file has disappeared since discovery.

        """
        path = Path(item.uri)
        if not path.is_file():
            raise SourceError(f"file no longer exists: {path}")
        return FetchedContent(
            item=item,
            path=path,
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
            final_uri=str(path),
        )
