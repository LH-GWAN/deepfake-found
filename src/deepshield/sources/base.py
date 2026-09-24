"""Content sources: where suspect media comes from before it is analysed.

The analysis pipelines take a file. A source is what turns "look here" into
files: a folder on disk, a list of URLs, or pages to crawl. Separating the two
keeps the analysis order in one place and lets a new source, a platform API or
a reverse image search, plug in without touching it.

Every source answers two questions. ``discover`` lists the media items a query
leads to without downloading them, so a caller can count, filter and bound the
work first. ``fetch`` produces one item's bytes as a local file, with the hash
of exactly what was received, because provenance and evidence are defined over
those bytes and a URL can serve different bytes tomorrow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from deepshield.types import utc_now


class ContentKind(StrEnum):
    """The kind of media an item holds, which decides the pipeline that reads it."""

    IMAGE = "image"
    VIDEO = "video"


@dataclass(frozen=True)
class ContentItem:
    """One piece of media a source discovered, not yet fetched."""

    uri: str
    kind: ContentKind
    source: str
    found_on: str | None = None
    discovered_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping."""
        return asdict(self)


@dataclass(frozen=True)
class FetchedContent:
    """An item's bytes on local disk, with what was learned while fetching them."""

    item: ContentItem
    path: Path
    sha256: str
    size_bytes: int
    final_uri: str
    content_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping without the local path."""
        return {
            "uri": self.item.uri,
            "final_uri": self.final_uri,
            "kind": self.item.kind.value,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
        }


class ContentSource(ABC):
    """Contract for anything that can supply media to the analysis pipelines."""

    name: str = "abstract"

    @abstractmethod
    def discover(self, query: str) -> Iterator[ContentItem]:
        """Yield the media items ``query`` leads to, without downloading them.

        Raises:
            SourceError: If the query itself cannot be read.
            BlockedSourceError: If policy forbids reading it.

        """

    @abstractmethod
    def fetch(self, item: ContentItem, directory: Path) -> FetchedContent:
        """Produce one item as a local file under ``directory``.

        Raises:
            SourceError: If the item cannot be retrieved.
            BlockedSourceError: If policy forbids retrieving it.

        """
