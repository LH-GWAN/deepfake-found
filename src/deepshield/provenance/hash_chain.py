"""Asset lineage over an append-only provenance log.

Each protection step writes a node recording the resulting file's hash and its
parent, giving a chain such as original to protected to compressed to uploaded.
When a suspect file turns up, the chain answers which of the user's own
publications it descends from.

The scope of that claim is narrow and worth stating plainly: this is a local,
self-asserted log. It records what this system did to a file. It says nothing
about what happened to that file elsewhere, and it is not a cryptographic proof
to a third party - that is what the C2PA adapter is for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from deepshield.storage.repository import FileProvenanceStore
from deepshield.types import ProvenanceRecord


class ProvenanceStore(ABC):
    """Contract for recording and querying asset lineage."""

    @abstractmethod
    def record(self, record: ProvenanceRecord) -> None:
        """Persist one provenance node."""

    @abstractmethod
    def get(self, asset_id: str) -> ProvenanceRecord | None:
        """Return one node by asset id, or ``None`` when unknown."""

    @abstractmethod
    def lineage(self, asset_id: str) -> list[ProvenanceRecord]:
        """Return the chain from an asset back to its root ancestor."""


ProvenanceStore.register(FileProvenanceStore)

__all__ = ["FileProvenanceStore", "ProvenanceStore"]
