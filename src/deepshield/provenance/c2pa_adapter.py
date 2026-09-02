"""C2PA content credentials adapter interface.

C2PA is the industry standard for cryptographically signed content credentials.
A full implementation needs a signing identity, a trust list and a manifest
store, all of which sit outside the MVP. The adapter is defined here so that
provenance code targets the standard's shape from the start; it remains a
declared stub until a later phase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from deepshield.exceptions import NotImplementedInPhaseError


class C2PAAdapter(ABC):
    """Contract for reading and writing C2PA manifests."""

    @abstractmethod
    def read_manifest(self, path: Path) -> dict[str, Any] | None:
        """Return the C2PA manifest attached to a file, or ``None``."""

    @abstractmethod
    def attach_manifest(self, path: Path, manifest: dict[str, Any]) -> Path:
        """Write a signed manifest into a copy of the file and return its path."""

    @abstractmethod
    def verify(self, path: Path) -> dict[str, Any]:
        """Verify manifest signatures and return the validation outcome."""


class StubC2PAAdapter(C2PAAdapter):
    """Declared, deliberately unimplemented C2PA adapter.

    A working implementation needs a signing identity, a trust list and a
    manifest store, none of which exist inside this project. Rather than emit a
    manifest that looks authoritative and verifies against nothing, every method
    reports that the capability is absent. The local provenance log remains the
    honest alternative, with its narrower claim stated.
    """

    name = "stub"

    def read_manifest(self, path: Path) -> dict[str, Any] | None:
        """Report that manifest reading is not implemented."""
        raise NotImplementedInPhaseError("C2PA manifest reading", "a post-MVP phase")

    def attach_manifest(self, path: Path, manifest: dict[str, Any]) -> Path:
        """Report that manifest signing is not implemented."""
        raise NotImplementedInPhaseError("C2PA manifest signing", "a post-MVP phase")

    def verify(self, path: Path) -> dict[str, Any]:
        """Return an explicit 'not verified' result rather than a false negative.

        A caller must be able to tell "this file has no valid credential" apart
        from "this system cannot check credentials at all".
        """
        return {
            "supported": False,
            "verified": None,
            "reason": (
                "C2PA verification is not implemented; the local provenance log covers "
                "only what this system did to a file, not third-party attestations"
            ),
        }
