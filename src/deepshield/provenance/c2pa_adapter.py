"""C2PA content credentials adapter: interface, a reader-verifier and the stub.

C2PA is the industry standard for cryptographically signed content credentials.
Two halves of it matter here and they are not symmetric.

Reading and verifying is something this system can do for any file a user
submits: parse the manifest store, check that the signature over the claim is
valid, check that the content hashes still match the bytes, and report who
signed it and when. That is :class:`C2paPythonAdapter`, built on the official
``c2pa-python`` bindings to the Rust SDK. What it cannot decide alone is
whether the signer should be believed - that needs a trust list, and none is
bundled - so the result separates *verified* (the signature and hashes hold)
from *trusted* (the signing certificate chains to a configured anchor).

Signing is not something this system can do honestly. A manifest that verifies
against a certificate nobody else recognises is not a credential, it is a
claim with a lock drawn on it. Until a deployment provides a signing identity,
``attach_manifest`` reports that it is unimplemented, and the local provenance
log remains the narrower, honest record of what this system did to a file.

:class:`StubC2PAAdapter` stays for installs without the optional dependency;
its ``verify`` says so explicitly rather than returning a false negative.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from deepshield.exceptions import NotImplementedInPhaseError
from deepshield.logging_utils import get_logger

logger = get_logger(__name__)

TRUSTED_STATES = frozenset({"Trusted"})
VERIFIED_STATES = frozenset({"Valid", "Trusted"})


class C2PAAdapter(ABC):
    """Contract for reading and writing C2PA manifests."""

    name: str = "abstract"

    @abstractmethod
    def read_manifest(self, path: Path) -> dict[str, Any] | None:
        """Return the C2PA manifest store attached to a file, or ``None``."""

    @abstractmethod
    def attach_manifest(self, path: Path, manifest: dict[str, Any]) -> Path:
        """Write a signed manifest into a copy of the file and return its path."""

    @abstractmethod
    def verify(self, path: Path) -> dict[str, Any]:
        """Verify manifest signatures and return the validation outcome."""


class StubC2PAAdapter(C2PAAdapter):
    """Declared, deliberately unimplemented C2PA adapter.

    Used when ``c2pa-python`` is not installed. Rather than emit a manifest
    that looks authoritative and verifies against nothing, every method reports
    that the capability is absent. The local provenance log remains the honest
    alternative, with its narrower claim stated.
    """

    name = "stub"

    def read_manifest(self, path: Path) -> dict[str, Any] | None:
        """Report that manifest reading is not implemented."""
        raise NotImplementedInPhaseError("C2PA manifest reading", "the 'provenance' extra")

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
            "present": None,
            "verified": None,
            "trusted": None,
            "reason": (
                "C2PA verification needs the 'provenance' extra (c2pa-python); the local "
                "provenance log covers only what this system did to a file, not "
                "third-party attestations"
            ),
        }


class C2paPythonAdapter(C2PAAdapter):
    """Read and verify Content Credentials with the ``c2pa-python`` bindings.

    Verification is what the Rust SDK reports: signature over the claim,
    content hashes against the bytes, timestamps, and certificate validity.
    Trust is reported separately, because without a configured trust list the
    SDK marks every signer untrusted, and "valid but from an unknown signer" is
    the truthful description of most credentials this system will meet.
    """

    name = "c2pa_python"

    def __init__(self) -> None:
        """Import the bindings, failing with the install hint when absent."""
        try:
            import c2pa
        except ImportError as exc:
            raise NotImplementedInPhaseError(
                "C2PA verification", "the 'provenance' extra (pip install -e '.[provenance]')"
            ) from exc
        self._c2pa = c2pa

    def _reader(self, path: Path) -> Any | None:
        """Open a reader over the file, or ``None`` when it carries no manifest."""
        try:
            return self._c2pa.Reader(str(path))
        except self._c2pa.C2paError as exc:
            message = str(exc)
            if "ManifestNotFound" in message or "no JUMBF" in message or "NotFound" in message:
                return None
            raise

    def read_manifest(self, path: Path) -> dict[str, Any] | None:
        """Return the manifest store as a mapping, or ``None`` without one."""
        reader = self._reader(Path(path))
        if reader is None:
            return None
        store: dict[str, Any] = json.loads(reader.json())
        return store

    def attach_manifest(self, path: Path, manifest: dict[str, Any]) -> Path:
        """Report that signing is not implemented: there is no signing identity."""
        raise NotImplementedInPhaseError(
            "C2PA manifest signing",
            "a deployment that provides a signing certificate and key; a manifest signed "
            "by a certificate nobody recognises would not be a credential",
        )

    def verify(self, path: Path) -> dict[str, Any]:
        """Return signature, hash and trust outcomes for the file's credentials."""
        try:
            reader = self._reader(Path(path))
        except self._c2pa.C2paError as exc:
            return {
                "supported": True,
                "present": None,
                "verified": None,
                "trusted": None,
                "reason": f"the C2PA reader could not parse the file: {exc}",
            }
        if reader is None:
            return {
                "supported": True,
                "present": False,
                "verified": None,
                "trusted": None,
                "reason": "the file carries no C2PA manifest",
            }

        store = json.loads(reader.json())
        state = str(reader.get_validation_state())
        active_label = store.get("active_manifest")
        active = store.get("manifests", {}).get(active_label, {}) if active_label else {}
        signature = active.get("signature_info") or {}
        issues = [
            {"code": item.get("code"), "explanation": item.get("explanation")}
            for item in store.get("validation_status") or []
        ]
        verified = state in VERIFIED_STATES
        trusted = state in TRUSTED_STATES
        if trusted:
            reason = "signature and content hashes verified; signer is on the trust list"
        elif verified:
            reason = (
                "signature and content hashes verified; the signing certificate is not on "
                "a configured trust list, so the signer's identity is asserted, not vouched for"
            )
        else:
            reason = "the manifest did not validate: " + "; ".join(
                str(item["code"]) for item in issues
            )
        return {
            "supported": True,
            "present": True,
            "verified": verified,
            "trusted": trusted,
            "validation_state": state,
            "active_manifest": active_label,
            "claim_generator": active.get("claim_generator"),
            "title": active.get("title"),
            "issuer": signature.get("issuer"),
            "signed_at": signature.get("time"),
            "algorithm": signature.get("alg"),
            "issues": issues,
            "reason": reason,
        }


def build_c2pa_adapter(backend: str = "auto") -> C2PAAdapter:
    """Return the adapter named by ``backend``.

    ``auto`` uses the bindings when they are installed and the stub otherwise,
    which is safe because the stub's ``verify`` says that it cannot check
    rather than that nothing was found.
    """
    if backend in ("auto", "c2pa_python"):
        try:
            return C2paPythonAdapter()
        except NotImplementedInPhaseError:
            if backend == "c2pa_python":
                raise
            logger.debug("c2pa-python not installed; using the stub C2PA adapter")
    return StubC2PAAdapter()
