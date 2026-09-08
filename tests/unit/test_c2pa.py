"""C2PA verification: real credentials verify, tampered bytes do not, none is none.

The fixture is ``C.jpg`` from the c2pa-python test suite: a JPEG signed with the
C2PA test certificate, which no trust list recognises. That makes it the right
fixture, because "valid signature, unknown signer" is what most credentials
this system will meet look like, and the adapter has to report those two facts
separately.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from deepshield.exceptions import NotImplementedInPhaseError
from deepshield.provenance.c2pa_adapter import (
    C2paPythonAdapter,
    StubC2PAAdapter,
    build_c2pa_adapter,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "c2pa" / "C.jpg"

c2pa = pytest.importorskip("c2pa")


@pytest.fixture
def adapter() -> C2paPythonAdapter:
    return C2paPythonAdapter()


def test_signed_fixture_verifies_but_is_not_trusted(adapter: C2paPythonAdapter) -> None:
    result = adapter.verify(FIXTURE)
    assert result["supported"] is True
    assert result["present"] is True
    assert result["verified"] is True
    assert result["trusted"] is False
    assert result["issuer"] == "C2PA Test Signing Cert"
    assert result["claim_generator"]
    assert any(issue["code"] == "signingCredential.untrusted" for issue in result["issues"])


def test_manifest_store_is_returned_as_a_mapping(adapter: C2paPythonAdapter) -> None:
    store = adapter.read_manifest(FIXTURE)
    assert store is not None
    assert store["active_manifest"] in store["manifests"]


def test_tampered_bytes_fail_verification(adapter: C2paPythonAdapter, tmp_path: Path) -> None:
    """Changing image bytes after signing must break the content hash."""
    tampered = tmp_path / "tampered.jpg"
    data = bytearray(FIXTURE.read_bytes())
    tail = len(data) - 2048
    for offset in range(tail, tail + 64):
        data[offset] ^= 0x55
    tampered.write_bytes(bytes(data))
    result = adapter.verify(tampered)
    assert result["present"] is True
    assert result["verified"] is False
    assert result["issues"]


def test_plain_jpeg_has_no_credentials(adapter: C2paPythonAdapter, tmp_path: Path) -> None:
    from PIL import Image

    plain = tmp_path / "plain.jpg"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(plain, quality=90)
    result = adapter.verify(plain)
    assert result["supported"] is True
    assert result["present"] is False
    assert result["verified"] is None
    assert adapter.read_manifest(plain) is None


def test_signing_is_refused_without_an_identity(adapter: C2paPythonAdapter, tmp_path: Path) -> None:
    copy = tmp_path / "copy.jpg"
    shutil.copy(FIXTURE, copy)
    with pytest.raises(NotImplementedInPhaseError):
        adapter.attach_manifest(copy, {})


def test_auto_backend_prefers_the_bindings() -> None:
    assert isinstance(build_c2pa_adapter("auto"), C2paPythonAdapter)
    assert isinstance(build_c2pa_adapter("stub"), StubC2PAAdapter)


def test_stub_still_reports_absence_not_a_false_negative() -> None:
    result = StubC2PAAdapter().verify(FIXTURE)
    assert result["supported"] is False
    assert result["verified"] is None
