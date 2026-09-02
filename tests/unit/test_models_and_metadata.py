"""Model resolution, checksum enforcement and metadata extraction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from deepshield.exceptions import InvalidMediaError, ModelNotAvailableError
from deepshield.media import save_image
from deepshield.models import (
    MODEL_ASSETS,
    ModelAsset,
    available_models,
    download_asset,
    resolve_model,
    sha256_of,
)
from deepshield.provenance.metadata import ImageMetadataExtractor


def local_asset(tmp_path: Path, payload: bytes, digest: str | None) -> ModelAsset:
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    return ModelAsset(
        key="test",
        filename="test.bin",
        url=source.as_uri(),
        sha256=digest,
        description="local test asset",
    )


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"deepshield")
    assert sha256_of(path) == hashlib.sha256(b"deepshield").hexdigest()


def test_download_verifies_the_checksum(tmp_path: Path) -> None:
    payload = b"weights"
    asset = local_asset(tmp_path, payload, hashlib.sha256(payload).hexdigest())
    path = download_asset(asset, tmp_path / "models")
    assert path.read_bytes() == payload


def test_checksum_mismatch_is_rejected_and_leaves_no_file(tmp_path: Path) -> None:
    """A corrupted or substituted model must never reach a component."""
    asset = local_asset(tmp_path, b"weights", "00" * 32)
    with pytest.raises(ModelNotAvailableError, match="checksum mismatch"):
        download_asset(asset, tmp_path / "models")
    assert not (tmp_path / "models" / "test.bin").exists()


def test_existing_file_is_not_redownloaded(tmp_path: Path) -> None:
    payload = b"weights"
    asset = local_asset(tmp_path, payload, hashlib.sha256(payload).hexdigest())
    first = download_asset(asset, tmp_path / "models")
    (tmp_path / "source.bin").unlink()
    assert download_asset(asset, tmp_path / "models") == first


def test_unreachable_url_reports_the_model_name(tmp_path: Path) -> None:
    asset = ModelAsset(
        key="ghost",
        filename="ghost.bin",
        url=(tmp_path / "missing.bin").as_uri(),
        sha256=None,
        description="missing",
    )
    with pytest.raises(ModelNotAvailableError, match="ghost"):
        download_asset(asset, tmp_path / "models")


def test_explicit_path_wins_over_download(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.onnx"
    explicit.write_bytes(b"x")
    assert resolve_model("yunet", tmp_path, explicit_path=explicit) == explicit


def test_missing_explicit_path_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelNotAvailableError, match="configured model file not found"):
        resolve_model("yunet", tmp_path, explicit_path=tmp_path / "nope.onnx")


def test_unknown_model_lists_alternatives(tmp_path: Path) -> None:
    with pytest.raises(ModelNotAvailableError, match="available"):
        resolve_model("no_such_model", tmp_path)


def test_download_can_be_disabled(tmp_path: Path) -> None:
    with pytest.raises(ModelNotAvailableError, match="downloading is disabled"):
        resolve_model("yunet", tmp_path / "empty", allow_download=False)


def test_available_models_reports_presence(tmp_path: Path) -> None:
    status = available_models(tmp_path)
    assert set(status) == set(MODEL_ASSETS)
    assert all(info["present"] is False for info in status.values())


def test_every_shipped_asset_is_checksum_pinned() -> None:
    """An unpinned model URL means the code cannot tell what it just executed."""
    for key, asset in MODEL_ASSETS.items():
        assert asset.sha256, f"model '{key}' has no pinned checksum"
        assert len(asset.sha256) == 64


def test_metadata_extraction_returns_container_fields(tmp_path: Path, photo) -> None:
    path = save_image(photo, tmp_path / "a.png")
    metadata = ImageMetadataExtractor().extract(path)
    assert metadata["container"]["format"] == "PNG"
    assert metadata["container"]["width"] == photo.shape[1]


def test_metadata_states_that_absence_proves_nothing(tmp_path: Path, photo) -> None:
    path = save_image(photo, tmp_path / "a.png")
    metadata = ImageMetadataExtractor().extract(path)
    assert metadata["exif_present"] is False
    assert "forged" in metadata["interpretation"]


def test_metadata_rejects_missing_and_corrupt_files(tmp_path: Path) -> None:
    with pytest.raises(InvalidMediaError, match="not found"):
        ImageMetadataExtractor().extract(tmp_path / "missing.png")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"nope")
    with pytest.raises(InvalidMediaError):
        ImageMetadataExtractor().extract(broken)


def test_c2pa_stub_reports_absence_not_a_false_negative() -> None:
    """A caller must distinguish 'no valid credential' from 'cannot check'."""
    from deepshield.provenance import StubC2PAAdapter

    result = StubC2PAAdapter().verify(Path("anything"))
    assert result["supported"] is False
    assert result["verified"] is None
