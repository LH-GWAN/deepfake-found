"""Repositories: separation of biometrics, round-trips and lineage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepshield.exceptions import IdentityNotFoundError
from deepshield.storage.repository import (
    FileAssetRepository,
    FileEvidenceRepository,
    FileIdentityRepository,
    FileProvenanceStore,
    utc_timestamp,
)
from deepshield.types import (
    AssetFingerprint,
    AssetRecord,
    EvidenceRecord,
    IdentityProfile,
    MediaType,
    ModelInfo,
    ProvenanceRecord,
)

MODEL = ModelInfo(name="m", version="1", backend="mock")


def make_profile(user_id: str = "u1", dimension: int = 16) -> IdentityProfile:
    rng = np.random.default_rng(0)
    references = rng.normal(size=(3, dimension)).astype(np.float32)
    references /= np.linalg.norm(references, axis=1, keepdims=True)
    centroid = references.mean(axis=0)
    return IdentityProfile(
        user_id=user_id,
        reference_embeddings=references,
        centroid_embedding=(centroid / np.linalg.norm(centroid)).astype(np.float32),
        image_count=3,
        model=MODEL,
        embedding_dimension=dimension,
    )


def test_identity_round_trip(tmp_path: Path) -> None:
    repository = FileIdentityRepository(tmp_path)
    profile = make_profile()
    repository.save(profile)
    loaded = repository.get("u1")
    assert loaded is not None
    np.testing.assert_allclose(loaded.reference_embeddings, profile.reference_embeddings)
    assert loaded.model.name == "m"


def test_biometrics_live_in_a_separate_directory(tmp_path: Path) -> None:
    """Embeddings must not sit in the same store as descriptive metadata."""
    repository = FileIdentityRepository(tmp_path)
    repository.save(make_profile())
    metadata = (tmp_path / "identities" / "u1.json").read_text(encoding="utf-8")
    assert (tmp_path / "embeddings" / "u1.npz").is_file()
    assert "reference" not in metadata.replace("embedding_file", "")


def test_missing_identity_returns_none(tmp_path: Path) -> None:
    assert FileIdentityRepository(tmp_path).get("nobody") is None


def test_require_raises_for_unknown_identity(tmp_path: Path) -> None:
    with pytest.raises(IdentityNotFoundError, match="no enrolled identity"):
        FileIdentityRepository(tmp_path).require("nobody")


def test_orphaned_metadata_is_an_explicit_error(tmp_path: Path) -> None:
    repository = FileIdentityRepository(tmp_path)
    repository.save(make_profile())
    (tmp_path / "embeddings" / "u1.npz").unlink()
    with pytest.raises(IdentityNotFoundError, match="embedding file"):
        repository.get("u1")


def test_list_and_delete(tmp_path: Path) -> None:
    repository = FileIdentityRepository(tmp_path)
    repository.save(make_profile("alice"))
    repository.save(make_profile("bob"))
    assert repository.list_users() == ["alice", "bob"]
    assert repository.delete("alice") is True
    assert repository.delete("alice") is False
    assert repository.list_users() == ["bob"]


def test_unsafe_user_ids_cannot_escape_the_directory(tmp_path: Path) -> None:
    repository = FileIdentityRepository(tmp_path)
    repository.save(make_profile("../../etc/passwd"))
    written = list((tmp_path / "identities").glob("*.json"))
    assert len(written) == 1
    assert written[0].parent == tmp_path / "identities"


def test_load_all_returns_every_identity(tmp_path: Path) -> None:
    repository = FileIdentityRepository(tmp_path)
    repository.save(make_profile("alice"))
    repository.save(make_profile("bob"))
    assert {p.user_id for p in repository.load_all()} == {"alice", "bob"}


def test_evidence_round_trip(tmp_path: Path) -> None:
    repository = FileEvidenceRepository(tmp_path)
    analysis_id = repository.save(
        EvidenceRecord(source_id="a.jpg", media_type=MediaType.IMAGE, face_similarity=0.9)
    )
    stored = repository.get(analysis_id)
    assert stored is not None
    assert stored["analysis_id"] == analysis_id
    assert stored["identity"]["similarity"] == 0.9
    assert repository.get("missing") is None
    assert analysis_id in repository.list_ids()


def test_asset_lookup_by_hash_and_watermark(tmp_path: Path) -> None:
    repository = FileAssetRepository(tmp_path)
    record = AssetRecord(
        asset_id="a1",
        user_id="u1",
        fingerprint=AssetFingerprint(asset_id="a1", sha256="ab" * 32, phash="ff", dhash="ee"),
        watermark_code="deadbeef",
        distribution_id="instagram",
    )
    repository.save(record)
    assert repository.get("a1") is not None
    assert repository.find_by_sha256("ab" * 32).asset_id == "a1"
    assert repository.find_by_watermark_code("deadbeef").distribution_id == "instagram"
    assert repository.find_by_sha256("00" * 32) is None
    assert repository.list_assets(user_id="other") == []


def test_provenance_lineage_walks_to_the_root(tmp_path: Path) -> None:
    store = FileProvenanceStore(tmp_path)
    store.record(ProvenanceRecord(asset_id="root", sha256="a" * 64, created_at=utc_timestamp()))
    store.record(
        ProvenanceRecord(
            asset_id="child",
            sha256="b" * 64,
            created_at=utc_timestamp(),
            parent_asset="root",
            watermark_id="wm1",
        )
    )
    chain = store.lineage("child")
    assert [record.asset_id for record in chain] == ["child", "root"]
    assert store.find_by_watermark("wm1").asset_id == "child"
    assert store.find_by_sha256("a" * 64).asset_id == "root"


def test_provenance_cycle_does_not_hang(tmp_path: Path) -> None:
    """A corrupted or forged log must not be able to hang the analysis pipeline."""
    store = FileProvenanceStore(tmp_path)
    store.record(
        ProvenanceRecord(
            asset_id="a", sha256="1" * 64, created_at=utc_timestamp(), parent_asset="b"
        )
    )
    store.record(
        ProvenanceRecord(
            asset_id="b", sha256="2" * 64, created_at=utc_timestamp(), parent_asset="a"
        )
    )
    assert len(store.lineage("a")) == 2


def test_malformed_provenance_lines_are_skipped(tmp_path: Path) -> None:
    store = FileProvenanceStore(tmp_path)
    store.record(ProvenanceRecord(asset_id="a", sha256="1" * 64, created_at=utc_timestamp()))
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n\n")
    assert store.get("a") is not None
