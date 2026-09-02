"""File-backed repositories for identities, evidence and provenance.

Pipelines depend on the abstract contracts here, not on any storage engine, so a
database can replace the filesystem later without touching pipeline code.

The identity store deliberately splits into two files per user. Descriptive
metadata - model name, image count, timestamps - lives in one place; the
biometric vectors live in a separate directory that can be given different
permissions, encrypted, or deleted independently. Face embeddings are biometric
identifiers, and keeping them adjacent to ordinary metadata makes it far too
easy for them to leak through a backup or a debug dump.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from deepshield.exceptions import IdentityNotFoundError, ModelNotAvailableError
from deepshield.logging_utils import get_logger
from deepshield.types import (
    AssetFingerprint,
    AssetRecord,
    EvidenceRecord,
    IdentityProfile,
    ModelInfo,
    ProvenanceRecord,
)

logger = get_logger(__name__)

IDENTITY_SUBDIR = "identities"
EMBEDDING_SUBDIR = "embeddings"
ANALYSIS_SUBDIR = "analyses"
PROVENANCE_FILE = "provenance.jsonl"


class IdentityRepository(ABC):
    """Contract for storing and retrieving enrolled identity templates."""

    @abstractmethod
    def save(self, profile: IdentityProfile) -> None:
        """Persist or replace an identity template."""

    @abstractmethod
    def get(self, user_id: str) -> IdentityProfile | None:
        """Return one identity template, or ``None`` when unknown."""

    @abstractmethod
    def list_users(self) -> list[str]:
        """Return the ids of all enrolled users."""

    @abstractmethod
    def delete(self, user_id: str) -> bool:
        """Remove an identity template and report whether it existed."""

    def require(self, user_id: str) -> IdentityProfile:
        """Return an identity template or raise when it is unknown.

        Raises:
            IdentityNotFoundError: If the user has not been enrolled.

        """
        profile = self.get(user_id)
        if profile is None:
            raise IdentityNotFoundError(f"no enrolled identity for user '{user_id}'")
        return profile

    def load_all(self) -> list[IdentityProfile]:
        """Return every enrolled identity template."""
        return [p for p in (self.get(user) for user in self.list_users()) if p is not None]


class EvidenceRepository(ABC):
    """Contract for storing and retrieving analysis evidence records."""

    @abstractmethod
    def save(self, record: EvidenceRecord) -> str:
        """Persist an evidence record and return its analysis id."""

    @abstractmethod
    def get(self, analysis_id: str) -> dict[str, Any] | None:
        """Return one stored evidence report, or ``None`` when unknown."""

    @abstractmethod
    def list_ids(self) -> list[str]:
        """Return the ids of all stored analyses, newest first."""


def _safe_name(value: str) -> str:
    """Return a filesystem-safe version of a user-supplied identifier.

    Raises:
        ValueError: If nothing usable remains after sanitising.

    """
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in value).strip("._")
    if not cleaned:
        raise ValueError(f"identifier '{value}' contains no usable characters")
    return cleaned


class FileIdentityRepository(IdentityRepository):
    """Identity templates on disk, with metadata and biometrics kept apart."""

    def __init__(self, data_dir: Path, embedding_dir: Path | None = None) -> None:
        """Create the metadata and embedding directories."""
        self.metadata_dir = Path(data_dir) / IDENTITY_SUBDIR
        self.embedding_dir = Path(embedding_dir or Path(data_dir) / EMBEDDING_SUBDIR)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, user_id: str) -> tuple[Path, Path]:
        """Return the metadata and embedding paths for a user."""
        name = _safe_name(user_id)
        return self.metadata_dir / f"{name}.json", self.embedding_dir / f"{name}.npz"

    def save(self, profile: IdentityProfile) -> None:
        """Write metadata and embeddings to their separate stores."""
        metadata_path, embedding_path = self._paths(profile.user_id)
        metadata = {
            "user_id": profile.user_id,
            "image_count": profile.image_count,
            "embedding_dimension": profile.embedding_dimension,
            "model": profile.model.to_dict(),
            "created_at": profile.created_at,
            "source_image_ids": list(profile.source_image_ids),
            "embedding_file": embedding_path.name,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        np.savez_compressed(
            embedding_path,
            references=profile.reference_embeddings.astype(np.float32),
            centroid=profile.centroid_embedding.astype(np.float32),
        )
        logger.info("saved identity '%s' (%d references)", profile.user_id, profile.image_count)

    def get(self, user_id: str) -> IdentityProfile | None:
        """Load an identity template, or return ``None`` when it is absent."""
        metadata_path, embedding_path = self._paths(user_id)
        if not metadata_path.is_file():
            return None
        if not embedding_path.is_file():
            raise IdentityNotFoundError(
                f"identity '{user_id}' has metadata but its embedding file "
                f"{embedding_path} is missing"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(embedding_path) as stored:
            references = np.asarray(stored["references"], dtype=np.float32)
            centroid = np.asarray(stored["centroid"], dtype=np.float32)
        return IdentityProfile(
            user_id=metadata["user_id"],
            reference_embeddings=references,
            centroid_embedding=centroid,
            image_count=int(metadata["image_count"]),
            model=ModelInfo(**metadata["model"]),
            embedding_dimension=int(metadata["embedding_dimension"]),
            created_at=metadata["created_at"],
            source_image_ids=list(metadata.get("source_image_ids", [])),
        )

    def list_users(self) -> list[str]:
        """Return every enrolled user id, sorted."""
        users = []
        for path in sorted(self.metadata_dir.glob("*.json")):
            try:
                users.append(json.loads(path.read_text(encoding="utf-8"))["user_id"])
            except (json.JSONDecodeError, KeyError):
                logger.warning("skipping unreadable identity metadata: %s", path)
        return users

    def delete(self, user_id: str) -> bool:
        """Delete both stores for a user and report whether anything existed."""
        metadata_path, embedding_path = self._paths(user_id)
        existed = metadata_path.exists()
        metadata_path.unlink(missing_ok=True)
        embedding_path.unlink(missing_ok=True)
        return existed


class FileEvidenceRepository(EvidenceRepository):
    """Evidence reports stored as one JSON document per analysis."""

    def __init__(self, results_dir: Path) -> None:
        """Create the analyses directory."""
        self.directory = Path(results_dir) / ANALYSIS_SUBDIR
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, record: EvidenceRecord) -> str:
        """Write an evidence report and return its generated analysis id."""
        analysis_id = uuid.uuid4().hex[:16]
        payload = record.to_dict()
        payload["analysis_id"] = analysis_id
        (self.directory / f"{analysis_id}.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        return analysis_id

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        """Return a stored evidence report, or ``None`` when unknown."""
        path = self.directory / f"{_safe_name(analysis_id)}.json"
        if not path.is_file():
            return None
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return payload

    def list_ids(self) -> list[str]:
        """Return stored analysis ids, newest first."""
        paths = sorted(
            self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        return [path.stem for path in paths]


class FileProvenanceStore:
    """Append-only provenance log, one JSON record per line.

    Append-only on purpose: a lineage record that can be edited in place proves
    nothing. This is still a local, self-asserted log - it records what this
    system did to a file, not what happened to that file elsewhere.
    """

    def __init__(self, data_dir: Path) -> None:
        """Create the provenance log directory."""
        self.path = Path(data_dir) / PROVENANCE_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, record: ProvenanceRecord) -> None:
        """Append one provenance node."""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), default=str) + "\n")

    def _load(self) -> dict[str, ProvenanceRecord]:
        """Return every stored node keyed by asset id, latest write winning."""
        if not self.path.is_file():
            return {}
        records: dict[str, ProvenanceRecord] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records[payload["asset_id"]] = ProvenanceRecord(**payload)
            except (json.JSONDecodeError, KeyError, TypeError):
                logger.warning("skipping malformed provenance line")
        return records

    def get(self, asset_id: str) -> ProvenanceRecord | None:
        """Return one provenance node by asset id."""
        return self._load().get(asset_id)

    def lineage(self, asset_id: str) -> list[ProvenanceRecord]:
        """Return the chain from an asset back to its root ancestor.

        Cycles are broken rather than followed, because a corrupted or forged
        log must not hang the analysis pipeline.
        """
        records = self._load()
        chain: list[ProvenanceRecord] = []
        seen: set[str] = set()
        current = records.get(asset_id)
        while current is not None and current.asset_id not in seen:
            seen.add(current.asset_id)
            chain.append(current)
            current = records.get(current.parent_asset) if current.parent_asset else None
        return chain

    def find_by_sha256(self, digest: str) -> ProvenanceRecord | None:
        """Return the node whose file hash matches, if this system produced it."""
        for record in self._load().values():
            if record.sha256 == digest:
                return record
        return None

    def find_by_watermark(self, watermark_id: str) -> ProvenanceRecord | None:
        """Return the node issued with a given watermark id."""
        for record in self._load().values():
            if record.watermark_id == watermark_id:
                return record
        return None


def utc_timestamp() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def build_identity_repository(config: Any) -> IdentityRepository:
    """Create the identity repository described by configuration."""
    storage = config.storage
    if not storage.database_url.startswith("sqlite"):
        raise ModelNotAvailableError(
            f"only the file-backed store is implemented; got {storage.database_url}"
        )
    embedding_dir = storage.embedding_store_dir if storage.separate_embedding_store else None
    return FileIdentityRepository(config.runtime.data_dir, embedding_dir)


def build_evidence_repository(config: Any) -> EvidenceRepository:
    """Create the evidence repository described by configuration."""
    return FileEvidenceRepository(config.runtime.results_dir)


def build_provenance_store(config: Any) -> FileProvenanceStore:
    """Create the provenance store described by configuration."""
    return FileProvenanceStore(config.runtime.data_dir)


ASSET_SUBDIR = "assets"


class FileAssetRepository:
    """Registry of protected assets, one JSON document per asset.

    The analysis side queries this to answer "is this suspect file derived from
    something the user published, and through which channel". Lookups by exact
    hash, by watermark code and by perceptual hash answer progressively weaker
    versions of that question.
    """

    def __init__(self, data_dir: Path) -> None:
        """Create the asset directory."""
        self.directory = Path(data_dir) / ASSET_SUBDIR
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, record: AssetRecord) -> None:
        """Persist or replace one asset record."""
        path = self.directory / f"{_safe_name(record.asset_id)}.json"
        path.write_text(json.dumps(record.to_dict(), indent=2, default=str), encoding="utf-8")

    def _load_one(self, path: Path) -> AssetRecord | None:
        """Parse one asset document, skipping unreadable files."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            fingerprint = payload["fingerprint"]
            return AssetRecord(
                asset_id=payload["asset_id"],
                user_id=payload["user_id"],
                fingerprint=AssetFingerprint(
                    asset_id=fingerprint["asset_id"],
                    sha256=fingerprint["sha256"],
                    phash=fingerprint["phash"],
                    dhash=fingerprint["dhash"],
                    created_at=fingerprint.get("created_at", payload["created_at"]),
                ),
                watermark_code=payload.get("watermark_code"),
                distribution_id=payload.get("distribution_id"),
                protected_path=payload.get("protected_path"),
                source_path=payload.get("source_path"),
                protection_version=payload.get("protection_version"),
                created_at=payload["created_at"],
            )
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("skipping unreadable asset record: %s", path)
            return None

    def list_assets(self, user_id: str | None = None) -> list[AssetRecord]:
        """Return every registered asset, optionally filtered by user."""
        loaded = (self._load_one(path) for path in sorted(self.directory.glob("*.json")))
        records = [record for record in loaded if record is not None]
        if user_id is None:
            return records
        return [record for record in records if record.user_id == user_id]

    def get(self, asset_id: str) -> AssetRecord | None:
        """Return one asset record by id."""
        path = self.directory / f"{_safe_name(asset_id)}.json"
        return self._load_one(path) if path.is_file() else None

    def find_by_sha256(self, digest: str) -> AssetRecord | None:
        """Return the asset whose exact bytes match, if any."""
        for record in self.list_assets():
            if record.fingerprint.sha256 == digest:
                return record
        return None

    def find_by_watermark_code(self, code: str) -> AssetRecord | None:
        """Return the asset issued with a given watermark code."""
        for record in self.list_assets():
            if record.watermark_code and record.watermark_code == code:
                return record
        return None


def build_asset_repository(config: Any) -> FileAssetRepository:
    """Create the asset repository described by configuration."""
    return FileAssetRepository(config.runtime.data_dir)
