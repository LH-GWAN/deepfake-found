"""Persistence layer for identities, assets, evidence and provenance."""

from deepshield.storage.repository import (
    EvidenceRepository,
    FileAssetRepository,
    FileEvidenceRepository,
    FileIdentityRepository,
    FileProvenanceStore,
    IdentityRepository,
    build_asset_repository,
    build_evidence_repository,
    build_identity_repository,
    build_provenance_store,
)

__all__ = [
    "EvidenceRepository",
    "FileAssetRepository",
    "FileEvidenceRepository",
    "FileIdentityRepository",
    "FileProvenanceStore",
    "IdentityRepository",
    "build_asset_repository",
    "build_evidence_repository",
    "build_identity_repository",
    "build_provenance_store",
]
