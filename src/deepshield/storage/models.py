"""Persistence schema notes for the Phase 14 SQLAlchemy layer.

The tables are deliberately split so that biometric templates never sit in the
same store as raw media:

``identities``
    user id, model name and version, image count, timestamps
``identity_embeddings``
    reference vectors, stored separately and referenced by identity id
``assets``
    protected assets, their fingerprints and their watermark ids
``analyses``
    one evidence record per analysed input, including detector versions
``provenance``
    lineage edges between assets

Declarative models are added when the API and database land; defining them now
would freeze a schema before the pipelines that use it exist.
"""

from __future__ import annotations

TABLE_NAMES: tuple[str, ...] = (
    "identities",
    "identity_embeddings",
    "assets",
    "analyses",
    "provenance",
)
