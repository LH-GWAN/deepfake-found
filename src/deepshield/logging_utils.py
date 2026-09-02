"""Logging setup with biometric redaction.

Face embeddings are biometric identifiers. Writing a full 512-dimensional vector
into a log file effectively copies the biometric template into an unprotected
store, so every helper here truncates vectors to a short preview plus a
non-reversible digest.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from deepshield.config import LoggingConfig

_CONFIGURED = False


def embedding_digest(vector: Sequence[float]) -> str:
    """Return a short, non-reversible digest identifying an embedding."""
    payload = ",".join(f"{float(value):.6f}" for value in vector)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def safe_embedding_repr(
    vector: Sequence[float] | None,
    preview_dims: int = 4,
    redact: bool = True,
) -> str:
    """Format an embedding for logs without exposing the full biometric template.

    Args:
        vector: Embedding values, or ``None``.
        preview_dims: How many leading dimensions to show.
        redact: When ``False``, the full vector is formatted; only use this in
            offline debugging, never in code paths that write to disk.

    Returns:
        A string such as ``dim=512 sha=1a2b3c4d5e6f head=[0.0123, -0.0456]``.

    """
    if vector is None:
        return "None"
    values = [float(value) for value in vector]
    if not redact:
        return repr(values)
    head = ", ".join(f"{value:.4f}" for value in values[:preview_dims])
    suffix = ", ..." if len(values) > preview_dims else ""
    return f"dim={len(values)} sha={embedding_digest(values)} head=[{head}{suffix}]"


class EmbeddingRedactionFilter(logging.Filter):
    """Drop log records that were explicitly flagged as carrying raw biometrics.

    A caller signals such a record with ``extra={"contains_raw_embedding": True}``.
    The filter suppresses it whenever redaction is enabled, which makes the unsafe
    path impossible to reach by accident in a configured deployment.
    """

    def __init__(self, redact: bool = True) -> None:
        """Store whether redaction is enforced."""
        super().__init__(name="deepshield.redaction")
        self.redact = redact

    def filter(self, record: logging.LogRecord) -> bool:
        """Return ``False`` for flagged records while redaction is enabled."""
        if not self.redact:
            return True
        return not getattr(record, "contains_raw_embedding", False)


def configure_logging(config: LoggingConfig | None = None, force: bool = False) -> None:
    """Install DeepShield's root logging handlers.

    Args:
        config: Logging configuration; model defaults are used when omitted.
        force: Reconfigure even if logging was already set up in this process.

    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    settings = config or LoggingConfig()
    root = logging.getLogger("deepshield")
    root.handlers.clear()
    root.setLevel(settings.level.upper())
    root.propagate = False

    formatter = logging.Formatter(settings.format)
    redaction = EmbeddingRedactionFilter(settings.redact_embeddings)

    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setFormatter(formatter)
    stream.addFilter(redaction)
    root.addHandler(stream)

    if settings.file is not None:
        path = Path(settings.file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redaction)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger of the ``deepshield`` root logger."""
    if name.startswith("deepshield"):
        return logging.getLogger(name)
    return logging.getLogger(f"deepshield.{name}")
