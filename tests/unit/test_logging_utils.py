"""Logging setup and biometric redaction."""

from __future__ import annotations

import logging

import numpy as np

from deepshield.config import LoggingConfig
from deepshield.logging_utils import (
    EmbeddingRedactionFilter,
    configure_logging,
    embedding_digest,
    get_logger,
    safe_embedding_repr,
)


def test_embedding_digest_is_stable_and_short() -> None:
    vector = [0.1, 0.2, 0.3]
    assert embedding_digest(vector) == embedding_digest(list(vector))
    assert len(embedding_digest(vector)) == 12


def test_embedding_digest_changes_with_content() -> None:
    assert embedding_digest([0.1, 0.2]) != embedding_digest([0.1, 0.3])


def test_safe_repr_hides_full_vector() -> None:
    vector = np.arange(512, dtype=np.float32) / 512.0
    rendered = safe_embedding_repr(vector, preview_dims=4)
    assert "dim=512" in rendered
    assert rendered.count(",") <= 5
    assert "0.9" not in rendered


def test_safe_repr_handles_none() -> None:
    assert safe_embedding_repr(None) == "None"


def test_redaction_filter_drops_flagged_records() -> None:
    filt = EmbeddingRedactionFilter(redact=True)
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "msg", None, None)
    record.contains_raw_embedding = True
    assert filt.filter(record) is False


def test_redaction_filter_keeps_normal_records() -> None:
    filt = EmbeddingRedactionFilter(redact=True)
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "msg", None, None)
    assert filt.filter(record) is True


def test_configure_logging_writes_to_file(tmp_path) -> None:
    log_file = tmp_path / "logs" / "deepshield.log"
    configure_logging(LoggingConfig(level="DEBUG", file=log_file), force=True)
    logger = get_logger("test_module")
    logger.info("hello")
    logger.info("secret", extra={"contains_raw_embedding": True})
    for handler in logging.getLogger("deepshield").handlers:
        handler.flush()
    content = log_file.read_text(encoding="utf-8")
    assert "hello" in content
    assert "secret" not in content


def test_get_logger_namespaces_children() -> None:
    assert get_logger("face.embedder").name == "deepshield.face.embedder"
    assert get_logger("deepshield.cli").name == "deepshield.cli"
