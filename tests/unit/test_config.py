"""Configuration loading, merging and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from deepshield.config import (
    DeepShieldConfig,
    deep_merge,
    default_config,
    load_config,
)
from deepshield.exceptions import ConfigurationError


def test_default_config_is_valid() -> None:
    config = default_config()
    assert config.project.name == "deepshield"
    assert config.face.embedder.embedding_dimension > 0


def test_shipped_yaml_files_load(project_root: Path) -> None:
    config = load_config(
        config_path=project_root / "configs" / "default.yaml",
        thresholds_path=project_root / "configs" / "thresholds.yaml",
        environ={},
    )
    assert config.face.detector.backend == "opencv_yunet"
    assert config.face.embedder.backend == "insightface"
    assert config.protection.watermark.backend == "dct"
    assert -1.0 <= config.thresholds.face_similarity.candidate_threshold <= 1.0


def test_calibrated_thresholds_cite_their_evidence(project_root: Path) -> None:
    """A threshold may only claim to be calibrated if it names the run that fitted it."""
    config = load_config(
        config_path=project_root / "configs" / "default.yaml",
        thresholds_path=project_root / "configs" / "thresholds.yaml",
        environ={},
    )
    face = config.thresholds.face_similarity
    if face.calibrated:
        assert face.calibration_source, "calibrated thresholds must record their source"
        assert face.candidate_threshold < face.high_confidence_threshold


def test_uncalibrated_thresholds_stay_marked(project_root: Path) -> None:
    """Nothing has been fitted for the deepfake detector, and it must say so."""
    config = load_config(
        config_path=project_root / "configs" / "default.yaml",
        thresholds_path=project_root / "configs" / "thresholds.yaml",
        environ={},
    )
    assert config.thresholds.deepfake.calibrated is False


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"b": 1, "c": 2}}
    override = {"a": {"c": 3}, "d": 4}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 1, "c": 3}, "d": 4}
    assert base == {"a": {"b": 1, "c": 2}}


def test_overrides_beat_files(project_root: Path) -> None:
    config = load_config(
        config_path=project_root / "configs" / "default.yaml",
        overrides={"runtime": {"random_seed": 999}},
        environ={},
    )
    assert config.runtime.random_seed == 999


def test_environment_beats_overrides(project_root: Path) -> None:
    config = load_config(
        config_path=project_root / "configs" / "default.yaml",
        overrides={"logging": {"level": "DEBUG"}},
        environ={"DEEPSHIELD_LOG_LEVEL": "ERROR"},
    )
    assert config.logging.level == "ERROR"


def test_missing_file_raises_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(config_path=tmp_path / "missing.yaml", environ={})


def test_malformed_yaml_raises_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("project: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid YAML"):
        load_config(config_path=path, environ={})


def test_non_mapping_yaml_raises_configuration_error(tmp_path: Path) -> None:
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="mapping"):
        load_config(config_path=path, environ={})


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "extra.yaml"
    path.write_text("project:\n  nonexistent_field: 1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid configuration"):
        load_config(config_path=path, environ={})


def test_out_of_range_threshold_is_rejected(tmp_path: Path) -> None:
    main = tmp_path / "default.yaml"
    main.write_text("project:\n  phase: 0\n", encoding="utf-8")
    thresholds = tmp_path / "thresholds.yaml"
    thresholds.write_text("deepfake:\n  suspicious_threshold: 1.7\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(config_path=main, thresholds_path=thresholds, environ={})


def test_config_is_immutable() -> None:
    config = default_config()
    with pytest.raises(ValidationError):
        config.runtime.random_seed = 1


def test_config_round_trips_through_dict() -> None:
    config = default_config()
    restored = DeepShieldConfig.model_validate(config.model_dump())
    assert restored == config
