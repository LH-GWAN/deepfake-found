"""CLI surface, exit codes and honesty about unimplemented phases."""

from __future__ import annotations

import json

import pytest

from deepshield.cli import (
    EXIT_ERROR,
    EXIT_NOT_IMPLEMENTED,
    EXIT_OK,
    EXIT_USAGE,
    build_parser,
    main,
)


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == EXIT_USAGE
    assert "usage:" in capsys.readouterr().out


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "deepshield" in capsys.readouterr().out


def test_info_reports_backends_and_calibration_state(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["info"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "selected backends" in out
    assert "registered backends" in out
    assert "calibrated" in out


def test_info_json_is_machine_readable(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json", "info"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert "mock" in payload["registered_backends"]["face_embedder"]
    assert payload["thresholds_calibrated"]["deepfake"] is False


def test_doctor_lists_optional_dependencies(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json", "doctor"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert "insightface" in payload["optional_dependencies"]


@pytest.mark.parametrize(
    ("argv", "pattern"),
    [
        (["enroll", "nowhere/", "--user-id", "u1"], "not found"),
        (["protect", "nowhere.jpg", "--user-id", "u1"], "not found"),
        (["analyze-image", "nowhere.jpg"], "not found"),
        (["watermark-detect", "nowhere.jpg"], "not found"),
        (["fingerprint", "nowhere.jpg"], "not found"),
        (["report", "does-not-exist"], "no stored analysis"),
    ],
)
def test_implemented_commands_fail_cleanly_on_missing_input(
    argv: list[str], pattern: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Implemented commands must report a clear cause, never a traceback."""
    assert main(argv) == EXIT_ERROR
    assert pattern in capsys.readouterr().err


@pytest.mark.parametrize(
    ("module", "attribute", "argv"),
    [
        (
            "deepshield.face.enrollment",
            "DefaultIdentityEnroller",
            ["enroll", "nowhere/", "--user-id", "u1"],
        ),
        (
            "deepshield.pipeline.analysis_pipeline",
            "DefaultAnalysisPipeline",
            ["analyze-image", "nowhere.jpg"],
        ),
    ],
)
def test_missing_input_is_reported_before_any_model_is_built(
    module: str,
    attribute: str,
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A mistyped path must not load or download a model to find that out."""
    import importlib

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a model backend was built before the input path was checked")

    monkeypatch.setattr(importlib.import_module(module), attribute, refuse)
    assert main(argv) == EXIT_ERROR
    assert "not found" in capsys.readouterr().err


def test_no_command_is_reported_as_not_implemented() -> None:
    """The dispatcher still has an explicit branch for an unhandled command name."""
    from deepshield.cli import command_not_implemented

    assert command_not_implemented("imaginary") == EXIT_NOT_IMPLEMENTED


def test_unknown_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["does-not-exist"])
    assert excinfo.value.code == 2


def test_missing_config_file_reports_error(tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(tmp_path / "nope.yaml"), "info"]) == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_parser_exposes_every_documented_command() -> None:
    from deepshield.cli import IMPLEMENTED_COMMANDS

    parser = build_parser()
    actions = [a for a in parser._subparsers._actions if hasattr(a, "choices") and a.choices]
    commands = set(actions[-1].choices)
    assert set(IMPLEMENTED_COMMANDS) <= commands


def test_info_reports_calibration_provenance(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json", "info"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    face = payload["face_thresholds"]
    if payload["thresholds_calibrated"]["face_similarity"]:
        assert face["source"]
    assert face["candidate"] <= face["high_confidence"]


def test_doctor_reports_model_presence(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json", "doctor"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert "yunet" in payload["models"]
    assert set(payload["models"]["yunet"]) >= {"present", "description", "filename"}


def test_description_states_the_attribution_limit() -> None:
    assert "training data" in build_parser().description
