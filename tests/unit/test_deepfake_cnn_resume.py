"""A detector checkpoint resumes only into the run that wrote it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_deepfake_cnn as trainer  # noqa: E402


def arguments(manifest: Path, **overrides: object) -> argparse.Namespace:
    settings = {
        "manifest": [manifest], "train": ["inswapper", "graphics"], "seed": 42, "epochs": 12,
        "holdout": 0.3, "batch": 32, "lr": 1e-4, "limit": None,
    }
    settings.update(overrides)
    return argparse.Namespace(**settings)


def test_the_same_run_resumes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"records": []}', encoding="utf-8")
    saved = {"run": trainer.run_signature(arguments(manifest))}
    # The order families are named in does not change what is trained.
    again = trainer.run_signature(arguments(manifest, train=["graphics", "inswapper"]))
    assert trainer.resume_mismatches(saved, again) == []


@pytest.mark.parametrize(
    "change", [{"seed": 7}, {"epochs": 20}, {"train": ["inswapper"]}, {"holdout": 0.2}]
)
def test_a_different_run_refuses_the_checkpoint(tmp_path: Path, change: dict[str, object]) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"records": []}', encoding="utf-8")
    saved = {"run": trainer.run_signature(arguments(manifest))}
    changed = trainer.run_signature(arguments(manifest, **change))
    mismatches = trainer.resume_mismatches(saved, changed)
    assert [line.split(":")[0] for line in mismatches] == list(change)


def test_a_rebuilt_manifest_at_the_same_path_refuses_the_checkpoint(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"records": []}', encoding="utf-8")
    saved = {"run": trainer.run_signature(arguments(manifest))}
    manifest.write_text('{"records": [{"path": "a.png"}]}', encoding="utf-8")
    assert trainer.resume_mismatches(saved, trainer.run_signature(arguments(manifest))) != []


def test_a_checkpoint_without_recorded_settings_is_not_trusted(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"records": []}', encoding="utf-8")
    assert trainer.resume_mismatches({"epoch": 3}, trainer.run_signature(arguments(manifest)))
