"""Tabulating the layered-watermark benchmark: when two marks contradict each other."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_layered_watermark as layered  # noqa: E402

CODES = 72


def peaked(at: int | None) -> list[float]:
    values = [0.1] * CODES
    if at is not None:
        values[at] = 0.9
    return values


def reading(index: int, config: str, attack: str, dct: int | None, learned: int | None,
            decoder: str = "v3_39db") -> dict[str, Any]:
    row: dict[str, Any] = {
        "index": index, "truth": 5, "victim": 6, "config": config, "attack": attack,
        "psnr_vs_original": None, "psnr_vs_marked": None, "dct_code": dct,
        "dct_soft": peaked(dct),
    }
    if config == "original":
        row.update({"v3_39db": peaked(None), "v3": peaked(None)})
    else:
        row[decoder] = peaked(learned)
    return row


def test_two_marks_naming_different_people_are_caught_whichever_was_forged() -> None:
    rows = [
        reading(0, "original", "none", None, None),
        reading(1, "original", "none", None, None),
        # The DCT mark forged to the victim while the learned mark still names the owner:
        # the direction the old count missed.
        reading(1, "dct_then_v3_39db", "forge_dct", dct=6, learned=5),
        # The learned mark framed while the DCT mark still names the owner.
        reading(1, "v3_39db_then_dct", "pgd_f4", dct=5, learned=6),
        # Both forged to the same victim: nothing contradicts, nothing to catch.
        reading(1, "dct_then_v3", "forge_all", dct=6, learned=6, decoder="v3"),
    ]
    cells = layered.summarise(rows, calibration={0})["cells"]
    assert cells["dct_then_v3_39db|forge_dct"]["conflict_caught"] == 1
    assert cells["v3_39db_then_dct|pgd_f4"]["conflict_caught"] == 1
    assert cells["dct_then_v3|forge_all"]["conflict_caught"] == 0


def test_a_part_is_written_whole_or_not_at_all(tmp_path: Path) -> None:
    part = tmp_path / "photo_00001.json"
    layered.write_atomically(part, json.dumps([{"index": 1}]))
    assert json.loads(part.read_text(encoding="utf-8")) == [{"index": 1}]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["photo_00001.json"]
