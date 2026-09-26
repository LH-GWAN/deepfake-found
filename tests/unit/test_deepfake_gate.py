"""The adoption gate that decides whether a deepfake detector may change a verdict."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_deepfake_detectors as gate  # noqa: E402


def item(label: str, people: list[str], family: str = "inswapper") -> dict[str, Any]:
    return {"path": "", "label": label, "family": family, "manifest": "m", "people": people}


def test_identity_is_read_from_an_evaluation_file_name() -> None:
    assert gate.identity_of("thomas_fargo_3.png") == "thomas_fargo"
    assert gate.identity_of("Angela_Bassett_12.png") == "angela_bassett"


def test_a_fake_whose_donor_and_target_straddle_the_halves_is_used_by_neither() -> None:
    items = [item("real", [f"p{i}"]) for i in range(40)]
    items.append(item("fake", ["p0", "p1"]))
    gate.assign_halves(items, 0.5, seed=3)
    sides = {entry["people"][0]: entry["half"] for entry in items[:40]}
    expected = "split" if sides["p0"] != sides["p1"] else sides["p0"]
    assert items[-1]["half"] == expected
    assert {entry["half"] for entry in items[:40]} == {"calibration", "test"}


def test_a_detector_trained_on_an_evaluated_manifest_is_in_sample() -> None:
    metadata = {"training_families_manifests": ["data/test/manipulated/manifest.json"]}
    assert gate.in_sample(metadata, [Path("data/test/manipulated/manifest.json")])
    assert not gate.in_sample(metadata, [Path("data/test/manipulated_inswapper/manifest.json")])
    assert not gate.in_sample({}, [Path("data/test/manipulated/manifest.json")])


def test_the_leakage_guard_does_not_depend_on_how_a_path_is_written() -> None:
    """``data/...`` and its absolute form, or a Colab mount of it, are one manifest."""
    relative = Path("data/test/manipulated/manifest.json")
    absolute = (gate.ROOT / relative).resolve()
    assert gate.in_sample({"training_families_manifests": [str(relative)]}, [absolute])
    assert gate.in_sample({"training_families_manifests": [absolute.as_posix()]}, [relative])
    mount = "/content/deepshield/data/test/manipulated/manifest.json"
    colab = {"training_families_manifests": [mount]}
    assert gate.in_sample(colab, [relative])
    assert not gate.in_sample(colab, [Path("data/test/manipulated_inswapper/manifest.json")])


def test_the_leakage_guard_recognises_a_manifest_by_its_content(tmp_path: Path) -> None:
    moved = tmp_path / "elsewhere.json"
    moved.write_text('{"records": []}', encoding="utf-8")
    metadata = {
        "training_families_manifests": ["/somewhere/else/manifest.json"],
        "training_manifests_sha256": [gate.file_sha256(moved)],
    }
    assert gate.in_sample(metadata, [moved])


def test_genuine_photographs_shared_by_two_manifests_count_once(tmp_path: Path) -> None:
    """The manipulation sets each keep a byte-identical copy of the same genuine photos."""
    import json

    manifests = []
    for name in ("manipulated", "manipulated_inswapper"):
        folder = tmp_path / name
        (folder / "real").mkdir(parents=True)
        (folder / "real" / "ann_0.jpg").write_bytes(b"the same photograph")
        records = [
            {"path": (folder / "real" / "ann_0.jpg").as_posix(), "label": "real",
             "identity": "ann", "source": "ann_0.png"},
            {"path": (folder / "fake.jpg").as_posix(), "label": "fake", "identity": "ann",
             "source": "ann_0.png", "donor": "bob_1.png"},
        ]
        (folder / "manifest.json").write_text(json.dumps({"records": records}), encoding="utf-8")
        manifests.append(folder / "manifest.json")
    records = gate.load_records(manifests)
    assert [record["label"] for record in records] == ["real", "fake", "fake"]
    assert {record["family"] for record in records if record["label"] == "fake"} == {
        "graphics", "inswapper"
    }


@pytest.mark.parametrize("threshold", [0.5, 0.95, 0.9996, 0.999557])
def test_the_high_confidence_threshold_stays_above_the_suspicious_one(threshold: float) -> None:
    suspicious, high = gate.operating_points(threshold)
    assert float(suspicious) == pytest.approx(threshold, abs=5e-7)
    assert float(high) > float(suspicious)
    assert float(high) < 1.0


@pytest.mark.parametrize("threshold", [1.0, 0.9999995])
def test_a_threshold_with_no_room_above_it_is_not_adopted(threshold: float) -> None:
    with pytest.raises(ValueError, match="no high-confidence band"):
        gate.operating_points(threshold)


def scored(
    genuine_test: int, alarms: int, fakes_test: int, caught: int
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    items: list[dict[str, Any]] = []
    values: list[float] = []
    for index in range(400):
        items.append({**item("real", [f"c{index}"]), "half": "calibration"})
        values.append(index / 400)
    for index in range(genuine_test):
        items.append({**item("real", [f"t{index}"]), "half": "test"})
        values.append(0.999 if index < alarms else 0.1)
    for index in range(fakes_test):
        items.append({**item("fake", [f"f{index}"]), "half": "test"})
        values.append(0.999 if index < caught else 0.2)
    return items, {"clean": np.asarray(values)}


def test_the_threshold_is_fitted_on_calibration_genuine_photos_only() -> None:
    items, scores = scored(genuine_test=500, alarms=0, fakes_test=50, caught=20)
    report = gate.evaluate(items, scores)
    expected = np.quantile(np.arange(400) / 400, gate.GENUINE_QUANTILE)
    assert report["threshold"] == pytest.approx(expected, abs=1e-6)
    assert report["genuine"]["clean"]["false_positives"] == 0
    assert report["families"]["inswapper"]["clean"]["test_recall"] == 0.4


def test_the_gate_needs_enough_genuine_photos_to_bound_false_alarms() -> None:
    items, scores = scored(genuine_test=100, alarms=0, fakes_test=50, caught=20)
    usable, reasons = gate.verdict(gate.evaluate(items, scores), leaked=False)
    assert not usable
    assert any("could be as high as" in reason for reason in reasons)


def test_the_gate_passes_a_detector_that_is_rarely_wrong_and_sometimes_right() -> None:
    items, scores = scored(genuine_test=1000, alarms=0, fakes_test=50, caught=20)
    usable, reasons = gate.verdict(gate.evaluate(items, scores), leaked=False)
    assert usable, reasons


def test_the_gate_refuses_a_detector_that_never_fires() -> None:
    items, scores = scored(genuine_test=1000, alarms=0, fakes_test=50, caught=5)
    usable, reasons = gate.verdict(gate.evaluate(items, scores), leaked=False)
    assert not usable
    assert any("below 20%" in reason for reason in reasons)


def test_the_gate_refuses_in_sample_numbers_however_good() -> None:
    items, scores = scored(genuine_test=1000, alarms=0, fakes_test=50, caught=50)
    usable, reasons = gate.verdict(gate.evaluate(items, scores), leaked=True)
    assert not usable
    assert any("in-sample" in reason for reason in reasons)
