"""The two screens run after the three limits: a detector's condition split and AdaFace."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compare_face_embedders as screen  # noqa: E402
import evaluate_detector_conditions as split  # noqa: E402
import fetch_face_embedder as adaface  # noqa: E402


def test_the_threshold_rule_reproduces_the_deployed_thresholds() -> None:
    """The deployed pair came from impostor max 0.3323 and genuine min 0.4397."""
    placed = screen.place_thresholds(np.array([0.4397, 0.9]), np.array([0.1, 0.3323]))
    assert placed["separated"]
    assert placed["candidate"] == pytest.approx(0.3591, abs=1e-3)
    assert placed["high_confidence"] == pytest.approx(0.4128, abs=1e-3)


def test_overlapping_scores_are_reported_as_such() -> None:
    placed = screen.place_thresholds(np.array([0.2, 0.9]), np.array([0.1, 0.3]))
    assert not placed["separated"]
    assert placed["gap"] < 0


def test_genuine_probes_fall_into_three_bands() -> None:
    thresholds = {"candidate": 0.36, "high_confidence": 0.41}
    counts = screen.bands(np.array([0.1, 0.36, 0.40, 0.41, 0.9]), thresholds)
    assert counts == {"high_confidence": 2, "review": 2, "below_candidate": 1}


def test_fusion_averages_the_same_pairs_and_refuses_misaligned_ones() -> None:
    labels = np.array([1, 0, 0])
    first = {"clean": {"scores": np.array([0.8, 0.2, 0.1]), "labels": labels,
                       "probe_failures": 0},
             "_meta": {"gallery_failures": 0, "seconds_per_image": 1.0}}
    second = {"clean": {"scores": np.array([0.6, 0.4, 0.1]), "labels": labels,
                        "probe_failures": 1},
              "_meta": {"gallery_failures": 2, "seconds_per_image": 7.0}}
    fused = screen.fuse(first, second)
    assert fused["clean"]["scores"] == pytest.approx([0.7, 0.3, 0.1])
    assert fused["clean"]["probe_failures"] == 1
    assert fused["_meta"]["gallery_failures"] == 2
    second["clean"]["labels"] = np.array([0, 1, 0])
    with pytest.raises(ValueError, match="different pairs"):
        screen.fuse(first, second)


def test_the_wrapped_checkpoint_is_reduced_to_the_network() -> None:
    weights = {"model.net.input_layer.0.weight": 1, "net.body.0.x": 2, "head.kernel": 3}
    assert adaface.strip_to_network(weights) == {"input_layer.0.weight": 1, "body.0.x": 2}


def test_only_batchnorm_counters_may_be_missing() -> None:
    adaface.check_keys(SimpleNamespace(
        missing_keys=["body.0.res_layer.0.num_batches_tracked"], unexpected_keys=[]
    ))
    with pytest.raises(SystemExit, match="missing"):
        adaface.check_keys(SimpleNamespace(missing_keys=["body.0.res_layer.1.weight"],
                                           unexpected_keys=[]))
    with pytest.raises(SystemExit, match="unexpected"):
        adaface.check_keys(SimpleNamespace(missing_keys=[], unexpected_keys=["head.kernel"]))


def test_ir101_has_the_published_layout() -> None:
    torch = pytest.importorskip("torch")
    model = adaface.build_ir101().eval()
    state = model.state_dict()
    assert len(model.body) == 3 + 13 + 30 + 3
    assert tuple(state["input_layer.0.weight"].shape) == (64, 3, 3, 3)
    assert tuple(state["body.3.shortcut_layer.0.weight"].shape) == (128, 64, 1, 1)
    assert "body.0.shortcut_layer.0.weight" not in state
    assert tuple(state["output_layer.3.weight"].shape) == (512, 512 * 7 * 7)
    assert "output_layer.4.running_mean" in state and "output_layer.4.weight" not in state
    with torch.no_grad():
        assert tuple(model(torch.zeros(1, 3, 112, 112)).shape) == (1, 512)


def item(label: str, half: str, family: str, manifest: str, path: str) -> dict[str, Any]:
    return {"label": label, "half": half, "family": family, "manifest": manifest, "path": path}


def test_the_split_judges_conditions_and_face_widths_at_the_gate_threshold() -> None:
    calibration = [item("real", "calibration", "lfw", "genuine", f"c{i}") for i in range(201)]
    tested = [
        item("real", "test", "lfw", "genuine", "g1"),
        item("real", "test", "lfw", "genuine", "g2"),
        item("real", "test", "graphics", "m", "s1"),
        item("fake", "test", "graphics", "m", "f1"),
        item("fake", "test", "graphics", "m", "f2"),
        item("fake", "test", "inswapper", "i", "f3"),
    ]
    items = calibration + tested
    clean = np.concatenate([np.linspace(0.0, 0.9, 201), [0.1, 0.95, 0.2, 0.95, 0.3, 0.95]])
    small = np.concatenate([np.linspace(0.0, 0.9, 201), [0.95, 0.95, 0.95, 0.95, 0.95, 0.1]])
    widths = {"clean": np.full(len(items), 100.0), "down55": np.full(len(items), 55.0)}
    report = split.summarise(items, {"clean": clean, "down55": small}, widths)

    assert report["threshold"] == pytest.approx(np.quantile(np.linspace(0.0, 0.9, 201), 0.995))
    assert report["conditions"]["clean"]["genuine"]["false_positives"] == 1
    assert report["conditions"]["down55"]["genuine"]["false_positives"] == 3
    assert report["conditions"]["clean"]["families"]["graphics"]["recall"] == 0.5
    assert report["conditions"]["down55"]["families"]["inswapper"]["recall"] == 0.0
    assert report["conditions"]["clean"]["median_genuine_face_width"] == 100.0
    assert report["by_face_width"]["48-64"]["genuine"]["false_positives"] == 3
    assert report["by_face_width"]["96-"]["genuine"]["false_positives"] == 1
    rule = report["size_rules"]["face_at_least_64px"]
    assert rule["genuine"]["false_positives"] == 1
    assert rule["abstains_on"] == 0.5


def test_only_clean_scores_the_calibration_half() -> None:
    items = [item("real", "calibration", "lfw", "genuine", "a"),
             item("fake", "test", "graphics", "m", "b")]
    assert split.scored(items, "clean").tolist() == [True, True]
    assert split.scored(items, "down55_crf35").tolist() == [False, True]
    assert list(split.GRID)[:4] == ["clean", "down55", "crf35", "down55_crf35"]


def test_a_cache_is_tied_to_its_items() -> None:
    first = [item("real", "test", "lfw", "genuine", "a")]
    second = [item("real", "test", "lfw", "genuine", "b")]
    assert split.signature(first) != split.signature(second)
    assert split.signature(first) == split.signature(list(first))
