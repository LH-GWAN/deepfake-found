"""Scoring and split logic behind the learned-watermark attribution benchmarks.

The scripts need torch at import time, so the whole module is skipped without
it, the same way the detector export tests are.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_learned_watermark_attribution as attribution  # noqa: E402
import evaluate_watermark_removal as removal  # noqa: E402

from deepshield.config import WatermarkConfig  # noqa: E402
from deepshield.protection.watermark import CODE_BITS, DctWatermarker  # noqa: E402
from deepshield.types import WatermarkPayload  # noqa: E402


def test_rank_auc_is_the_probability_a_positive_outscores_a_negative() -> None:
    assert attribution.rank_auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert attribution.rank_auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 0.0
    assert attribution.rank_auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    assert np.isnan(attribution.rank_auc(np.array([]), np.array([1.0])))


def test_soft_correlation_prefers_the_embedded_code() -> None:
    rng = np.random.default_rng(0)
    codebook = rng.integers(0, 2, (72, 32)).astype(np.uint8)
    truth = 17
    logits = (2.0 * codebook[truth] - 1.0) * 3.0 + rng.normal(0, 0.5, 32)
    scores = attribution.score(logits, codebook)
    assert scores.shape == (72,)
    assert int(np.argmax(scores)) == truth


def test_closed_set_top1_counts_wins_over_random_subsets() -> None:
    scores = np.zeros(72)
    scores[5] = 10.0
    rng = np.random.default_rng(1)
    assert attribution.closed_set_top1(scores, 5, 8, rng) == 1.0
    assert attribution.closed_set_top1(scores, 5, 72, rng) == 1.0
    scores[6] = 20.0
    assert attribution.closed_set_top1(scores, 5, 72, rng) == 0.0
    assert 0.0 <= attribution.closed_set_top1(scores, 5, 2, rng) <= 1.0


def _rows(scores: list[float], top1: float) -> list[dict]:
    return [
        {
            "detection_score": value,
            "top1": top1,
            "closed_set": {str(size): top1 for size in attribution.SUBSETS},
        }
        for value in scores
    ]


def test_summary_fits_thresholds_on_one_half_and_reports_on_the_other() -> None:
    marked = _rows([30.0] * 20, 1.0)
    unmarked = _rows([10.0 + index * 0.1 for index in range(20)], 0.0)
    summary = attribution.summarise(marked, unmarked, seed=0)
    assert summary["split"] == {"calibration": 10, "test": 10}
    detection = summary["detection"]
    assert detection["test_detection_rate"] == 1.0
    assert detection["test_false_positive_rate"] == 0.0
    assert detection["test_auc"] == 1.0
    assert 10.0 < detection["threshold_zero_fp_on_calibration"] < 30.0
    assert summary["attribution_test_half"]["top1_of_72"] == 1.0


def test_identity_pairs_never_pair_a_person_with_themselves(tmp_path: Path) -> None:
    for name in ("Ann_Lee", "Bob_Ray", "Cy_Young", "Di_Fox"):
        folder = tmp_path / name
        folder.mkdir()
        for index in (1, 2):
            (folder / f"{name}_{index:04d}.jpg").write_bytes(b"")
    pairs = attribution.identity_pairs(tmp_path)
    assert len(pairs) == 2
    for own, other in pairs:
        assert own.parent.name != other.parent.name
        assert own.name.endswith("_0001.jpg")


PAYLOAD = WatermarkPayload(version=1, user_token="owner", asset_id="photo", distribution_id="a")


def test_equalising_the_carrier_coefficients_removes_the_dct_mark(
    large_photo: np.ndarray,
) -> None:
    """A keyless mark is removable by anyone who has read the source."""
    watermarker = DctWatermarker(WatermarkConfig(strength=0.16))
    marked = watermarker.embed(large_photo, PAYLOAD)
    assert watermarker.detect(marked).detected is True
    assert watermarker.detect(removal.dct_equalise(marked)).detected is False


def test_overwriting_forges_another_code_over_the_dct_mark(large_photo: np.ndarray) -> None:
    watermarker = DctWatermarker(WatermarkConfig(strength=0.16))
    forger = WatermarkPayload(
        version=1, user_token="forger", asset_id="photo", distribution_id="b"
    )
    forged = watermarker.embed(watermarker.embed(large_photo, PAYLOAD), forger)
    result = watermarker.detect(forged)
    assert result.detected is True
    assert result.watermark_code == f"{forger.code(CODE_BITS):08x}"


def test_uninformed_attacks_preserve_shape_and_dtype(large_photo: np.ndarray) -> None:
    for name, attack in removal.uninformed_attacks().items():
        out = attack(large_photo)
        assert out.shape == large_photo.shape, name
        assert out.dtype == np.uint8, name
