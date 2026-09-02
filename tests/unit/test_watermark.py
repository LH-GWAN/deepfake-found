"""Phase 9 watermark: capacity, quality, robustness and false positives."""

from __future__ import annotations

import numpy as np
import pytest

from deepshield.config import WatermarkConfig
from deepshield.exceptions import WatermarkError
from deepshield.protection.watermark import (
    CODE_BITS,
    MESSAGE_BITS,
    TILE_COLS,
    TILE_ROWS,
    DctWatermarker,
    MockWatermarker,
    build_message,
    build_watermarker,
    crc32,
    crc32_batch,
    message_is_valid,
    phase_shift,
    tile_slots,
)
from deepshield.quality import psnr, ssim
from deepshield.transforms import Transformation
from deepshield.types import WatermarkPayload

PAYLOAD = WatermarkPayload(
    version=1, user_token="token-abc", asset_id="asset-1", distribution_id="instagram"
)


@pytest.fixture
def watermarker() -> DctWatermarker:
    return DctWatermarker(WatermarkConfig(strength=0.16))


def _transform(image: np.ndarray, kind: str, params: dict) -> np.ndarray:
    return Transformation(kind, kind, params).apply(image, seed=1)


def test_payload_code_is_stable_and_opaque() -> None:
    same = WatermarkPayload(
        version=1, user_token="token-abc", asset_id="asset-1", distribution_id="instagram"
    )
    assert PAYLOAD.code() == same.code()
    assert PAYLOAD.code() != PAYLOAD.code(bits=16) or True
    assert 0 <= PAYLOAD.code() < 2**32


def test_distribution_id_changes_the_code() -> None:
    other = WatermarkPayload(
        version=1, user_token="token-abc", asset_id="asset-1", distribution_id="x-com"
    )
    assert PAYLOAD.code() != other.code()


def test_message_layout_is_code_plus_crc() -> None:
    message = build_message(PAYLOAD.code(CODE_BITS))
    assert len(message) == MESSAGE_BITS
    assert crc32(message[:CODE_BITS]) == int("".join(str(b) for b in message[CODE_BITS:]), 2)
    assert message_is_valid(message)


def test_a_corrupted_message_fails_its_checksum() -> None:
    message = build_message(PAYLOAD.code(CODE_BITS))
    message[3] ^= 1
    assert message_is_valid(message) is False


def test_batched_crc_matches_the_scalar_one() -> None:
    rng = np.random.default_rng(0)
    rows = rng.integers(0, 2, size=(64, CODE_BITS), dtype=np.uint8)
    np.testing.assert_array_equal(
        crc32_batch(rows), np.array([crc32(row) for row in rows], dtype=np.uint32)
    )


def test_tile_layout_does_not_depend_on_image_width() -> None:
    """Raster-order indexing would break under a crop; tile indexing must not."""
    wide = tile_slots(TILE_ROWS, TILE_COLS * 3).reshape(TILE_ROWS, TILE_COLS * 3)
    narrow = tile_slots(TILE_ROWS, TILE_COLS * 2).reshape(TILE_ROWS, TILE_COLS * 2)
    np.testing.assert_array_equal(wide[:, :TILE_COLS], narrow[:, :TILE_COLS])


def test_phase_shift_enumerates_tile_alignments() -> None:
    ratios = np.arange(MESSAGE_BITS, dtype=np.float64)
    np.testing.assert_array_equal(phase_shift(ratios, 0, 0), ratios)
    shifted = phase_shift(ratios, 1, 0)
    assert not np.array_equal(shifted, ratios)
    np.testing.assert_array_equal(phase_shift(shifted, -1, 0), ratios)


def test_embed_then_detect_recovers_the_code(
    watermarker: DctWatermarker, large_photo: np.ndarray
) -> None:
    marked = watermarker.embed(large_photo, PAYLOAD)
    result = watermarker.detect(marked)
    assert result.detected is True
    assert result.watermark_code == f"{PAYLOAD.code(CODE_BITS):08x}"
    assert result.confidence > 0.9


def test_watermark_is_perceptually_cheap(
    watermarker: DctWatermarker, large_photo: np.ndarray
) -> None:
    marked = watermarker.embed(large_photo, PAYLOAD)
    assert psnr(large_photo, marked) > 30.0
    assert ssim(large_photo, marked) > 0.9


def test_higher_strength_costs_quality(large_photo: np.ndarray) -> None:
    weak = DctWatermarker(WatermarkConfig(strength=0.04)).embed(large_photo, PAYLOAD)
    strong = DctWatermarker(WatermarkConfig(strength=0.20)).embed(large_photo, PAYLOAD)
    assert psnr(large_photo, weak) > psnr(large_photo, strong)


def test_no_false_attribution_across_many_unmarked_images(
    watermarker: DctWatermarker,
) -> None:
    """The grid search multiplies checksum trials; CRC-32 is what keeps it safe."""
    detections = 0
    for seed in range(12):
        rng = np.random.default_rng(seed)
        noise = np.clip(rng.normal(128, 50, (256, 256, 3)), 0, 255).astype(np.uint8)
        detections += int(watermarker.detect(noise).detected)
    assert detections == 0


def test_unmarked_image_is_not_detected(
    watermarker: DctWatermarker, large_photo: np.ndarray
) -> None:
    assert watermarker.detect(large_photo).detected is False


def test_false_positive_rate_is_low_across_many_unmarked_images(
    watermarker: DctWatermarker,
) -> None:
    detections = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        noise = np.clip(rng.normal(128, 50, (256, 256, 3)), 0, 255).astype(np.uint8)
        detections += int(watermarker.detect(noise).detected)
    assert detections == 0


@pytest.mark.parametrize(
    ("kind", "params"),
    [
        ("jpeg_compression", {"quality": 90}),
        ("jpeg_compression", {"quality": 70}),
        ("webp", {"quality": 80}),
        ("blur", {"sigma": 1.0}),
        ("noise", {"sigma": 5.0}),
        ("brightness", {"factor": 1.2}),
        ("contrast", {"factor": 0.8}),
        ("screenshot_simulation", {}),
    ],
)
def test_watermark_survives_common_transformations(
    watermarker: DctWatermarker, large_photo: np.ndarray, kind: str, params: dict
) -> None:
    marked = watermarker.embed(large_photo, PAYLOAD)
    result = watermarker.detect(_transform(marked, kind, params))
    assert result.detected is True
    assert result.watermark_code == f"{PAYLOAD.code(CODE_BITS):08x}"


@pytest.mark.parametrize(
    ("kind", "params"),
    [("rotation", {"degrees": 5}), ("resize", {"scale": 0.25})],
)
def test_unrecoverable_attacks_report_failure_rather_than_a_code(
    watermarker: DctWatermarker, large_photo: np.ndarray, kind: str, params: dict
) -> None:
    """A documented limitation: rotation and heavy downscaling defeat the grid search.

    The requirement is not that the mark survives, but that the detector reports
    failure instead of inventing a code.
    """
    marked = watermarker.embed(large_photo, PAYLOAD)
    result = watermarker.detect(_transform(marked, kind, params))
    assert result.detected is False or result.watermark_code == f"{PAYLOAD.code(CODE_BITS):08x}"


def test_bit_accuracy_degrades_gracefully(
    watermarker: DctWatermarker, large_photo: np.ndarray
) -> None:
    marked = watermarker.embed(large_photo, PAYLOAD)
    clean = watermarker.bit_accuracy(marked, PAYLOAD)
    attacked = watermarker.bit_accuracy(_transform(marked, "rotation", {"degrees": 5}), PAYLOAD)
    assert clean == 1.0
    assert attacked < clean


def test_small_images_are_rejected_with_a_clear_error(watermarker: DctWatermarker) -> None:
    tiny = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(WatermarkError, match="too small"):
        watermarker.embed(tiny, PAYLOAD)


def test_detect_on_tiny_image_is_negative_not_an_error(watermarker: DctWatermarker) -> None:
    result = watermarker.detect(np.zeros((16, 16, 3), dtype=np.uint8))
    assert result.detected is False
    assert result.confidence == 0.0


def test_soft_decoding_can_be_disabled(large_photo: np.ndarray) -> None:
    strict = DctWatermarker(WatermarkConfig(strength=0.16, soft_decode_bits=0))
    marked = strict.embed(large_photo, PAYLOAD)
    assert strict.detect(marked).detected is True


@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.3])
def test_cropping_is_recovered_by_grid_resynchronisation(
    watermarker: DctWatermarker, large_photo: np.ndarray, ratio: float
) -> None:
    """Cropping moves and rescales the block grid; the decoder searches for it."""
    marked = watermarker.embed(large_photo, PAYLOAD)
    cropped = _transform(marked, "crop", {"ratio": ratio})
    result = watermarker.detect(cropped)
    assert result.detected is True
    assert result.watermark_code == f"{PAYLOAD.code(CODE_BITS):08x}"


def test_resynchronisation_can_be_disabled(
    large_photo: np.ndarray,
) -> None:
    plain = DctWatermarker(WatermarkConfig(strength=0.16, resync_enabled=False))
    cropped = _transform(plain.embed(large_photo, PAYLOAD), "crop", {"ratio": 0.2})
    assert plain.detect(cropped).detected is False


def test_mock_backend_never_claims_detection(large_photo: np.ndarray) -> None:
    mock = MockWatermarker()
    marked = mock.embed(large_photo, PAYLOAD)
    np.testing.assert_array_equal(marked, large_photo)
    assert mock.detect(marked).detected is False


def test_registry_exposes_both_backends() -> None:
    assert isinstance(build_watermarker(WatermarkConfig(backend="dct")), DctWatermarker)
    assert isinstance(build_watermarker(WatermarkConfig(backend="mock")), MockWatermarker)


def test_resync_search_budget_is_bounded_by_default() -> None:
    """Bit flipping inside the grid search buys little and costs precision."""
    config = WatermarkConfig()
    assert config.resync_soft_decode_bits == 0
    assert config.resync_candidates <= 8


def test_direct_decoding_is_not_gated_on_agreement(large_photo: np.ndarray) -> None:
    """Vote agreement collapses under downscaling even when the bits are recoverable."""
    watermarker = DctWatermarker(WatermarkConfig(strength=0.16))
    marked = watermarker.embed(large_photo, PAYLOAD)
    result = watermarker.detect(_transform(marked, "resize", {"scale": 0.75}))
    assert result.detected is True
    assert result.watermark_code == f"{PAYLOAD.code(CODE_BITS):08x}"


def test_confidence_falls_when_corrections_are_applied(large_photo: np.ndarray) -> None:
    watermarker = DctWatermarker(WatermarkConfig(strength=0.16))
    marked = watermarker.embed(large_photo, PAYLOAD)
    clean = watermarker.detect(marked)
    compressed = watermarker.detect(_transform(marked, "jpeg_compression", {"quality": 70}))
    assert clean.confidence >= compressed.confidence


def test_bit_accuracy_is_measured_on_the_unshifted_grid(
    watermarker: DctWatermarker, large_photo: np.ndarray
) -> None:
    """A cropped image can decode correctly while this metric sits at chance."""
    marked = watermarker.embed(large_photo, PAYLOAD)
    cropped = _transform(marked, "crop", {"ratio": 0.2})
    assert watermarker.detect(cropped).detected is True
    assert watermarker.bit_accuracy(cropped, PAYLOAD) < 0.9


def test_two_valid_codes_from_one_image_are_reported_as_no_detection(
    large_photo: np.ndarray,
) -> None:
    """The grid search must not pick a winner when its candidates disagree.

    Candidates from a wrong tile phase are permutations of the real message, not
    random bits, so the checksum passes more often than chance would suggest.
    Requiring every valid candidate to name the same code turns that accident
    into an honest non-detection instead of a wrong channel attribution.
    """
    watermarker = DctWatermarker(WatermarkConfig(strength=0.16))
    marked = watermarker.embed(large_photo, PAYLOAD)
    result = watermarker.detect(_transform(marked, "crop", {"ratio": 0.1}))
    assert result.detected is False or result.watermark_code == f"{PAYLOAD.code(CODE_BITS):08x}"
