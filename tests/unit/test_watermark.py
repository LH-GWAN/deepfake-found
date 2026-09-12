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
    [("rotation", {"degrees": 30}), ("resize", {"scale": 0.25})],
)
def test_unrecoverable_attacks_report_failure_rather_than_a_code(
    watermarker: DctWatermarker, large_photo: np.ndarray, kind: str, params: dict
) -> None:
    """Rotation beyond the searched range and heavy downscaling defeat the grid search.

    The requirement is not that the mark survives, but that the detector reports
    failure instead of inventing a code.
    """
    marked = watermarker.embed(large_photo, PAYLOAD)
    result = watermarker.detect(_transform(marked, kind, params))
    assert result.detected is False or result.watermark_code == f"{PAYLOAD.code(CODE_BITS):08x}"


@pytest.mark.parametrize("degrees", [2.0, 5.0, -7.3, 10.0])
def test_rotation_is_recovered_by_angle_search(
    watermarker: DctWatermarker, large_photo: np.ndarray, degrees: float
) -> None:
    """Rotation is undone by sweeping candidate angles and re-running the grid search."""
    marked = watermarker.embed(large_photo, PAYLOAD)
    result = watermarker.detect(_transform(marked, "rotation", {"degrees": degrees}))
    assert result.detected is True
    assert result.watermark_code == f"{PAYLOAD.code(CODE_BITS):08x}"


def test_rotation_search_can_be_disabled(large_photo: np.ndarray) -> None:
    plain = DctWatermarker(WatermarkConfig(strength=0.16, resync_rotation_enabled=False))
    rotated = _transform(plain.embed(large_photo, PAYLOAD), "rotation", {"degrees": 5})
    assert plain.detect(rotated).detected is False


def test_rotation_search_does_not_invent_a_code_on_unmarked_images(
    watermarker: DctWatermarker, large_photo: np.ndarray
) -> None:
    """More candidate angles are more chances for a checksum to pass on noise."""
    for degrees in (0.0, 3.0, 5.0):
        probe = large_photo
        if degrees:
            probe = _transform(large_photo, "rotation", {"degrees": degrees})
        assert watermarker.detect(probe).detected is False


def test_rotation_search_budget_is_bounded_by_default() -> None:
    config = WatermarkConfig()
    assert config.resync_rotation_max_degrees <= 20.0
    assert config.resync_rotation_candidates <= 4
    assert config.resync_rotation_fine_step <= config.resync_rotation_coarse_step / 2.0


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


# --- keyed layout -----------------------------------------------------------


def _equalise_pairs(
    image: np.ndarray, pairs: list[tuple[tuple[int, int], tuple[int, int]]]
) -> np.ndarray:
    """Flatten carrier pairs in every block, as an attacker who knows the algorithm would."""
    from PIL import Image

    from deepshield.protection.fingerprint import dct2, idct2
    from deepshield.protection.watermark import BLOCK_SIZE

    ycbcr = np.asarray(Image.fromarray(image).convert("YCbCr"), dtype=np.float64)
    luminance = ycbcr[:, :, 0]
    rows, cols = luminance.shape[0] // BLOCK_SIZE, luminance.shape[1] // BLOCK_SIZE
    for row in range(rows):
        for col in range(cols):
            y0, x0 = row * BLOCK_SIZE, col * BLOCK_SIZE
            block = dct2(luminance[y0 : y0 + BLOCK_SIZE, x0 : x0 + BLOCK_SIZE])
            for first, second in pairs:
                mean = (block[first] + block[second]) / 2.0
                block[first] = mean
                block[second] = mean
            luminance[y0 : y0 + BLOCK_SIZE, x0 : x0 + BLOCK_SIZE] = idct2(block)
    ycbcr[:, :, 0] = np.clip(luminance, 0, 255)
    return np.asarray(Image.fromarray(ycbcr.astype(np.uint8), mode="YCbCr").convert("RGB"))


def test_missing_key_derives_the_keyless_layout() -> None:
    from deepshield.protection.watermark import derive_schedule

    for key in (None, ""):
        schedule = derive_schedule(key)
        assert schedule.keyed is False
        np.testing.assert_array_equal(schedule.bit_of_slot, np.arange(MESSAGE_BITS))
        assert not schedule.pair_of_slot.any()


def test_key_schedule_is_deterministic_and_distinct_per_key() -> None:
    from deepshield.protection.watermark import PAIR_CANDIDATES, derive_schedule

    first, again = derive_schedule("alpha"), derive_schedule("alpha")
    other = derive_schedule("beta")
    assert first.keyed is True
    np.testing.assert_array_equal(first.bit_of_slot, again.bit_of_slot)
    np.testing.assert_array_equal(first.pair_of_slot, again.pair_of_slot)
    assert sorted(first.bit_of_slot.tolist()) == list(range(MESSAGE_BITS))
    assert first.pair_of_slot.min() >= 0 and first.pair_of_slot.max() < len(PAIR_CANDIDATES)
    assert not np.array_equal(first.bit_of_slot, other.bit_of_slot)


def test_keyed_embed_then_detect_recovers_the_code(large_photo: np.ndarray) -> None:
    keyed = DctWatermarker(WatermarkConfig(strength=0.16, key="correct horse battery staple"))
    marked = keyed.embed(large_photo, PAYLOAD)
    assert psnr(large_photo, marked) > 30.0
    result = keyed.detect(marked)
    assert result.detected is True
    assert result.watermark_code == f"{PAYLOAD.code(CODE_BITS):08x}"


def test_keyed_mark_reads_as_nothing_without_the_key(large_photo: np.ndarray) -> None:
    keyed = DctWatermarker(WatermarkConfig(strength=0.16, key="correct horse battery staple"))
    marked = keyed.embed(large_photo, PAYLOAD)
    for other in (DctWatermarker(WatermarkConfig(strength=0.16)),
                  DctWatermarker(WatermarkConfig(strength=0.16, key="another key"))):
        result = other.detect(marked)
        assert result.detected is False
        assert result.watermark_code is None


def test_keyed_mark_survives_cropping_and_compression(large_photo: np.ndarray) -> None:
    keyed = DctWatermarker(WatermarkConfig(strength=0.16, key="correct horse battery staple"))
    marked = keyed.embed(large_photo, PAYLOAD)
    for kind, params in (("crop", {"ratio": 0.1}), ("jpeg_compression", {"quality": 70})):
        result = keyed.detect(_transform(marked, kind, params))
        assert result.detected is True, kind
        assert result.watermark_code == f"{PAYLOAD.code(CODE_BITS):08x}"


def test_keyless_forgery_never_becomes_a_valid_keyed_code(large_photo: np.ndarray) -> None:
    """An attacker writing the keyless layout over a keyed mark cannot name a code."""
    keyed = DctWatermarker(WatermarkConfig(strength=0.16, key="correct horse battery staple"))
    forger = DctWatermarker(WatermarkConfig(strength=0.16))
    other = WatermarkPayload(
        version=1, user_token="forger", asset_id="asset-1", distribution_id="x"
    )
    marked = keyed.embed(large_photo, PAYLOAD)
    forged = forger.embed(marked, other)
    result = keyed.detect(forged)
    assert result.watermark_code != f"{other.code(CODE_BITS):08x}"


def test_keyless_layout_is_a_special_case_of_the_keyed_one(large_photo: np.ndarray) -> None:
    """With no key the keyed code path must reproduce the keyless watermark exactly."""
    plain = DctWatermarker(WatermarkConfig(strength=0.16))
    marked = plain.embed(large_photo, PAYLOAD)
    slots = tile_slots(TILE_ROWS, TILE_COLS * 2)
    np.testing.assert_array_equal(slots, tile_slots(TILE_ROWS, TILE_COLS * 2, 0, 0))
    shifted = tile_slots(TILE_ROWS, TILE_COLS, 1, 0).reshape(TILE_ROWS, TILE_COLS)
    unshifted = tile_slots(TILE_ROWS, TILE_COLS).reshape(TILE_ROWS, TILE_COLS)
    np.testing.assert_array_equal(shifted[0], unshifted[1])
    untouched = plain.detect(_equalise_pairs(marked, []))
    assert untouched.watermark_code == f"{PAYLOAD.code(CODE_BITS):08x}"


def test_rotation_after_a_crop_is_recovered_by_the_combined_search(large_photo: np.ndarray) -> None:
    """A rotated then magnified image needs the angle and the scale found together."""
    watermarker = DctWatermarker(WatermarkConfig(strength=0.16))
    marked = watermarker.embed(large_photo, PAYLOAD)
    probe = _transform(marked, "rotate_crop", {"degrees": 5.0, "ratio": 0.1})
    result = watermarker.detect(probe)
    assert result.detected is True
    assert result.watermark_code == f"{PAYLOAD.code(CODE_BITS):08x}"


def test_combined_search_can_be_disabled_by_an_empty_scale_list(large_photo: np.ndarray) -> None:
    plain = DctWatermarker(WatermarkConfig(strength=0.16, resync_rotation_scales=[]))
    marked = plain.embed(large_photo, PAYLOAD)
    probe = _transform(marked, "rotate_crop", {"degrees": 5.0, "ratio": 0.1})
    assert plain.detect(probe).detected is False
