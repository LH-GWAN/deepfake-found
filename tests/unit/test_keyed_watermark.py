"""Keyed DCT layout: what a key changes, and what an attacker without it cannot do.

The keyless mark is breakable by anyone who reads the source, and this
repository already measures the cost: equalising the two carrier coefficients
removes it, and re-embedding forges it. These tests state the property a key is
supposed to buy - that both attacks need the key - and the control that keeps
the claim honest: an attacker who *has* the key must still succeed, or the
attack code is broken rather than the key working.
"""

from __future__ import annotations

import numpy as np
import pytest

from deepshield.config import WatermarkConfig
from deepshield.protection.fingerprint import dct2, idct2
from deepshield.protection.watermark import (
    BLOCK_SIZE,
    CARRIER_PAIRS,
    CODE_BITS,
    COEFFICIENT_A,
    COEFFICIENT_B,
    MESSAGE_BITS,
    DctWatermarker,
    derive_layout,
)
from deepshield.quality import psnr
from deepshield.transforms import Transformation
from deepshield.types import WatermarkPayload

OWNER = WatermarkPayload(
    version=1, user_token="owner", asset_id="photo", distribution_id="instagram"
)
FORGER = WatermarkPayload(version=1, user_token="forger", asset_id="photo", distribution_id="x-com")

OWNER_KEY = "owner-secret"
FORGER_KEY = "forger-secret"


def _code(payload: WatermarkPayload) -> str:
    return f"{payload.code(CODE_BITS):08x}"


def _equalise(image: np.ndarray, key: str | None) -> np.ndarray:
    """Run the informed removal attack against the layout ``key`` selects.

    This is ``dct_equalise`` from ``scripts/evaluate_watermark_removal.py`` with
    the carrier pair taken from a key instead of from the module constants, so
    the test can hand the attacker a right or a wrong key.
    """
    from PIL import Image

    layout = derive_layout(key)
    ycbcr = np.asarray(Image.fromarray(image).convert("YCbCr"), dtype=np.float64)
    luminance = ycbcr[:, :, 0]
    rows, cols = luminance.shape[0] // BLOCK_SIZE, luminance.shape[1] // BLOCK_SIZE
    for row in range(rows):
        for col in range(cols):
            y0, x0 = row * BLOCK_SIZE, col * BLOCK_SIZE
            block = dct2(luminance[y0 : y0 + BLOCK_SIZE, x0 : x0 + BLOCK_SIZE])
            mean = (block[layout.coefficient_a] + block[layout.coefficient_b]) / 2.0
            block[layout.coefficient_a] = mean
            block[layout.coefficient_b] = mean
            luminance[y0 : y0 + BLOCK_SIZE, x0 : x0 + BLOCK_SIZE] = idct2(block)
    ycbcr[:, :, 0] = np.clip(luminance, 0, 255)
    return np.asarray(Image.fromarray(ycbcr.astype(np.uint8), mode="YCbCr").convert("RGB"))


def _marker(key: str | None) -> DctWatermarker:
    return DctWatermarker(WatermarkConfig(strength=0.16, key=key))


def _transform(image: np.ndarray, kind: str, params: dict) -> np.ndarray:
    return Transformation(kind, kind, params).apply(image, seed=1)


# --- the default is keyless, deliberately -----------------------------------


def test_the_default_key_is_none_and_reproduces_the_published_layout() -> None:
    """A default key baked into the source is not a key.

    Shipping a fixed one would be as breakable as the constants it replaced
    while sounding safe, so the default stays keyless and every number measured
    for the keyless mark keeps applying to it.
    """
    assert WatermarkConfig().key is None
    layout = derive_layout(None)
    assert layout.coefficient_a == COEFFICIENT_A
    assert layout.coefficient_b == COEFFICIENT_B
    np.testing.assert_array_equal(layout.permutation, np.arange(MESSAGE_BITS))
    np.testing.assert_array_equal(layout.inverse, np.arange(MESSAGE_BITS))


def test_the_keyless_layout_is_one_of_the_carriers() -> None:
    assert (COEFFICIENT_A, COEFFICIENT_B) in CARRIER_PAIRS


def test_a_key_moves_the_mark(large_photo: np.ndarray) -> None:
    keyless = _marker(None).embed(large_photo, OWNER)
    keyed = _marker(OWNER_KEY).embed(large_photo, OWNER)
    assert not np.array_equal(keyless, keyed)


def test_different_keys_give_different_layouts() -> None:
    owner, forger = derive_layout(OWNER_KEY), derive_layout(FORGER_KEY)
    assert (owner.coefficient_a, owner.permutation.tobytes()) != (
        forger.coefficient_a,
        forger.permutation.tobytes(),
    )


@pytest.mark.parametrize("pair", CARRIER_PAIRS)
def test_every_carrier_pair_round_trips(large_photo: np.ndarray, pair: tuple) -> None:
    """A key may select any carrier, so every carrier has to decode."""
    keys = [k for k in (f"k{i}" for i in range(200)) if derive_layout(k).coefficient_a == pair[0]]
    if not keys:
        pytest.skip(f"no probe key selects {pair}")
    marker = _marker(keys[0])
    result = marker.detect(marker.embed(large_photo, OWNER))
    assert result.detected is True
    assert result.watermark_code == _code(OWNER)


def test_a_keyed_mark_reads_back(large_photo: np.ndarray) -> None:
    marker = _marker(OWNER_KEY)
    result = marker.detect(marker.embed(large_photo, OWNER))
    assert result.detected is True
    assert result.watermark_code == _code(OWNER)


def test_the_wrong_key_reads_nothing(large_photo: np.ndarray) -> None:
    marked = _marker(OWNER_KEY).embed(large_photo, OWNER)
    assert _marker(FORGER_KEY).detect(marked).detected is False


# --- the resynchronisation the key must not break ---------------------------


@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.3])
def test_a_keyed_mark_still_survives_cropping(large_photo: np.ndarray, ratio: float) -> None:
    """The tile permutation is applied after the phase roll, not instead of it.

    The roll enumerates crop offsets in slot-position space; the key permutes
    position to message bit. They compose, and this is the measurement that
    says so rather than the argument that says so.
    """
    marker = _marker(OWNER_KEY)
    marked = marker.embed(large_photo, OWNER)
    result = marker.detect(_transform(marked, "crop", {"ratio": ratio}))
    assert result.detected is True
    assert result.watermark_code == _code(OWNER)


def test_a_keyed_mark_still_survives_rotation(large_photo: np.ndarray) -> None:
    """The angle search still resynchronises when a key has moved the mark.

    A key costs the angle search a little, so the cost is measured rather than
    asserted. ``scripts/evaluate_watermark_key_rotation.py`` reads twenty LFW
    photographs back for eight keys and the keyless layout: at 250 pixels every
    key recovers five and ten degrees, and at 512 the median key still does
    while the worst gives up one photograph in twenty at five degrees.

    This fixture is synthetic texture rather than a photograph and it is much
    thinner than that - sampling twenty-four keys, five degrees recovers for
    nineteen at 250 pixels and twenty-one at 512. Two degrees is the angle that
    holds for every key sampled against it, so that is what is asserted, and the
    limit is written down rather than hidden by choosing a fixture that passes.
    """
    marker = _marker(OWNER_KEY)
    marked = marker.embed(large_photo, OWNER)
    result = marker.detect(_transform(marked, "rotation", {"degrees": 2.0}))
    assert result.detected is True
    assert result.watermark_code == _code(OWNER)


def test_a_keyed_mark_costs_the_same_quality(large_photo: np.ndarray) -> None:
    keyless = psnr(large_photo, _marker(None).embed(large_photo, OWNER))
    keyed = psnr(large_photo, _marker(OWNER_KEY).embed(large_photo, OWNER))
    assert abs(keyed - keyless) < 2.0


def test_a_key_does_not_invent_codes_on_unmarked_images(large_photo: np.ndarray) -> None:
    assert _marker(OWNER_KEY).detect(large_photo).detected is False


# --- success criteria 1, 2 and 3 --------------------------------------------


def test_an_attacker_without_the_key_cannot_erase(large_photo: np.ndarray) -> None:
    """Criterion 1. The keyless mark loses this 60 times out of 60."""
    owner = _marker(OWNER_KEY)
    marked = owner.embed(large_photo, OWNER)
    attacked = _equalise(marked, FORGER_KEY)
    result = owner.detect(attacked)
    assert result.detected is True
    assert result.watermark_code == _code(OWNER)


def test_an_attacker_without_the_key_cannot_forge(large_photo: np.ndarray) -> None:
    """Criterion 2. The keyless mark reads the forger's code 60 times out of 60."""
    owner = _marker(OWNER_KEY)
    marked = owner.embed(large_photo, OWNER)
    overwritten = _marker(FORGER_KEY).embed(marked, FORGER)
    assert owner.detect(overwritten).watermark_code != _code(FORGER)


def test_an_attacker_with_the_key_still_forges(large_photo: np.ndarray) -> None:
    """Criterion 3, the forgery control.

    Without it the test above proves nothing: an attack that has stopped working
    attacks nothing, and would pass while the key did no work. Re-embedding with
    the owner's key succeeds on every carrier, which is what makes the wrong-key
    failure attributable to the key.
    """
    owner = _marker(OWNER_KEY)
    marked = owner.embed(large_photo, OWNER)
    forged = owner.detect(_marker(OWNER_KEY).embed(marked, FORGER))
    assert forged.detected is True
    assert forged.watermark_code == _code(FORGER)


def test_an_attacker_with_the_key_still_erases(large_photo: np.ndarray) -> None:
    """Criterion 3, the removal control - and it needs a carrier the attack clears.

    Equalising is not a clean erasure. It sets the two coefficients equal in
    floating point, and the round trip back through uint8 RGB leaves a residual
    that still correlates with the bit it was meant to destroy. On sixty LFW
    photographs a correctly-keyed removal left something the detector still
    called a watermark 45 per cent of the time, while the owner's code came back
    0 per cent of the time. Attribution turns on the code, and the code is gone.

    How much residual survives depends on the carrier, so this control fixes one
    rather than taking whatever a key happens to choose. The per-carrier numbers
    belong in the measurement scripts, not in an assertion here.
    """
    key = next(k for k in (f"c{i}" for i in range(500)) if derive_layout(k).coefficient_a == (1, 5))
    owner = _marker(key)
    marked = owner.embed(large_photo, OWNER)
    assert owner.detect(_equalise(marked, key)).watermark_code != _code(OWNER)
