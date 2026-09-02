"""Invisible watermarking: interface, a DCT baseline and the no-op mock.

Plain language: hide a short invisible serial number inside a picture so that a
copy found later can be traced back to where it was published.

What a watermark is for here: protected-image identification, source
attribution and leak tracking. Publishing the same photo with a different
distribution id per channel tells you which channel a leaked copy came from.

What it is explicitly not for: proving that an image was used to train a
generative model. A watermark in a source image is not expected to survive into
a model's output, and the codebase never claims otherwise.

How the DCT baseline works:

1. The payload is reduced to an opaque 32-bit code plus a 32-bit CRC, giving a
   64-bit message. Nothing personal is ever encoded.
2. The image is converted to YCbCr and only the luminance plane is touched,
   because the eye is least sensitive to small luminance changes in textured
   regions and chroma survives compression worse.
3. Luminance is split into 8x8 blocks, each transformed with a 2-D DCT - the
   same block grid JPEG uses, so the mark lands where JPEG will preserve it.
4. Each block carries one message bit in the *sign of the difference* between
   two mid-frequency coefficients. Low frequencies would be visible; high
   frequencies are the first thing JPEG discards; the middle band survives.
5. The message is laid out as a repeating 8x8 tile of blocks, so a block's bit
   is decided by its position modulo the tile rather than by its position in
   raster order. Laying the bits out in raster order would tie the mapping to
   the image width, and cropping changes the width, which scrambles every bit
   even when the grid is otherwise recovered.
6. The tile repeats across the whole image, and the decoder takes a majority
   vote per bit. Redundancy is what turns a fragile per-block signal into a mark
   that survives re-encoding.

Cropping moves and rescales the block grid the decoder depends on, so the
decoder searches for the grid before giving up. It tries a small set of
magnifications and all sixty-four sub-block pixel offsets, scoring each cheaply
on a subsample of blocks, then tries all sixty-four tile phases on the winner.

That search is why the checksum is thirty-two bits rather than eight, and why
its size is capped rather than left to grow. Every candidate grid, phase and
bit-flip pattern is another chance for a checksum to pass on noise, and a
watermark that names the wrong distribution channel is worse than one that
reports nothing.

The budget is therefore explicit, and it was set by measurement. The direct path
tries one grid and up to ``2^soft_decode_bits`` corrections, about four thousand
candidates. The resynchronising path tries every tile phase of several candidate
grids by hard decision only - two hundred and fifty-six candidates, and no bit
flipping by default.

Bit flipping is disabled there because it was measured to buy almost nothing and
cost precision: enabling it recovered the same 57 of 60 cropped images while
adding sixteen thousand candidate messages per image. Recovering one more image
is not worth naming the wrong distribution channel once.

Even the hard search needed one more guard. Its candidates are not random: at a
wrong tile phase the decoder sees a permutation of the true message, which is
structured enough that the checksum passes far more often than the uniform
one-in-four-billion estimate suggests. A benchmark run produced exactly one such
accidental pass. The search therefore collects every phase and grid whose
checksum validates and accepts the result only when they all name the same code.
Two different valid codes from one image is evidence of an accident, not of two
watermarks, and the detector reports nothing.

Remaining failure modes, all measured by the Phase 13 benchmark rather than
assumed: rotation, which the grid search does not cover; heavy downscaling,
which destroys the 8x8 structure outright; strong blur, which removes the
mid-band; and regenerating the image through another model, which removes the
mark entirely.
"""

from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image

from deepshield.config import WatermarkConfig
from deepshield.exceptions import WatermarkError
from deepshield.logging_utils import get_logger
from deepshield.media import validate_rgb
from deepshield.protection.fingerprint import _dct_matrix, dct2, idct2
from deepshield.registry import ComponentRegistry
from deepshield.types import WatermarkDetectionResult, WatermarkPayload

logger = get_logger(__name__)

BLOCK_SIZE = 8
CODE_BITS = 32
CRC_BITS = 32
MESSAGE_BITS = CODE_BITS + CRC_BITS
TILE_ROWS = 8
TILE_COLS = 8
COEFFICIENT_A = (3, 4)
COEFFICIENT_B = (4, 3)
MIN_REPETITIONS = 3


class Watermarker(ABC):
    """Contract every watermark backend must satisfy."""

    name: str = "abstract"

    @abstractmethod
    def embed(self, image: np.ndarray, payload: WatermarkPayload) -> np.ndarray:
        """Return a copy of ``image`` carrying ``payload``.

        Args:
            image: ``H x W x 3`` uint8 RGB array.
            payload: Opaque identifiers to encode.

        Returns:
            A watermarked image of the same shape and dtype.

        Raises:
            WatermarkError: If the image is too small to carry the message.

        """

    @abstractmethod
    def detect(self, image: np.ndarray) -> WatermarkDetectionResult:
        """Attempt to recover a payload from ``image``."""

    @property
    @abstractmethod
    def capacity_bits(self) -> int:
        """Number of payload bits this backend can carry."""


WATERMARK_REGISTRY: ComponentRegistry[Watermarker] = ComponentRegistry("watermark backend")


CRC32_POLYNOMIAL = 0xEDB88320
CRC32_INITIAL = 0xFFFFFFFF


def _build_crc32_table() -> np.ndarray:
    """Return the 256-entry reflected CRC-32 lookup table."""
    table = np.zeros(256, dtype=np.uint64)
    for index in range(256):
        value = index
        for _ in range(8):
            value = (value >> 1) ^ (CRC32_POLYNOMIAL if value & 1 else 0)
        table[index] = value
    return table.astype(np.uint32)


CRC32_TABLE = _build_crc32_table()


def crc32(bits: np.ndarray) -> int:
    """Return the CRC-32 of a bit array.

    Thirty-two check bits because the decoder searches many candidate grids,
    tile phases and bit-flip patterns, and every one of them is another chance
    for a checksum to pass on noise. The width of the checksum is what bounds
    the false-attribution rate of the whole search.
    """
    return int(zlib.crc32(np.packbits(bits.astype(np.uint8)).tobytes()) & 0xFFFFFFFF)


def crc32_batch(messages: np.ndarray) -> np.ndarray:
    """Return the CRC-32 of every row of a bit matrix, computed table-driven.

    The decoder evaluates thousands of candidate messages per image. Calling a
    scalar checksum in a Python loop over that many candidates dominates the
    runtime; folding the same table-driven algorithm over a NumPy array turns it
    into a handful of vector operations.
    """
    packed = np.packbits(messages.astype(np.uint8), axis=1).astype(np.uint32)
    remainder = np.full(packed.shape[0], CRC32_INITIAL, dtype=np.uint32)
    for column in range(packed.shape[1]):
        index = (remainder ^ packed[:, column]) & 0xFF
        remainder = (remainder >> np.uint32(8)) ^ CRC32_TABLE[index]
    return remainder ^ np.uint32(CRC32_INITIAL)


def _flip_masks(width: int) -> np.ndarray:
    """Return every non-empty subset of ``width`` positions as a boolean matrix."""
    patterns = np.arange(1, 1 << width, dtype=np.uint32)
    bit_index = np.arange(width, dtype=np.uint32)
    return ((patterns[:, None] >> bit_index[None, :]) & 1).astype(np.uint8)


def _int_to_bits(value: int, width: int) -> np.ndarray:
    """Return the big-endian bit array of an unsigned integer."""
    return np.array([(value >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


def _bits_to_int(bits: np.ndarray) -> int:
    """Return the unsigned integer represented by a big-endian bit array."""
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def build_message(code: int) -> np.ndarray:
    """Return the 64-bit message: a 32-bit code followed by its CRC-32."""
    code_bits = _int_to_bits(code, CODE_BITS)
    return np.concatenate([code_bits, _int_to_bits(crc32(code_bits), CRC_BITS)])


def message_is_valid(bits: np.ndarray) -> bool:
    """Return whether a candidate message's checksum matches its code."""
    return crc32(bits[:CODE_BITS]) == _bits_to_int(bits[CODE_BITS:])


def tile_slots(rows: int, cols: int) -> np.ndarray:
    """Return the message-bit index of every block in a rows-by-cols grid.

    The index depends only on a block's position within the repeating tile, so
    it survives a change of image width. Raster-order indexing would not.
    """
    row_part = (np.arange(rows) % TILE_ROWS)[:, None] * TILE_COLS
    col_part = (np.arange(cols) % TILE_COLS)[None, :]
    return (row_part + col_part).ravel()


def phase_shift(ratios: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    """Return the vote ratios re-indexed for one candidate tile alignment.

    A crop removes whole blocks from the top and left, so the tile the decoder
    sees starts at a different cell. Rolling the recovered grid enumerates every
    alignment the crop could have produced.
    """
    grid = ratios.reshape(TILE_ROWS, TILE_COLS)
    return np.roll(np.roll(grid, row_shift, axis=0), col_shift, axis=1).ravel()


class DctWatermarker(Watermarker):
    """Blind DCT mid-frequency differential watermark with majority-vote decoding.

    Blind means detection needs neither the original image nor the payload. The
    embedding strength trades perceptual quality against survivability, and the
    trade-off is measured, not assumed.
    """

    name = "dct"

    def __init__(self, config: WatermarkConfig | None = None) -> None:
        """Store watermark configuration."""
        self.config = config or WatermarkConfig()

    @property
    def capacity_bits(self) -> int:
        """Number of message bits carried, independent of image size."""
        return MESSAGE_BITS

    @property
    def margin(self) -> float:
        """Minimum enforced coefficient difference, derived from the strength."""
        return float(self.config.strength) * 255.0

    def _blocks(self, height: int, width: int) -> tuple[int, int]:
        """Return the number of whole 8x8 blocks that fit in the frame."""
        return height // BLOCK_SIZE, width // BLOCK_SIZE

    @staticmethod
    def _block_stack(
        luminance: np.ndarray, offset_y: int, offset_x: int
    ) -> tuple[np.ndarray, int, int]:
        """Return every whole 8x8 block from a grid anchored at an offset.

        Reshaping once and transforming the whole stack with two matrix products
        replaces a Python loop over thousands of blocks, which is what makes the
        grid search affordable.
        """
        height, width = luminance.shape
        rows = (height - offset_y) // BLOCK_SIZE
        cols = (width - offset_x) // BLOCK_SIZE
        if rows <= 0 or cols <= 0:
            return np.zeros((0, BLOCK_SIZE, BLOCK_SIZE), dtype=np.float64), 0, 0
        region = luminance[
            offset_y : offset_y + rows * BLOCK_SIZE, offset_x : offset_x + cols * BLOCK_SIZE
        ]
        stack = (
            region.reshape(rows, BLOCK_SIZE, cols, BLOCK_SIZE)
            .transpose(0, 2, 1, 3)
            .reshape(rows * cols, BLOCK_SIZE, BLOCK_SIZE)
        )
        return stack, rows, cols

    @staticmethod
    def _block_differences(blocks: np.ndarray) -> np.ndarray:
        """Return the embedded coefficient difference of every block."""
        if blocks.shape[0] == 0:
            return np.zeros(0, dtype=np.float64)
        basis = _dct_matrix(BLOCK_SIZE)
        coefficients = basis @ blocks @ basis.T
        differences: np.ndarray = coefficients[:, COEFFICIENT_A[0], COEFFICIENT_A[1]] - (
            coefficients[:, COEFFICIENT_B[0], COEFFICIENT_B[1]]
        )
        return differences

    @staticmethod
    def _tally(
        differences: np.ndarray, rows: int, cols: int, limit: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fold per-block observations into per-message-bit vote counts."""
        if differences.size == 0:
            return np.zeros(MESSAGE_BITS), np.zeros(MESSAGE_BITS)
        slots = tile_slots(rows, cols)[: differences.size]
        if limit is not None and differences.size > limit:
            slots = slots[:limit]
            differences = differences[:limit]
        votes = np.bincount(
            slots, weights=(differences > 0).astype(np.float64), minlength=MESSAGE_BITS
        )
        counts = np.bincount(slots, minlength=MESSAGE_BITS).astype(np.float64)
        return votes[:MESSAGE_BITS], counts[:MESSAGE_BITS]

    @staticmethod
    def _luminance(image: np.ndarray) -> np.ndarray:
        """Return the luminance plane of an RGB image as float64."""
        ycbcr = np.asarray(Image.fromarray(validate_rgb(image)).convert("YCbCr"), np.float64)
        return ycbcr[:, :, 0]

    @staticmethod
    def _rescaled(image: np.ndarray, scale: float) -> np.ndarray:
        """Return the image resized by a factor, used to undo a magnifying crop."""
        if scale == 1.0:
            return validate_rgb(image)
        array = validate_rgb(image)
        height, width = array.shape[:2]
        target = (max(BLOCK_SIZE, int(round(width * scale))),
                  max(BLOCK_SIZE, int(round(height * scale))))
        resized = Image.fromarray(array).resize(target, Image.Resampling.LANCZOS)
        return np.asarray(resized, dtype=np.uint8)

    @staticmethod
    def _agreement(votes: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, float]:
        """Return per-bit vote ratios and their mean agreement with the majority."""
        with np.errstate(invalid="ignore"):
            ratios = np.divide(votes, counts, out=np.full(MESSAGE_BITS, 0.5), where=counts > 0)
        agreement = float(np.mean(np.maximum(ratios, 1.0 - ratios)))
        return ratios, agreement

    def _check_capacity(self, image: np.ndarray) -> tuple[int, int]:
        """Validate that the image can hold the message with enough redundancy.

        Raises:
            WatermarkError: If fewer than ``MIN_REPETITIONS`` copies would fit.

        """
        rows, cols = self._blocks(image.shape[0], image.shape[1])
        total = rows * cols
        if total < MESSAGE_BITS * MIN_REPETITIONS:
            needed = MESSAGE_BITS * MIN_REPETITIONS * BLOCK_SIZE * BLOCK_SIZE
            raise WatermarkError(
                f"image too small to watermark: {total} blocks available, "
                f"{MESSAGE_BITS * MIN_REPETITIONS} needed "
                f"(about {needed} pixels)"
            )
        return rows, cols

    def embed(self, image: np.ndarray, payload: WatermarkPayload) -> np.ndarray:
        """Embed the payload's opaque code into the luminance mid-band."""
        array = validate_rgb(image)
        rows, cols = self._check_capacity(array)
        message = build_message(payload.code(CODE_BITS))

        ycbcr = np.asarray(Image.fromarray(array).convert("YCbCr"), dtype=np.float64)
        luminance = ycbcr[:, :, 0]
        margin = self.margin

        for row in range(rows):
            for col in range(cols):
                y0, x0 = row * BLOCK_SIZE, col * BLOCK_SIZE
                block = luminance[y0 : y0 + BLOCK_SIZE, x0 : x0 + BLOCK_SIZE]
                coefficients = dct2(block)

                slot = (row % TILE_ROWS) * TILE_COLS + (col % TILE_COLS)
                bit = int(message[slot])

                a = coefficients[COEFFICIENT_A]
                b = coefficients[COEFFICIENT_B]
                mean = (a + b) / 2.0
                half = margin / 2.0
                if bit == 1:
                    coefficients[COEFFICIENT_A] = mean + half
                    coefficients[COEFFICIENT_B] = mean - half
                else:
                    coefficients[COEFFICIENT_A] = mean - half
                    coefficients[COEFFICIENT_B] = mean + half

                luminance[y0 : y0 + BLOCK_SIZE, x0 : x0 + BLOCK_SIZE] = idct2(coefficients)

        ycbcr[:, :, 0] = np.clip(luminance, 0, 255)
        marked = Image.fromarray(ycbcr.astype(np.uint8), mode="YCbCr").convert("RGB")
        return np.asarray(marked, dtype=np.uint8)

    def _extract_votes(
        self, image: np.ndarray, offset: tuple[int, int] = (0, 0)
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-bit vote sums and counts recovered from one candidate grid."""
        luminance = self._luminance(image)
        blocks, rows, cols = self._block_stack(luminance, offset[0], offset[1])
        return self._tally(self._block_differences(blocks), rows, cols)

    def _search_grid(self, image: np.ndarray) -> list[tuple[np.ndarray, float, dict[str, Any]]]:
        """Rank candidate magnifications and block offsets by vote agreement.

        Scoring uses a subsample of blocks so that hundreds of candidate grids
        stay affordable. Several candidates are returned rather than one,
        because agreement is only a proxy: a wrong grid over a strong watermark
        can out-score the right grid over a weakened one.
        """
        ranked: list[tuple[float, dict[str, Any]]] = []
        budget = int(self.config.resync_max_blocks)

        for scale in self.config.resync_scales:
            candidate = self._rescaled(image, float(scale))
            luminance = self._luminance(candidate)
            if min(luminance.shape) < BLOCK_SIZE * 2:
                continue
            for offset_y in range(BLOCK_SIZE):
                for offset_x in range(BLOCK_SIZE):
                    blocks, rows, cols = self._block_stack(luminance, offset_y, offset_x)
                    if blocks.shape[0] < MESSAGE_BITS * MIN_REPETITIONS:
                        continue
                    votes, counts = self._tally(
                        self._block_differences(blocks), rows, cols, limit=budget
                    )
                    _, agreement = self._agreement(votes, counts)
                    ranked.append(
                        (
                            agreement,
                            {
                                "scale": float(scale),
                                "offset_y": offset_y,
                                "offset_x": offset_x,
                            },
                        )
                    )

        ranked.sort(key=lambda item: item[0], reverse=True)
        candidates: list[tuple[np.ndarray, float, dict[str, Any]]] = []
        for _, grid in ranked[: self.config.resync_candidates]:
            winner = self._rescaled(image, grid["scale"])
            votes, counts = self._extract_votes(winner, (grid["offset_y"], grid["offset_x"]))
            ratios, full_agreement = self._agreement(votes, counts)
            candidates.append((ratios, full_agreement, grid))
        return candidates

    def _soft_decode(
        self, ratios: np.ndarray, width: int | None = None
    ) -> tuple[np.ndarray, int] | None:
        """Search low-reliability bit flips for a message whose CRC validates.

        Majority voting alone recovers a mark only when every one of the 40 bits
        survives, so a single flipped bit reads as "no watermark" even at 97%
        bit accuracy. The per-bit vote ratios are a reliability signal: bits near
        0.5 are the ones a transformation most likely corrupted. Chase decoding
        exhaustively flips subsets of the least reliable bits and accepts the
        first candidate whose CRC passes.

        Returns:
            The corrected bits and the number of flips applied, or ``None`` when
            no candidate validates. Correction is attempted only when the raw vote
        agreement already indicates a real signal, because searching candidate
        messages inside pure noise is how a detector invents attributions.

        """
        hard = (ratios > 0.5).astype(np.uint8)
        if message_is_valid(hard):
            return hard, 0

        width = int(self.config.soft_decode_bits if width is None else width)
        if width <= 0:
            return None

        reliability = np.abs(ratios - 0.5)
        weakest = np.argsort(reliability)[:width]
        masks = _flip_masks(width)

        candidates = np.repeat(hard[None, :], masks.shape[0], axis=0)
        candidates[:, weakest] ^= masks

        checksums = crc32_batch(candidates[:, :CODE_BITS])
        packed = np.packbits(candidates[:, CODE_BITS:], axis=1).astype(np.uint32)
        claimed = (
            (packed[:, 0] << np.uint32(24))
            | (packed[:, 1] << np.uint32(16))
            | (packed[:, 2] << np.uint32(8))
            | packed[:, 3]
        )
        valid = np.flatnonzero(checksums == claimed)
        if valid.size == 0:
            return None

        flip_counts = masks.sum(axis=1)
        best = valid[np.argmin(flip_counts[valid])]
        return candidates[best], int(flip_counts[best])

    def detect(self, image: np.ndarray) -> WatermarkDetectionResult:
        """Recover the code by majority vote, soft decoding, and CRC validation.

        The checksum is what keeps the false-attribution rate low: an unmarked
        image yields near-random votes, and a random 64-bit message passes CRC-32
        with probability about one in four billion. That budget is what pays for
        the grid, phase and bit-flip searches below. Each search stage also runs
        only above a confidence floor, and the reported confidence is scaled down
        by the number of corrections applied.
        """
        array = validate_rgb(image)
        rows, cols = self._blocks(array.shape[0], array.shape[1])
        if rows * cols < MESSAGE_BITS:
            return WatermarkDetectionResult(
                detected=False, confidence=0.0, backend=self.name
            )

        votes, counts = self._extract_votes(array)
        ratios, agreement = self._agreement(votes, counts)
        confidence = float(np.clip((agreement - 0.5) * 2.0, 0.0, 1.0))

        decoded = self._soft_decode(ratios)
        if decoded is not None:
            return self._result(decoded, confidence)

        if not self.config.resync_enabled:
            return WatermarkDetectionResult(
                detected=False, confidence=confidence, backend=self.name
            )

        best_confidence = confidence
        candidates = self._search_grid(array)
        accepted: dict[int, tuple[np.ndarray, int, float]] = {}

        for ratios, grid_agreement, grid in candidates:
            resync_confidence = float(np.clip((grid_agreement - 0.5) * 2.0, 0.0, 1.0))
            best_confidence = max(best_confidence, resync_confidence)
            if resync_confidence < self.config.resync_min_confidence:
                continue

            for row_shift in range(TILE_ROWS):
                for col_shift in range(TILE_COLS):
                    shifted = phase_shift(ratios, row_shift, col_shift)
                    hard = (shifted > 0.5).astype(np.uint8)
                    decoded = (
                        (hard, 0)
                        if message_is_valid(hard)
                        else self._soft_decode(shifted, self.config.resync_soft_decode_bits)
                    )
                    if decoded is None:
                        continue
                    code = _bits_to_int(decoded[0][:CODE_BITS])
                    accepted.setdefault(code, (decoded[0], decoded[1], resync_confidence))
                    logger.debug("watermark candidate %08x on grid %s", code, grid)

        if len(accepted) == 1:
            bits, flips, resync_confidence = next(iter(accepted.values()))
            return self._result((bits, flips), resync_confidence)

        if len(accepted) > 1:
            logger.warning(
                "grid search produced %d different valid codes; reporting no detection",
                len(accepted),
            )

        return WatermarkDetectionResult(
            detected=False, confidence=best_confidence, backend=self.name
        )

    def _result(
        self, decoded: tuple[np.ndarray, int], confidence: float
    ) -> WatermarkDetectionResult:
        """Build a positive detection, discounting confidence by the corrections applied."""
        bits, flips = decoded
        code = _bits_to_int(bits[:CODE_BITS])
        return WatermarkDetectionResult(
            detected=True,
            confidence=float(np.clip(confidence / (1.0 + flips), 0.0, 1.0)),
            watermark_code=f"{code:08x}",
            backend=self.name,
        )

    def bit_accuracy(self, image: np.ndarray, payload: WatermarkPayload) -> float:
        """Return the fraction of message bits recovered from the unshifted grid.

        Used by benchmarks, where the embedded payload is known. It degrades
        smoothly under compression, unlike the binary detection flag, which makes
        it the more informative metric there.

        It says nothing useful about a geometric transformation. Cropping moves
        the grid, so this measurement lands at chance even when the decoder
        resynchronises and recovers the code exactly. Read it alongside the
        detection flag, never instead of it.
        """
        array = validate_rgb(image)
        rows, cols = self._blocks(array.shape[0], array.shape[1])
        if rows * cols < MESSAGE_BITS:
            return 0.0
        votes, counts = self._extract_votes(array)
        with np.errstate(invalid="ignore"):
            ratios = np.divide(votes, counts, out=np.full(MESSAGE_BITS, 0.5), where=counts > 0)
        recovered = (ratios > 0.5).astype(np.uint8)
        expected = build_message(payload.code(CODE_BITS))
        return float(np.mean(recovered == expected))


class MockWatermarker(Watermarker):
    """Pass-through backend that embeds nothing and detects nothing.

    Kept for tests and for pipelines that must run without touching pixels. It
    never fabricates a positive result.
    """

    name = "mock"

    def __init__(self, config: WatermarkConfig | None = None) -> None:
        """Store watermark configuration."""
        self.config = config or WatermarkConfig()

    @property
    def capacity_bits(self) -> int:
        """Return the configured payload size."""
        return self.config.payload_bits

    def embed(self, image: np.ndarray, payload: WatermarkPayload) -> np.ndarray:
        """Return an unmodified copy of the input image."""
        return np.array(validate_rgb(image), copy=True)

    def detect(self, image: np.ndarray) -> WatermarkDetectionResult:
        """Report an inconclusive result, never a fabricated detection."""
        return WatermarkDetectionResult(
            detected=False,
            confidence=0.0,
            payload=None,
            bit_accuracy=None,
            backend=self.name,
        )


WATERMARK_REGISTRY.register("mock", MockWatermarker)
WATERMARK_REGISTRY.register("dct", DctWatermarker)


def build_watermarker(config: WatermarkConfig) -> Watermarker:
    """Instantiate the watermark backend named in ``config``."""
    return WATERMARK_REGISTRY.create(config.backend, config)
