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
flipping by default. The rotation search that follows adds sixty-four phases for
each of a few offsets at each of its two refined angles, and only runs when the
crop search found nothing.

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

Rotation is handled by a second search that runs only when the first one has
found nothing. Rotating a marked image back by the exact angle restores every
bit - the interpolation costs nothing the mid-band cannot absorb - so the whole
problem is estimating the angle. The decoder sweeps candidate angles one degree
apart over a bounded range, scores each by the same vote agreement the crop
search uses, then refines the best few to a quarter of a degree. The tolerance
was measured before the steps were chosen: half a degree of error still decodes
exactly, and three quarters does not, so a one-degree coarse grid always holds
a candidate that decodes. The sweep runs at unit scale first; when that finds
nothing, the image is shrunk back by each of a few candidate scales and swept
again, which covers rotation combined with a magnifying crop at the cost of one
more sweep per scale.

A key changes what the layout is, not how it is read. Without one, every block
carries its bit on the same coefficient pair and the tile maps slots to bits in
raster order, which is why an attacker who has read this file can erase the mark
by equalising that pair (measured: 100% at 36 dB) or forge a different code
(100% at 34 dB). With a key, a hash of the secret seeds a permutation of the
64 bit positions and a choice among five carrier pairs per slot. The decoder
then has to read each candidate grid under all 64 tile alignments, because the
alignment decides which pair a block is read on. Read one alignment at a time
that cost sixty-four times the keyless search; folding the blocks by tile cell
once and gathering every alignment from that 64-by-64 table brings it to about
a third more, with identical votes. The keyless layout is the schedule a
missing key derives, and every keyless number is unchanged.

A key can also change what a carrier is. A pair is the smallest carrier: the
bit is the sign of one coefficient minus another, and an attacker who knows
the five candidate pairs can flatten all five in every block and be done. With
``carrier_coefficients`` above two, each tile slot instead carries its bit in
the sign of a keyed sum of that many mid-band coefficients, each with a keyed
sign, drawn from the eleven coefficients on the sixth and seventh diagonals.
The pair is the special case of two coefficients with opposite signs, so the
decoder is the same projection-and-vote in both cases. What the attacker loses
is the ability to aim: without the key there is no pair to equalise, only a
band to blank or to drown in noise. The removal benchmark measured what that
buys, and it is less than it sounds: flattening the five candidate pairs no
longer removes the mark, but blanking the eleven-coefficient band still does,
at about the PSNR the mark itself cost, because the band holds only about a
third of the mark's energy in natural content (measured: blanking it in an
unmarked 250-pixel portrait costs 40.6 dB, the mark 36.2 dB). A spread carrier also needs a larger
strength for the same JPEG survival (its change per coefficient is smaller),
which returns it to the pair's PSNR. Removal cost is bounded by the mark's
energy plus what the band held already, and no carrier choice moves that.

Remaining failure modes, all measured by the Phase 13 benchmark rather than
assumed: rotation beyond the searched range, or combined with a crop deeper
than the searched scales; heavy
downscaling, which destroys the 8x8 structure outright; strong blur, which
removes the mid-band; and regenerating the image through another model, which
removes the mark entirely.
"""

from __future__ import annotations

import hashlib
import zlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
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
TILE_SLOTS = TILE_ROWS * TILE_COLS
# Transposed mid-band coefficient pairs with similar JPEG quantisation steps
# (51/56, 57/55, 40/37, 60/49, 58/35 in the standard luminance table). The
# keyless layout uses the first for every block; a key picks one per tile slot.
PAIR_CANDIDATES: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((3, 4), (4, 3)),
    ((2, 5), (5, 2)),
    ((2, 4), (4, 2)),
    ((1, 6), (6, 1)),
    ((1, 5), (5, 1)),
)
COEFFICIENT_A, COEFFICIENT_B = PAIR_CANDIDATES[0]
# The band a keyed spread carrier draws from: every coefficient on the sixth
# and seventh anti-diagonals, which is where the candidate pairs live.
CARRIER_BAND: tuple[tuple[int, int], ...] = tuple(
    (i, j) for i in range(1, 7) for j in range(1, 7) if i + j in (6, 7)
)
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


def tile_slots(rows: int, cols: int, row_shift: int = 0, col_shift: int = 0) -> np.ndarray:
    """Return the tile slot of every block in a rows-by-cols grid.

    The slot depends only on a block's position within the repeating tile, so
    it survives a change of image width. Raster-order indexing would not. A
    shift names the tile alignment a crop could have produced: the block the
    decoder sees at row ``r`` sat at row ``r + row_shift`` of the original tile.
    Without a key the slot is the message-bit index; with one it is looked up
    through the :class:`KeySchedule`.
    """
    row_part = ((np.arange(rows) + row_shift) % TILE_ROWS)[:, None] * TILE_COLS
    col_part = ((np.arange(cols) + col_shift) % TILE_COLS)[None, :]
    return (row_part + col_part).ravel()


def phase_shift(ratios: np.ndarray, row_shift: int, col_shift: int) -> np.ndarray:
    """Return the vote ratios re-indexed for one candidate tile alignment.

    A crop removes whole blocks from the top and left, so the tile the decoder
    sees starts at a different cell. Rolling the recovered grid enumerates every
    alignment the crop could have produced.
    """
    grid = ratios.reshape(TILE_ROWS, TILE_COLS)
    return np.roll(np.roll(grid, row_shift, axis=0), col_shift, axis=1).ravel()


def _phase_slot_table() -> np.ndarray:
    """Return, for every tile alignment, the slot each tile cell is read as.

    Row ``p`` is the alignment ``(p // TILE_COLS, p % TILE_COLS)`` and column
    ``c`` is a block's own cell in the tile; the entry is the slot that block
    is read as under that alignment. Every row is a permutation of the cells,
    which is what lets the decoder tally all 64 alignments in one pass.
    """
    table = np.zeros((TILE_SLOTS, TILE_SLOTS), dtype=np.intp)
    for row_shift in range(TILE_ROWS):
        for col_shift in range(TILE_COLS):
            table[row_shift * TILE_COLS + col_shift] = tile_slots(
                TILE_ROWS, TILE_COLS, row_shift, col_shift
            )
    return table


PHASE_SLOT = _phase_slot_table()


def _pair_carrier(pair: tuple[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    """Return a coefficient pair as a carrier: +1 on the first, -1 on the second."""
    carrier = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float64)
    carrier[pair[0]] = 1.0
    carrier[pair[1]] = -1.0
    return carrier


@dataclass(frozen=True)
class KeySchedule:
    """What each of the 64 tile slots carries: which message bit, on which carrier.

    ``bit_of_slot`` is a permutation of the message bits. ``carriers`` holds
    every distinct carrier as an 8x8 pattern of +1, -1 and 0 over the DCT
    coefficients, and ``carrier_of_slot`` says which one a slot reads on; a
    block's bit is the sign of the sum of its coefficients weighted by the
    pattern. ``pair_of_slot`` indexes :data:`PAIR_CANDIDATES` when the
    carriers are pairs and is all zeros otherwise. Without a key everything is
    the fixed keyless layout, so every number the keyless design ever produced
    is unchanged.
    """

    bit_of_slot: np.ndarray
    pair_of_slot: np.ndarray
    carriers: np.ndarray
    carrier_of_slot: np.ndarray
    keyed: bool
    carrier_size: int


def derive_schedule(key: str | None, carrier_size: int = 2) -> KeySchedule:
    """Derive the tile layout from a secret, or return the keyless layout.

    The secret is hashed and the digest seeds a PCG64 generator, whose bit
    stream numpy keeps stable across versions, so a key always yields the
    same layout. An attacker who knows the algorithm but not the key faces a
    64! permutation of bit positions and, with pairs, five carrier pairs per
    slot: forging a message that passes the checksum under an unknown
    permutation is not a search, and erasing the mark means flattening every
    candidate pair in every block rather than one.

    ``carrier_size`` above two replaces the pair with a keyed spread carrier:
    that many coefficients of :data:`CARRIER_BAND`, each with a keyed sign,
    chosen per slot. The bit-position permutation is drawn first, so a key
    maps the same bits to the same slots whichever carrier size it is used
    with. Without a key the carrier size is ignored: a spread carrier that
    everyone knows would only be a more expensive pair.
    """
    if not key:
        return KeySchedule(
            bit_of_slot=np.arange(TILE_SLOTS, dtype=np.intp),
            pair_of_slot=np.zeros(TILE_SLOTS, dtype=np.intp),
            carriers=_pair_carrier(PAIR_CANDIDATES[0])[None],
            carrier_of_slot=np.zeros(TILE_SLOTS, dtype=np.intp),
            keyed=False,
            carrier_size=2,
        )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:16], "big"))
    bit_of_slot = generator.permutation(TILE_SLOTS).astype(np.intp)
    if carrier_size <= 2:
        pair_of_slot = generator.integers(0, len(PAIR_CANDIDATES), TILE_SLOTS).astype(np.intp)
        return KeySchedule(
            bit_of_slot=bit_of_slot,
            pair_of_slot=pair_of_slot,
            carriers=np.stack([_pair_carrier(pair) for pair in PAIR_CANDIDATES]),
            carrier_of_slot=pair_of_slot,
            keyed=True,
            carrier_size=2,
        )
    if carrier_size > len(CARRIER_BAND):
        raise WatermarkError(
            f"carrier size {carrier_size} exceeds the {len(CARRIER_BAND)}-coefficient band"
        )
    carriers = np.zeros((TILE_SLOTS, BLOCK_SIZE, BLOCK_SIZE), dtype=np.float64)
    for slot in range(TILE_SLOTS):
        chosen = generator.choice(len(CARRIER_BAND), carrier_size, replace=False)
        signs = generator.integers(0, 2, carrier_size) * 2 - 1
        for index, sign in zip(chosen, signs, strict=True):
            carriers[slot][CARRIER_BAND[int(index)]] = float(sign)
    return KeySchedule(
        bit_of_slot=bit_of_slot,
        pair_of_slot=np.zeros(TILE_SLOTS, dtype=np.intp),
        carriers=carriers,
        carrier_of_slot=np.arange(TILE_SLOTS, dtype=np.intp),
        keyed=True,
        carrier_size=int(carrier_size),
    )


PhaseReader = Callable[[int, int], np.ndarray]
Candidate = tuple[np.ndarray, float, dict[str, Any], PhaseReader]


class DctWatermarker(Watermarker):
    """Blind DCT mid-frequency differential watermark with majority-vote decoding.

    Blind means detection needs neither the original image nor the payload. The
    embedding strength trades perceptual quality against survivability, and the
    trade-off is measured, not assumed.
    """

    name = "dct"

    def __init__(self, config: WatermarkConfig | None = None) -> None:
        """Store watermark configuration and derive the tile layout from its key."""
        self.config = config or WatermarkConfig()
        self.schedule = derive_schedule(self.config.key, self.config.carrier_coefficients)

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
    def _coefficients(blocks: np.ndarray) -> np.ndarray:
        """Return the 2-D DCT of every block in the stack."""
        if blocks.shape[0] == 0:
            return np.zeros((0, BLOCK_SIZE, BLOCK_SIZE), dtype=np.float64)
        basis = _dct_matrix(BLOCK_SIZE)
        coefficients: np.ndarray = basis @ blocks @ basis.T
        return coefficients

    def _positive(self, coefficients: np.ndarray) -> np.ndarray:
        """Return whether each block reads as a one on each distinct carrier.

        A block's reading on a carrier is the sum of its coefficients weighted
        by the carrier pattern; for a pair that is exactly the difference of
        the two coefficients. Every distinct carrier is projected at once, so
        the tally under any tile alignment is a gather rather than a re-read.
        """
        flat = coefficients.reshape(coefficients.shape[0], BLOCK_SIZE * BLOCK_SIZE)
        carriers = self.schedule.carriers.reshape(-1, BLOCK_SIZE * BLOCK_SIZE)
        projections: np.ndarray = flat @ carriers.T
        positive: np.ndarray = (projections > 0).astype(np.float64)
        return positive

    def _tally(
        self,
        coefficients: np.ndarray,
        rows: int,
        cols: int,
        phase: tuple[int, int] = (0, 0),
        limit: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fold per-block observations into per-message-bit vote counts.

        ``phase`` is the tile alignment assumed while reading. Without a key it
        does not matter here, because a misaligned read is only a permutation
        of the bits and the decoder rolls the votes afterwards. With a key the
        alignment decides which carrier each block is read on, so the decoder
        has to read under every alignment instead; :meth:`_tally_phases` does
        that in one pass.
        """
        if coefficients.shape[0] == 0:
            return np.zeros(MESSAGE_BITS), np.zeros(MESSAGE_BITS)
        slots = tile_slots(rows, cols, phase[0], phase[1])[: coefficients.shape[0]]
        if limit is not None and coefficients.shape[0] > limit:
            slots = slots[:limit]
            coefficients = coefficients[:limit]
        positive = self._positive(coefficients)
        ones = positive[np.arange(positive.shape[0]), self.schedule.carrier_of_slot[slots]]
        bits = self.schedule.bit_of_slot[slots]
        votes = np.bincount(bits, weights=ones, minlength=MESSAGE_BITS)
        counts = np.bincount(bits, minlength=MESSAGE_BITS).astype(np.float64)
        return votes[:MESSAGE_BITS], counts[:MESSAGE_BITS]

    def _tally_phases(
        self, coefficients: np.ndarray, rows: int, cols: int, limit: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return per-bit votes and counts under every tile alignment, phases by bits.

        Blocks are first folded by their own tile cell, giving the number of
        ones each cell would contribute on each carrier. Under an alignment
        every cell is read as one slot, so the tally for that alignment is a
        gather from that table rather than another pass over the blocks. The
        sums are of whole numbers, so the result is identical to tallying each
        alignment on its own.
        """
        count = coefficients.shape[0]
        if count == 0:
            return np.zeros((TILE_SLOTS, MESSAGE_BITS)), np.zeros((TILE_SLOTS, MESSAGE_BITS))
        cells = tile_slots(rows, cols)[:count]
        if limit is not None and count > limit:
            cells = cells[:limit]
            coefficients = coefficients[:limit]
            count = limit
        membership = np.zeros((count, TILE_SLOTS), dtype=np.float64)
        membership[np.arange(count), cells] = 1.0
        ones_by_cell = membership.T @ self._positive(coefficients)
        blocks_by_cell = membership.sum(axis=0)

        read_as = PHASE_SLOT
        gathered = ones_by_cell[
            np.arange(TILE_SLOTS)[None, :], self.schedule.carrier_of_slot[read_as]
        ]
        bits = self.schedule.bit_of_slot[read_as]
        phase_index = np.arange(TILE_SLOTS)[:, None]
        votes = np.zeros((TILE_SLOTS, MESSAGE_BITS), dtype=np.float64)
        counts = np.zeros((TILE_SLOTS, MESSAGE_BITS), dtype=np.float64)
        votes[phase_index, bits] = gathered
        counts[phase_index, bits] = blocks_by_cell[None, :]
        return votes, counts

    def _ratios_by_phase(self, votes: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Return per-bit vote ratios for every alignment, phases by bits."""
        with np.errstate(invalid="ignore"):
            ratios: np.ndarray = np.divide(
                votes, counts, out=np.full(votes.shape, 0.5), where=counts > 0
            )
        return ratios

    def _best_agreement(
        self, coefficients: np.ndarray, rows: int, cols: int, limit: int | None = None
    ) -> float:
        """Return the vote agreement of one grid, over every tile alignment if keyed."""
        if not self.schedule.keyed:
            votes, counts = self._tally(coefficients, rows, cols, limit=limit)
            return self._agreement(votes, counts)[1]
        ratios = self._ratios_by_phase(*self._tally_phases(coefficients, rows, cols, limit))
        return float(np.max(np.mean(np.maximum(ratios, 1.0 - ratios), axis=1)))

    def _candidate(
        self, image: np.ndarray, offset: tuple[int, int], grid: dict[str, Any]
    ) -> Candidate:
        """Read one candidate grid fully and attach a reader for its tile alignments."""
        luminance = self._luminance(image)
        blocks, rows, cols = self._block_stack(luminance, offset[0], offset[1])
        coefficients = self._coefficients(blocks)
        votes, counts = self._tally(coefficients, rows, cols)
        ratios, _ = self._agreement(votes, counts)

        if self.schedule.keyed:
            by_phase = self._ratios_by_phase(*self._tally_phases(coefficients, rows, cols))
            agreement = float(np.max(np.mean(np.maximum(by_phase, 1.0 - by_phase), axis=1)))

            def read(row_shift: int, col_shift: int) -> np.ndarray:
                row: np.ndarray = by_phase[row_shift * TILE_COLS + col_shift]
                return row

        else:
            agreement = self._best_agreement(coefficients, rows, cols)

            def read(row_shift: int, col_shift: int) -> np.ndarray:
                return phase_shift(ratios, row_shift, col_shift)

        return ratios, agreement, grid, read

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
    def _rotated(image: np.ndarray, degrees: float) -> np.ndarray:
        """Return the image rotated about its centre, frame size kept.

        Undoing a rotation about the centre puts the block grid back where it
        was; the corners the rotation dragged in carry no votes worth counting,
        and the majority vote absorbs them.
        """
        rotated = Image.fromarray(validate_rgb(image)).rotate(
            float(degrees), resample=Image.Resampling.BICUBIC, expand=False
        )
        return np.asarray(rotated, dtype=np.uint8)

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
                bit = int(message[self.schedule.bit_of_slot[slot]])

                if self.schedule.carrier_size == 2:
                    first, second = PAIR_CANDIDATES[self.schedule.pair_of_slot[slot]]
                    a = coefficients[first]
                    b = coefficients[second]
                    mean = (a + b) / 2.0
                    half = margin / 2.0
                    if bit == 1:
                        coefficients[first] = mean + half
                        coefficients[second] = mean - half
                    else:
                        coefficients[first] = mean - half
                        coefficients[second] = mean + half
                else:
                    # Move the block the shortest distance that puts its reading
                    # on the carrier at exactly +/- margin: along the carrier.
                    carrier = self.schedule.carriers[self.schedule.carrier_of_slot[slot]]
                    reading = float(np.sum(carrier * coefficients))
                    target = margin if bit == 1 else -margin
                    coefficients += (target - reading) / float(self.schedule.carrier_size) * carrier

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
        return self._tally(self._coefficients(blocks), rows, cols)

    def _search_grid(self, image: np.ndarray) -> list[Candidate]:
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
                    agreement = self._best_agreement(
                        self._coefficients(blocks[:budget]), rows, cols, limit=budget
                    )
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
        candidates: list[Candidate] = []
        for _, grid in ranked[: self.config.resync_candidates]:
            winner = self._rescaled(image, grid["scale"])
            candidates.append(self._candidate(winner, (grid["offset_y"], grid["offset_x"]), grid))
        return candidates

    def _rank_offsets(self, luminance: np.ndarray) -> list[tuple[float, int, int]]:
        """Return every sub-block offset of one grid, best agreement first.

        Several are kept rather than one because agreement barely separates
        them on a strongly marked image: a grid shifted by two pixels reads the
        same structure with some signs inverted and scores almost as high, and
        only the checksum can tell the two apart.
        """
        budget = int(self.config.resync_max_blocks)
        ranked: list[tuple[float, int, int]] = []
        for offset_y in range(BLOCK_SIZE):
            for offset_x in range(BLOCK_SIZE):
                blocks, rows, cols = self._block_stack(luminance, offset_y, offset_x)
                if blocks.shape[0] < MESSAGE_BITS * MIN_REPETITIONS:
                    continue
                agreement = self._best_agreement(
                    self._coefficients(blocks[:budget]), rows, cols, limit=budget
                )
                ranked.append((agreement, offset_y, offset_x))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _score_angle(self, image: np.ndarray, angle: float) -> list[tuple[float, int, int]]:
        """Undo one candidate rotation and rank its offsets by agreement."""
        luminance = self._luminance(self._rotated(image, -angle))
        if min(luminance.shape) < BLOCK_SIZE * 2:
            return []
        return self._rank_offsets(luminance)

    def _search_rotation(self, image: np.ndarray, scale: float = 1.0) -> list[Candidate]:
        """Rank candidate rotation angles by vote agreement, coarse then fine.

        A one-degree coarse grid is enough to find the peak because half a
        degree of error was measured to decode exactly; the fine pass only
        sharpens the estimate before the full vote is taken. The image is
        expected to be at the scale named by ``scale`` already; the caller
        rescales it first, so the same sweep serves rotation on its own and
        rotation combined with a magnifying crop. Each refined angle
        contributes its best few offsets as separate candidates, for the
        reason given on :meth:`_rank_offsets`.
        """
        limit = float(self.config.resync_rotation_max_degrees)
        coarse_step = float(self.config.resync_rotation_coarse_step)
        fine_step = float(self.config.resync_rotation_fine_step)
        keep = int(self.config.resync_rotation_candidates)
        per_angle = int(self.config.resync_candidates)
        if limit <= 0.0:
            return []

        def peak(offsets: list[tuple[float, int, int]]) -> float:
            return offsets[0][0] if offsets else 0.0

        coarse = [
            float(angle)
            for angle in np.arange(-limit, limit + coarse_step / 2.0, coarse_step)
            if abs(angle) > 1e-9
        ]
        ranked = sorted(
            ((self._score_angle(image, angle), angle) for angle in coarse),
            key=lambda item: peak(item[0]),
            reverse=True,
        )

        candidates: list[Candidate] = []
        for offsets, angle in ranked[:keep]:
            best_offsets, best_angle = offsets, angle
            fine = np.arange(
                angle - coarse_step / 2.0, angle + coarse_step / 2.0 + 1e-9, fine_step
            )
            for refined in fine:
                if abs(float(refined) - angle) < 1e-9 or abs(float(refined)) > limit + 1e-9:
                    continue
                scored = self._score_angle(image, float(refined))
                if peak(scored) > peak(best_offsets):
                    best_offsets, best_angle = scored, float(refined)
            derotated = self._rotated(image, -best_angle)
            for _, offset_y, offset_x in best_offsets[:per_angle]:
                grid = {
                    "rotation": round(best_angle, 4),
                    "scale": float(scale),
                    "offset_y": offset_y,
                    "offset_x": offset_x,
                }
                candidates.append(self._candidate(derotated, (offset_y, offset_x), grid))
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

        accepted, best_confidence = self._decode_candidates(self._search_grid(array), confidence)

        if not accepted and self.config.resync_rotation_enabled:
            accepted, best_confidence = self._decode_candidates(
                self._search_rotation(array), best_confidence
            )

        if not accepted and self.config.resync_rotation_enabled:
            # Rotation after a magnifying crop: shrink back by each candidate
            # scale, then sweep the angle again. Rotation about the centre and
            # uniform scaling about the centre commute, so the order in which
            # the attacker applied them does not matter here.
            for scale in self.config.resync_rotation_scales:
                shrunk = self._rescaled(array, float(scale))
                if min(shrunk.shape[:2]) < BLOCK_SIZE * TILE_ROWS * 2:
                    continue
                accepted, best_confidence = self._decode_candidates(
                    self._search_rotation(shrunk, float(scale)), best_confidence
                )
                if accepted:
                    break

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

    def _decode_candidates(
        self,
        candidates: list[Candidate],
        confidence: float,
    ) -> tuple[dict[int, tuple[np.ndarray, int, float]], float]:
        """Try every tile phase of every candidate grid and collect the valid codes.

        Every code whose checksum validates is kept, keyed by its value, so the
        caller can apply the one guard the search depends on: a single image
        that yields two different valid codes is an accident of the search, not
        two watermarks, and is reported as nothing. The best confidence seen is
        returned alongside so a negative result still says how close it came.
        """
        accepted: dict[int, tuple[np.ndarray, int, float]] = {}
        best_confidence = confidence

        for _, grid_agreement, grid, read in candidates:
            resync_confidence = float(np.clip((grid_agreement - 0.5) * 2.0, 0.0, 1.0))
            best_confidence = max(best_confidence, resync_confidence)
            if resync_confidence < self.config.resync_min_confidence:
                continue

            for row_shift in range(TILE_ROWS):
                for col_shift in range(TILE_COLS):
                    shifted = read(row_shift, col_shift)
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

        return accepted, best_confidence

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
