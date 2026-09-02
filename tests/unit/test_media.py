"""Media loading, saving, validation and hashing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepshield.exceptions import InvalidMediaError
from deepshield.media import (
    is_image_path,
    is_video_path,
    load_image,
    save_image,
    sha256_array,
    sha256_file,
    to_grayscale,
    validate_rgb,
)


def test_save_and_load_round_trip(tmp_path: Path, rgb_image: np.ndarray) -> None:
    path = save_image(rgb_image, tmp_path / "out.png")
    np.testing.assert_array_equal(load_image(path), rgb_image)


def test_save_creates_parent_directories(tmp_path: Path, rgb_image: np.ndarray) -> None:
    path = save_image(rgb_image, tmp_path / "nested" / "deep" / "out.png")
    assert path.exists()


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(InvalidMediaError, match="not found"):
        load_image(tmp_path / "missing.png")


def test_load_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(InvalidMediaError, match="not a file"):
        load_image(tmp_path)


def test_load_corrupt_file_raises(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"this is not a png")
    with pytest.raises(InvalidMediaError, match="unsupported or corrupt"):
        load_image(corrupt)


def test_load_converts_grayscale_to_rgb(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "gray.png"
    Image.fromarray(np.full((16, 16), 128, dtype=np.uint8), mode="L").save(path)
    loaded = load_image(path)
    assert loaded.shape == (16, 16, 3)
    assert loaded.dtype == np.uint8


def test_validate_rgb_rejects_wrong_rank() -> None:
    with pytest.raises(InvalidMediaError, match="RGB"):
        validate_rgb(np.zeros((8, 8), dtype=np.uint8))


def test_validate_rgb_rejects_empty() -> None:
    with pytest.raises(InvalidMediaError, match="empty"):
        validate_rgb(np.zeros((0, 4, 3), dtype=np.uint8))


def test_validate_rgb_casts_float_input() -> None:
    result = validate_rgb(np.full((4, 4, 3), 300.0))
    assert result.dtype == np.uint8
    assert result.max() == 255


def test_grayscale_uses_luma_weights() -> None:
    red = np.zeros((2, 2, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    assert float(to_grayscale(red)[0, 0]) == pytest.approx(76.245, abs=0.01)


def test_sha256_file_matches_known_content(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    path.write_bytes(b"deepshield")
    assert sha256_file(path) == sha256_file(path)
    assert len(sha256_file(path)) == 64


def test_sha256_array_is_content_addressed(rgb_image: np.ndarray) -> None:
    changed = rgb_image.copy()
    changed[0, 0, 0] = (int(changed[0, 0, 0]) + 1) % 256
    assert sha256_array(rgb_image) == sha256_array(rgb_image.copy())
    assert sha256_array(rgb_image) != sha256_array(changed)


def test_file_and_pixel_hashes_answer_different_questions(
    tmp_path: Path, rgb_image: np.ndarray
) -> None:
    png = save_image(rgb_image, tmp_path / "a.png")
    bmp = save_image(rgb_image, tmp_path / "a.bmp")
    assert sha256_file(png) != sha256_file(bmp)
    assert sha256_array(load_image(png)) == sha256_array(load_image(bmp))


def test_suffix_helpers() -> None:
    assert is_image_path(Path("a.JPG"))
    assert not is_image_path(Path("a.mp4"))
    assert is_video_path(Path("a.MP4"))
    assert not is_video_path(Path("a.png"))
