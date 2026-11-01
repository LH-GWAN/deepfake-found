"""Reading a FaceForensics++ download whatever mirror it came from."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_ffpp_manifest as ffpp  # noqa: E402


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("FaceForensics++/original_sequences/youtube/c23/videos/000.mp4", ("real", "000")),
        ("manipulated_sequences/Deepfakes/c23/videos/000_003.mp4", ("deepfakes", "000_003")),
        ("manipulated_sequences/NeuralTextures/c23/videos/071_054.mp4",
         ("neuraltextures", "071_054")),
        ("FF++/Face2Face/000_003.mp4", ("face2face", "000_003")),
        ("ffpp_c23/FaceSwap_c23/912_927.mp4", ("faceswap", "912_927")),
        ("frames/Deepfakes/000_003/0001.png", ("deepfakes", "000_003")),
        ("frames/original/000/0042.png", ("real", "000")),
    ],
)
def test_clips_are_recognised_across_mirror_layouts(name: str, expected: tuple[str, str]) -> None:
    assert ffpp.classify(name) == expected


def test_deepfake_detection_is_not_read_as_deepfakes() -> None:
    assert ffpp.classify("DeepFakeDetection/c23/videos/123_456.mp4") == (
        "deepfakedetection", "123_456"
    )


def test_another_compression_is_skipped_unless_any_is_asked_for() -> None:
    name = "manipulated_sequences/Deepfakes/c40/videos/000_003.mp4"
    assert ffpp.classify(name, "c23") is None
    assert ffpp.classify(name, None) == ("deepfakes", "000_003")


def test_files_that_are_neither_real_nor_a_manipulation_are_skipped() -> None:
    assert ffpp.classify("FaceForensics++/benchmark/000.mp4") is None
    assert ffpp.classify("manipulated_sequences/Deepfakes/c23/videos/readme.mp4") is None


def test_clips_group_frames_by_folder_and_keep_only_the_wanted_methods() -> None:
    names = [
        "original_sequences/youtube/c23/videos/000.mp4",
        "manipulated_sequences/Deepfakes/c23/videos/000_003.mp4",
        "manipulated_sequences/FaceShifter/c23/videos/000_003.mp4",
        "frames/Face2Face/003_000/0001.png",
        "frames/Face2Face/003_000/0002.png",
        "manipulated_sequences/Deepfakes/c23/videos/notes.txt",
    ]
    clips = ffpp.find_clips(names, ["deepfakes", "face2face"], "c23")
    assert [(c.kind, c.video, c.is_video, len(c.members)) for c in clips] == [
        ("deepfakes", "000_003", True, 1),
        ("face2face", "003_000", False, 2),
        ("real", "000", True, 1),
    ]


def test_every_clip_a_fake_name_joins_shares_one_identity() -> None:
    groups = ffpp.identity_groups(["000", "003", "005", "000_003", "003_005", "001"])
    assert groups["000"] == groups["003"] == groups["005"]
    assert groups["001"] != groups["000"]


def test_frames_are_spread_away_from_both_ends() -> None:
    assert ffpp.spread(100, 3) == [25, 50, 74]
    assert ffpp.spread(2, 5) == [0, 1]
    assert ffpp.spread(0, 3) == []


def test_a_zip_is_read_member_by_member(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    Image.fromarray(np.full((4, 6, 3), 200, np.uint8)).save(buffer, format="PNG")
    archive = tmp_path / "ffpp.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("frames/original/000/0001.png", buffer.getvalue())
        handle.writestr("frames/original/", "")
        handle.writestr("frames/original/000/clip.mp4", b"not really a video")
    source = ffpp.Source(archive)
    assert sorted(source.names()) == [
        "frames/original/000/0001.png", "frames/original/000/clip.mp4"
    ]
    assert source.image("frames/original/000/0001.png").shape == (4, 6, 3)
    with source.local("frames/original/000/clip.mp4") as local:
        assert local.read_bytes() == b"not really a video"
        extracted = local
    assert not extracted.exists()


def test_a_clip_found_twice_is_kept_once() -> None:
    """Mask videos and a second compression would weight a clip, and its person, twice."""
    names = [
        "manipulated_sequences/Deepfakes/masks/videos/000_003.mp4",
        "manipulated_sequences/Deepfakes/c40/videos/000_003.mp4",
        "manipulated_sequences/Deepfakes/c23/videos/000_003.mp4",
        "original_sequences/youtube/c23/videos/000.mp4",
        "original_sequences/youtube/raw/videos/000.mp4",
        "frames/original/000/0001.png",
    ]
    clips = ffpp.find_clips(names, ["deepfakes"], None)
    assert [(c.kind, c.video, c.members) for c in clips] == [
        ("deepfakes", "000_003", ("manipulated_sequences/Deepfakes/c23/videos/000_003.mp4",)),
        ("real", "000", ("original_sequences/youtube/c23/videos/000.mp4",)),
    ]
    only_masks = ffpp.find_clips(names[:1], ["deepfakes"], "c23")
    assert only_masks == []


def test_frame_folders_at_two_compressions_are_not_merged() -> None:
    names = [
        "frames/c40/Deepfakes/000_003/0001.png",
        "frames/c23/Deepfakes/000_003/0001.png",
        "frames/c23/Deepfakes/000_003/0002.png",
    ]
    (clip,) = ffpp.find_clips(names, ["deepfakes"], None)
    assert clip.members == (
        "frames/c23/Deepfakes/000_003/0001.png", "frames/c23/Deepfakes/000_003/0002.png"
    )


def test_crops_made_with_other_settings_are_not_reused(tmp_path: Path) -> None:
    settings = {"frames_real": 8, "max_side": 512}
    ffpp.check_parameters(tmp_path, settings)
    ffpp.check_parameters(tmp_path, dict(settings))
    with pytest.raises(SystemExit, match="max_side: 512 -> 256"):
        ffpp.check_parameters(tmp_path, {**settings, "max_side": 256})


def test_a_video_without_a_frame_count_is_counted_by_decoding() -> None:
    assert ffpp.frame_total(300, lambda: 999) == (300, False)
    assert ffpp.frame_total(0, lambda: 240) == (240, True)
    assert ffpp.frame_total(-1, lambda: 240) == (240, True)
