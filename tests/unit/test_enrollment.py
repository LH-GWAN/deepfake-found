"""Phase 3: enrollment quality filtering and template construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepshield.config import default_config
from deepshield.exceptions import EnrollmentError
from deepshield.face.enrollment import DefaultIdentityEnroller
from deepshield.media import save_image


@pytest.fixture
def mock_config():
    config = default_config()
    return config.model_copy(
        update={
            "enrollment": config.enrollment.model_copy(
                update={
                    "min_images": 2,
                    "quality": config.enrollment.quality.model_copy(
                        update={"min_sharpness": 0.0, "min_detection_confidence": 0.5}
                    ),
                }
            )
        }
    )


def write_images(directory: Path, count: int, seed: int = 0) -> list[Path]:
    from tests.conftest import synthetic_photo

    directory.mkdir(parents=True, exist_ok=True)
    return [
        save_image(synthetic_photo(seed=seed + i, size=200), directory / f"img_{i}.png")
        for i in range(count)
    ]


def test_enroll_builds_a_template(tmp_path: Path, mock_config) -> None:
    paths = write_images(tmp_path / "faces", 3)
    result = DefaultIdentityEnroller(mock_config).enroll("user-1", paths)
    profile = result.profile
    assert profile.user_id == "user-1"
    assert profile.image_count == 3
    assert profile.reference_embeddings.shape[0] == 3
    assert result.accepted_count == 3


def test_references_and_centroid_are_unit_norm(tmp_path: Path, mock_config) -> None:
    paths = write_images(tmp_path / "faces", 3)
    profile = DefaultIdentityEnroller(mock_config).enroll("u", paths).profile
    norms = np.linalg.norm(profile.reference_embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    assert float(np.linalg.norm(profile.centroid_embedding)) == pytest.approx(1.0, abs=1e-5)


def test_references_are_kept_not_just_the_centroid(tmp_path: Path, mock_config) -> None:
    """Averaging alone discards the pose variation multiple photos were collected for."""
    paths = write_images(tmp_path / "faces", 4)
    profile = DefaultIdentityEnroller(mock_config).enroll("u", paths).profile
    assert profile.reference_embeddings.shape[0] == 4
    assert profile.centroid_embedding.shape == profile.reference_embeddings[0].shape


def test_too_few_usable_images_is_an_error_with_reasons(tmp_path: Path, mock_config) -> None:
    paths = write_images(tmp_path / "faces", 1)
    with pytest.raises(EnrollmentError, match="quality filtering"):
        DefaultIdentityEnroller(mock_config).enroll("u", paths)


def test_max_images_is_respected(tmp_path: Path, mock_config) -> None:
    config = mock_config.model_copy(
        update={"enrollment": mock_config.enrollment.model_copy(update={"max_images": 2})}
    )
    paths = write_images(tmp_path / "faces", 5)
    result = DefaultIdentityEnroller(config).enroll("u", paths)
    assert len(result.reports) == 2


def test_unreadable_image_is_reported_not_crashed(tmp_path: Path, mock_config) -> None:
    paths = write_images(tmp_path / "faces", 3)
    broken = tmp_path / "faces" / "broken.png"
    broken.write_bytes(b"not an image")
    result = DefaultIdentityEnroller(mock_config).enroll("u", [*paths, broken])
    reasons = [r.reason for r in result.reports if not r.accepted]
    assert any("unreadable" in reason for reason in reasons)


def test_empty_user_id_is_rejected(tmp_path: Path, mock_config) -> None:
    paths = write_images(tmp_path / "faces", 2)
    with pytest.raises(EnrollmentError, match="user_id"):
        DefaultIdentityEnroller(mock_config).enroll("", paths)


def test_no_images_supplied_is_rejected(mock_config) -> None:
    with pytest.raises(EnrollmentError, match="no enrollment images"):
        DefaultIdentityEnroller(mock_config).enroll("u", [])


def test_missing_directory_is_rejected(tmp_path: Path, mock_config) -> None:
    with pytest.raises(EnrollmentError, match="not found"):
        DefaultIdentityEnroller(mock_config).enroll_directory("u", tmp_path / "nope")


def test_directory_without_images_is_rejected(tmp_path: Path, mock_config) -> None:
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "notes.txt").write_text("hello", encoding="utf-8")
    with pytest.raises(EnrollmentError, match="no supported images"):
        DefaultIdentityEnroller(mock_config).enroll_directory("u", tmp_path / "empty")


def test_report_records_the_quality_measurements(tmp_path: Path, mock_config) -> None:
    paths = write_images(tmp_path / "faces", 2)
    result = DefaultIdentityEnroller(mock_config).enroll("u", paths)
    for report in result.reports:
        assert report.detection_confidence is not None
        assert report.sharpness is not None
        assert report.to_dict()["accepted"] is True


def test_profile_dict_hides_biometric_vectors(tmp_path: Path, mock_config) -> None:
    paths = write_images(tmp_path / "faces", 2)
    payload = DefaultIdentityEnroller(mock_config).enroll("u", paths).profile.to_dict()
    assert "reference_embeddings" not in payload
    assert "centroid_embedding" not in payload
    assert len(payload["centroid_digest"]) == 12
