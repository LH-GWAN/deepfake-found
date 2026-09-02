"""Phase 13: reproducibility metadata and benchmark row shape."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from deepshield.config import default_config
from deepshield.experiments import (
    ExperimentResult,
    FaceRobustnessExperiment,
    WatermarkRobustnessExperiment,
    environment,
    load_transformations,
)
from deepshield.media import save_image


@pytest.fixture
def mock_config():
    base = default_config()
    return base.model_copy(
        update={
            "face": base.face.model_copy(
                update={
                    "detector": base.face.detector.model_copy(update={"backend": "mock"}),
                    "aligner": base.face.aligner.model_copy(update={"backend": "mock"}),
                    "embedder": base.face.embedder.model_copy(update={"backend": "mock"}),
                }
            ),
            "protection": base.protection.model_copy(
                update={
                    "watermark": base.protection.watermark.model_copy(
                        update={"backend": "dct"}
                    )
                }
            ),
        }
    )


def test_environment_records_everything_needed_to_reproduce(mock_config) -> None:
    captured = environment(mock_config)
    for key in ("git_commit", "python", "random_seed", "face_embedder", "deepshield_version"):
        assert key in captured


def test_result_writes_csv_with_provenance_columns(tmp_path: Path, mock_config) -> None:
    result = ExperimentResult("demo", [{"a": 1}], environment(mock_config))
    path = result.write_csv(tmp_path / "out.csv")
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["a"] == "1"
    assert rows[0]["env_git_commit"]
    assert rows[0]["env_random_seed"]


def test_empty_result_refuses_to_write(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no rows"):
        ExperimentResult("demo", []).write_csv(tmp_path / "out.csv")


def test_summary_reports_statistics() -> None:
    result = ExperimentResult("demo", [{"v": 0.2}, {"v": 0.8}, {"v": None}])
    summary = result.summary("v")
    assert summary["count"] == 2
    assert summary["mean"] == pytest.approx(0.5)


def test_load_transformations_from_shipped_config(project_root: Path) -> None:
    pipeline = load_transformations(project_root / "configs" / "experiments.yaml")
    assert len(pipeline.transformations) >= 10


def test_face_experiment_row_shape(tmp_path: Path, mock_config, photo) -> None:
    path = save_image(photo, tmp_path / "a.png")
    pipeline = load_transformations(Path("configs/experiments.yaml"), ["jpeg_q70", "resize_50"])
    result = FaceRobustnessExperiment(mock_config).run([path], pipeline)
    assert len(result.rows) == 2
    row = result.rows[0]
    for key in ("image_id", "transformation", "face_detected", "face_similarity", "psnr", "ssim"):
        assert key in row


def test_face_experiment_reports_detection_failure_distinctly(
    tmp_path: Path, mock_config, photo
) -> None:
    """A missed detection is a different failure from a wrong identity."""
    path = save_image(photo, tmp_path / "a.png")
    pipeline = load_transformations(Path("configs/experiments.yaml"), ["jpeg_q90"])
    row = FaceRobustnessExperiment(mock_config).run([path], pipeline).rows[0]
    assert row["face_detected"] in (True, False)
    if not row["face_detected"]:
        assert row["face_similarity"] is None


def test_watermark_experiment_tracks_false_attribution(
    tmp_path: Path, mock_config, large_photo
) -> None:
    """Recovering the wrong code would be worse than recovering none."""
    path = save_image(large_photo, tmp_path / "a.png")
    pipeline = load_transformations(Path("configs/experiments.yaml"), ["jpeg_q90", "crop_30"])
    result = WatermarkRobustnessExperiment(mock_config).run([path], pipeline)
    assert len(result.rows) == 2
    assert all(row["false_attribution"] is False for row in result.rows)
    clean = next(r for r in result.rows if r["transformation"] == "jpeg_q90")
    assert clean["watermark_detected"] is True
    assert clean["code_correct"] is True
