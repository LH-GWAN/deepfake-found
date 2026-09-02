"""End-to-end protection and analysis over the real components.

These use mock face backends so they run without model downloads, but the
pipeline order, the candidate gate, the watermark and the risk arithmetic are
the production ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from deepshield.config import DeepShieldConfig, default_config
from deepshield.exceptions import InvalidMediaError
from deepshield.media import save_image
from deepshield.pipeline.analysis_pipeline import DefaultAnalysisPipeline, crop_with_margin
from deepshield.pipeline.protection_pipeline import DefaultProtectionPipeline
from deepshield.storage import (
    build_asset_repository,
    build_identity_repository,
    build_provenance_store,
)
from deepshield.types import BoundingBox, DetectedFace

pytestmark = pytest.mark.integration


@pytest.fixture
def config(tmp_path: Path) -> DeepShieldConfig:
    base = default_config()
    return base.model_copy(
        update={
            "runtime": base.runtime.model_copy(
                update={
                    "data_dir": tmp_path / "data",
                    "results_dir": tmp_path / "data" / "results",
                    "model_dir": tmp_path / "models",
                }
            ),
            "storage": base.storage.model_copy(
                update={"embedding_store_dir": tmp_path / "data" / "embeddings"}
            ),
            "face": base.face.model_copy(
                update={
                    "detector": base.face.detector.model_copy(update={"backend": "mock"}),
                    "aligner": base.face.aligner.model_copy(update={"backend": "mock"}),
                    "embedder": base.face.embedder.model_copy(update={"backend": "mock"}),
                }
            ),
            "detection": base.detection.model_copy(
                update={"deepfake": base.detection.deepfake.model_copy(update={"backend": "mock"})}
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


@pytest.fixture
def source_image(tmp_path: Path, large_photo: np.ndarray) -> Path:
    return save_image(large_photo, tmp_path / "source.png")


def test_protect_produces_a_verifiable_asset(config, source_image: Path) -> None:
    report = DefaultProtectionPipeline(config).protect(source_image, "u1", "instagram")
    assert Path(report["protected_path"]).is_file()
    assert report["watermark"]["embedded"] is True
    assert report["watermark"]["verified_after_save"] is True
    assert report["quality"]["ssim"] > 0.85


def test_protection_registers_asset_and_provenance(config, source_image: Path) -> None:
    report = DefaultProtectionPipeline(config).protect(source_image, "u1", "x-com")
    asset_id = report["asset_id"]

    asset = build_asset_repository(config).get(asset_id)
    assert asset is not None
    assert asset.distribution_id == "x-com"

    lineage = build_provenance_store(config).lineage(asset_id)
    assert [record.asset_id for record in lineage] == [asset_id, f"{asset_id}:source"]


def test_distribution_ids_produce_different_codes(config, source_image: Path) -> None:
    """Per-channel codes are what make a leak traceable to a channel."""
    pipeline = DefaultProtectionPipeline(config)
    first = pipeline.protect(source_image, "u1", "instagram")
    second = pipeline.protect(source_image, "u1", "x-com")
    assert first["watermark"]["code"] != second["watermark"]["code"]


def test_analysis_recovers_the_watermark_and_the_asset(config, source_image: Path) -> None:
    report = DefaultProtectionPipeline(config).protect(source_image, "u1", "instagram")
    record = DefaultAnalysisPipeline(config).analyze_image(Path(report["protected_path"]))
    assert record.watermark_detected is True
    assert record.watermark_code == report["watermark"]["code"]
    assert record.matched_asset_id == report["asset_id"]
    assert record.provenance_confidence == 1.0


def test_analysis_of_an_unprotected_image_claims_nothing(config, source_image: Path) -> None:
    record = DefaultAnalysisPipeline(config).analyze_image(source_image)
    assert record.watermark_detected is False
    assert record.matched_asset_id is None
    assert record.matched_user_id is None
    assert "No face was compared" in record.summary


def test_deepfake_detector_is_gated_by_identity(config, source_image: Path) -> None:
    """Without an enrolled identity there is nothing to attribute, so no detector runs."""
    record = DefaultAnalysisPipeline(config).analyze_image(source_image)
    assert record.deepfake_score is None
    assert any("was not run" in line for line in record.limitations)


def test_enrolled_identity_opens_the_gate(config, source_image: Path, tmp_path: Path) -> None:
    from deepshield.face.enrollment import DefaultIdentityEnroller
    from tests.conftest import synthetic_photo

    directory = tmp_path / "refs"
    directory.mkdir()
    paths = [
        save_image(synthetic_photo(seed=5, size=300), directory / f"r{i}.png") for i in range(3)
    ]
    result = DefaultIdentityEnroller(config).enroll("u1", paths)
    build_identity_repository(config).save(result.profile)

    probe = save_image(synthetic_photo(seed=5, size=300), tmp_path / "probe.png")
    record = DefaultAnalysisPipeline(config).analyze_image(probe, "u1")
    assert record.face_similarity is not None
    assert record.matched_user_id == "u1"
    assert record.deepfake_score is not None


def test_evidence_record_is_json_serialisable(config, source_image: Path) -> None:
    import json

    record = DefaultAnalysisPipeline(config).analyze_image(source_image)
    payload = json.loads(json.dumps(record.to_dict(), default=str))
    assert payload["limitations"]
    assert payload["detector_versions"]["face_embedder"]["backend"] == "mock"


def test_analysis_rejects_a_video_path(config, tmp_path: Path) -> None:
    fake = tmp_path / "clip.mp4"
    fake.write_bytes(b"0")
    with pytest.raises(InvalidMediaError, match="analyze_video"):
        DefaultAnalysisPipeline(config).analyze_image(fake)


def test_analysis_rejects_a_corrupt_image(config, tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(InvalidMediaError):
        DefaultAnalysisPipeline(config).analyze_image(broken)


def test_crop_with_margin_expands_and_clamps(large_photo: np.ndarray) -> None:
    face = DetectedFace(BoundingBox(10, 10, 60, 60), 0.9)
    crop = crop_with_margin(large_photo, face, 0.25)
    assert crop.shape[0] > 50
    edge = DetectedFace(BoundingBox(-100, -100, 20, 20), 0.9)
    assert crop_with_margin(large_photo, edge, 0.5).size > 0


def test_borderline_similarity_is_not_reported_as_a_match(config, tmp_path: Path) -> None:
    """Clearing the candidate threshold justifies a detector run, not an identity claim."""
    from deepshield.face.enrollment import DefaultIdentityEnroller
    from tests.conftest import synthetic_photo

    strict = config.model_copy(
        update={
            "thresholds": config.thresholds.model_copy(
                update={
                    "face_similarity": config.thresholds.face_similarity.model_copy(
                        update={
                            "candidate_threshold": -1.0,
                            "high_confidence_threshold": 1.5,
                        }
                    )
                }
            )
        }
    )
    directory = tmp_path / "refs"
    directory.mkdir()
    paths = [
        save_image(synthetic_photo(seed=5, size=300), directory / f"r{i}.png") for i in range(3)
    ]
    result = DefaultIdentityEnroller(strict).enroll("u1", paths)
    build_identity_repository(strict).save(result.profile)

    probe = save_image(synthetic_photo(seed=5, size=300), tmp_path / "probe.png")
    record = DefaultAnalysisPipeline(strict).analyze_image(probe, "u1")

    assert record.face_similarity is not None
    assert record.matched_user_id is None
    assert record.identity_decision == "candidate"
    assert "worth reviewing, not a match" in record.summary
    assert any("high-confidence threshold" in line for line in record.limitations)
    assert record.deepfake_score is not None


def test_probe_quality_is_recorded_per_face(config, source_image: Path) -> None:
    record = DefaultAnalysisPipeline(config).analyze_image(source_image)
    assert record.faces
    assert "probe_quality" in record.faces[0]


def test_evidence_reports_the_identity_decision(config, source_image: Path) -> None:
    record = DefaultAnalysisPipeline(config).analyze_image(source_image)
    payload = record.to_dict()
    assert payload["identity"]["decision"] in {
        "no_match",
        "candidate",
        "ambiguous",
        "high_confidence",
    }
