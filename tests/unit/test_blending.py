"""Phase 7 extension: blending-artefact features and the negative result they produced."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from deepshield.config import DeepfakeDetectorConfig
from deepshield.detection.blending import (
    FEATURE_NAMES,
    BlendingArtifactDetector,
    extract_features,
)
from deepshield.exceptions import InvalidMediaError, ModelNotAvailableError
from deepshield.transforms import Transformation


def write_model(path: Path, usable: bool = True, clean_auc: float = 0.9) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": "test",
                "features": list(FEATURE_NAMES),
                "mean": [0.0] * len(FEATURE_NAMES),
                "scale": [1.0] * len(FEATURE_NAMES),
                "coefficients": [0.1] * len(FEATURE_NAMES),
                "intercept": 0.0,
                "trained_on": "unit-test swaps",
                "usable": usable,
                "metrics": {"clean_auc": clean_auc},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_feature_vector_has_the_declared_length(photo: np.ndarray) -> None:
    features = extract_features(photo)
    assert features.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(features))


def test_features_are_deterministic(photo: np.ndarray) -> None:
    np.testing.assert_allclose(extract_features(photo), extract_features(photo.copy()))


def test_features_respond_to_interior_resampling(photo: np.ndarray) -> None:
    """Blurring only the middle is the artefact these features exist to measure."""
    altered = photo.copy()
    height, width = altered.shape[:2]
    inner = Transformation("b", "blur", {"sigma": 3.0}).apply(photo, seed=0)
    y0, y1 = height // 4, 3 * height // 4
    x0, x1 = width // 4, 3 * width // 4
    altered[y0:y1, x0:x1] = inner[y0:y1, x0:x1]
    assert not np.allclose(extract_features(photo), extract_features(altered))


def test_tiny_crops_are_rejected() -> None:
    with pytest.raises(InvalidMediaError, match="too small"):
        extract_features(np.zeros((16, 16, 3), dtype=np.uint8))


def test_detector_requires_a_trained_model(tmp_path: Path) -> None:
    config = DeepfakeDetectorConfig(backend="blending", model_path=tmp_path / "missing.json")
    with pytest.raises(ModelNotAvailableError, match="no trained blending detector"):
        BlendingArtifactDetector(config)


def test_detector_rejects_a_model_of_the_wrong_width(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"mean": [0.0], "scale": [1.0], "coefficients": [1.0], "intercept": 0.0}),
        encoding="utf-8",
    )
    with pytest.raises(ModelNotAvailableError, match="coefficients"):
        BlendingArtifactDetector(DeepfakeDetectorConfig(model_path=path))


def test_scores_are_probabilities(tmp_path: Path, photo: np.ndarray) -> None:
    path = write_model(tmp_path / "m.json")
    detector = BlendingArtifactDetector(DeepfakeDetectorConfig(model_path=path))
    result = detector.predict_image(photo)
    assert 0.0 <= result.score <= 1.0


def test_every_score_names_its_training_family(tmp_path: Path, photo: np.ndarray) -> None:
    path = write_model(tmp_path / "m.json")
    detector = BlendingArtifactDetector(DeepfakeDetectorConfig(model_path=path))
    notes = " ".join(detector.predict_image(photo).notes)
    assert "unit-test swaps" in notes
    assert "does not detect GAN" in notes


def test_an_unusable_model_says_so_in_every_score(tmp_path: Path, photo: np.ndarray) -> None:
    """A detector at chance level must not be mistaken for evidence."""
    path = write_model(tmp_path / "m.json", usable=False, clean_auc=0.58)
    detector = BlendingArtifactDetector(DeepfakeDetectorConfig(model_path=path))
    notes = " ".join(detector.predict_image(photo).notes)
    assert "0.580" in notes
    assert "not evidence" in notes


def test_unusable_crops_return_a_neutral_score(tmp_path: Path) -> None:
    path = write_model(tmp_path / "m.json")
    detector = BlendingArtifactDetector(DeepfakeDetectorConfig(model_path=path))
    result = detector.predict_image(np.zeros((40, 40, 3), dtype=np.uint8))
    assert 0.0 <= result.score <= 1.0


def test_video_aggregation(tmp_path: Path, photo: np.ndarray, other_photo: np.ndarray) -> None:
    path = write_model(tmp_path / "m.json")
    detector = BlendingArtifactDetector(DeepfakeDetectorConfig(model_path=path))
    result = detector.predict_video([photo, other_photo])
    assert len(result.per_frame_scores) == 2
    with pytest.raises(InvalidMediaError):
        detector.predict_video([])


def test_shipped_model_records_that_it_is_not_usable() -> None:
    """The measured result is a negative one, and the model file must carry it."""
    path = Path("models/blending_detector.json")
    if not path.is_file():
        pytest.skip("model not built in this environment")
    model = json.loads(path.read_text(encoding="utf-8"))
    assert model["usable"] is False
    assert model["metrics"]["clean_auc"] < model["usable_floor"]


def test_deepfake_threshold_is_not_calibrated(project_root: Path) -> None:
    """The chance-level result must keep the signal out of the risk score."""
    from deepshield.config import load_config

    config = load_config(
        config_path=project_root / "configs" / "default.yaml",
        thresholds_path=project_root / "configs" / "thresholds.yaml",
        environ={},
    )
    assert config.thresholds.deepfake.calibrated is False


def test_published_detector_survey_records_its_verdicts() -> None:
    """The survey is a result, not a scratch file: its verdicts must be readable."""
    path = Path("data/results/deepfake_detector_survey.json")
    if not path.is_file():
        pytest.skip("survey not run in this environment")
    survey = json.loads(path.read_text(encoding="utf-8"))
    assert survey["criteria"]["min_auc"] > 0.5
    assert survey["criteria"]["max_false_positive_rate"] < 0.5
    for name, detector in survey["detectors"].items():
        clean = detector["per_condition"]["clean"]
        expected = (
            clean["auc"] >= survey["criteria"]["min_auc"]
            and clean["fpr_at_default"] <= survey["criteria"]["max_false_positive_rate"]
        )
        assert detector["usable"] is expected, f"{name} verdict disagrees with its metrics"


def test_no_detector_was_adopted_while_none_qualifies(project_root: Path) -> None:
    """A detector that fails the survey must not become the configured backend."""
    import json as json_module

    from deepshield.config import load_config

    path = project_root / "data" / "results" / "deepfake_detector_survey.json"
    if not path.is_file():
        pytest.skip("survey not run in this environment")
    survey = json_module.loads(path.read_text(encoding="utf-8"))
    if any(detector["usable"] for detector in survey["detectors"].values()):
        pytest.skip("a detector qualified; adoption is expected")

    config = load_config(
        config_path=project_root / "configs" / "default.yaml",
        thresholds_path=project_root / "configs" / "thresholds.yaml",
        environ={},
    )
    assert config.thresholds.deepfake.calibrated is False
