"""The shield must find every face the swapper would, and fail clearly without its model."""

from __future__ import annotations

from pathlib import Path

import pytest

from deepshield.config import default_config

pytest.importorskip("insightface")
pytest.importorskip("onnxruntime")


def test_the_shield_detector_keeps_faces_the_swapper_would_take(monkeypatch) -> None:
    """The swapper's FaceAnalysis (inswapper) keeps detections from 0.5 up with no size floor."""
    from deepshield.face import backends
    from deepshield.pipeline.protection_pipeline import DefaultProtectionPipeline

    captured = {}

    class Recording:
        def __init__(self, config, model_dir) -> None:
            captured["config"] = config

    monkeypatch.setattr(backends, "InsightFaceDetector", Recording)
    pipeline = DefaultProtectionPipeline.__new__(DefaultProtectionPipeline)
    pipeline.config = default_config()
    pipeline._shield_detector()
    config = captured["config"]
    assert config.min_face_size <= 1
    assert config.detection_confidence_threshold <= 0.5
    assert config.implausible_below_confidence == 0.0


def test_a_missing_detection_model_is_reported_not_asserted(tmp_path: Path) -> None:
    """An empty pack folder used to surface as a bare AssertionError (HTTP 500)."""
    from deepshield.face import backends
    from deepshield.face.backends import InsightFaceDetector

    (tmp_path / "insightface" / "models" / "buffalo_l").mkdir(parents=True)
    # The class the backend raises, which other tests may have reloaded under it.
    with pytest.raises(backends.ModelNotAvailableError, match="det_10g"):
        InsightFaceDetector(model_dir=tmp_path)
