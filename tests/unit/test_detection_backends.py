"""Phase 7: spectral heuristic, ONNX adapter contract and frame aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from deepshield.config import DeepfakeDetectorConfig
from deepshield.detection.deepfake import build_deepfake_detector
from deepshield.detection.deepfake_backends import (
    OnnxDeepfakeDetector,
    SpectralArtifactDetector,
    aggregate_frame_scores,
    azimuthal_power_spectrum,
)
from deepshield.exceptions import InvalidMediaError, ModelNotAvailableError
from deepshield.transforms import Transformation


def test_spectrum_is_monotone_in_length(photo: np.ndarray) -> None:
    profile = azimuthal_power_spectrum(photo, side=128)
    assert profile.shape == (64,)
    assert np.all(np.isfinite(profile))


def test_spectral_score_is_bounded_and_deterministic(photo: np.ndarray) -> None:
    detector = SpectralArtifactDetector()
    first = detector.predict_image(photo)
    second = detector.predict_image(photo.copy())
    assert 0.0 <= first.score <= 1.0
    assert first.score == second.score


def test_spectral_result_always_carries_its_caveat(photo: np.ndarray) -> None:
    """An uncalibrated heuristic must never be presented as a finding."""
    result = SpectralArtifactDetector().predict_image(photo)
    assert result.notes
    assert any("uncalibrated" in note for note in result.notes)
    assert result.model.training_dataset is None


def test_spectral_responds_to_low_pass_filtering(photo: np.ndarray) -> None:
    """Blurring removes the high-frequency band the heuristic reads."""
    detector = SpectralArtifactDetector()
    sharp = detector.predict_image(photo).score
    blurred = detector.predict_image(
        Transformation("b", "blur", {"sigma": 4.0}).apply(photo, seed=0)
    ).score
    assert blurred < sharp


def test_spectral_handles_a_flat_image() -> None:
    flat = np.full((128, 128, 3), 128, dtype=np.uint8)
    assert 0.0 <= SpectralArtifactDetector().predict_image(flat).score <= 1.0


def test_spectral_video_aggregates(photo: np.ndarray, other_photo: np.ndarray) -> None:
    result = SpectralArtifactDetector().predict_video([photo, other_photo])
    assert len(result.per_frame_scores) == 2
    assert 0.0 <= result.score <= 1.0


def test_spectral_video_rejects_empty_input() -> None:
    with pytest.raises(InvalidMediaError, match="no frames"):
        SpectralArtifactDetector().predict_video([])


def test_spectral_rejects_invalid_images() -> None:
    with pytest.raises(InvalidMediaError):
        SpectralArtifactDetector().predict_image(np.zeros((8, 8), dtype=np.uint8))


def test_trimmed_mean_is_the_default_not_max() -> None:
    """One outlier frame must not pin an entire video at 1.0."""
    scores = [0.1] * 20 + [1.0]
    trimmed = aggregate_frame_scores(scores, "trimmed_mean")
    assert trimmed < 0.2
    assert aggregate_frame_scores(scores, "max") == 1.0


def test_trimmed_mean_falls_back_on_short_sequences() -> None:
    assert aggregate_frame_scores([0.2, 0.4], "trimmed_mean") == pytest.approx(0.3)


def test_aggregation_rejects_empty_and_unknown() -> None:
    with pytest.raises(InvalidMediaError):
        aggregate_frame_scores([])
    with pytest.raises(ModelNotAvailableError, match="unsupported frame aggregation"):
        aggregate_frame_scores([0.5], "median")


def test_onnx_backend_requires_a_model_path() -> None:
    with pytest.raises(ModelNotAvailableError, match="model_path"):
        OnnxDeepfakeDetector(DeepfakeDetectorConfig(backend="onnx"))


def test_onnx_backend_reports_a_missing_file(tmp_path) -> None:
    config = DeepfakeDetectorConfig(backend="onnx", model_path=tmp_path / "missing.onnx")
    with pytest.raises(ModelNotAvailableError, match="not found"):
        OnnxDeepfakeDetector(config)


def test_registry_resolves_every_backend_name() -> None:
    assert isinstance(
        build_deepfake_detector(DeepfakeDetectorConfig(backend="spectral")),
        SpectralArtifactDetector,
    )
    with pytest.raises(ModelNotAvailableError):
        build_deepfake_detector(DeepfakeDetectorConfig(backend="nope"))


def _tiny_onnx_classifier(path, classes: int = 2) -> object:
    """Export a minimal two-class image classifier so the adapter can be exercised."""
    torch = pytest.importorskip("torch")

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(3, classes)

        def forward(self, pixels):
            return self.linear(pixels.mean(dim=(2, 3)))

    model = Tiny().eval()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 5.0]]))
        model.linear.bias.zero_()
    torch.onnx.export(
        model,
        torch.rand(1, 3, 64, 64),
        str(path),
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=14,
    )
    return path


def test_onnx_adapter_runs_a_real_model(tmp_path) -> None:
    """The adapter must actually execute a graph, not just validate its path."""
    pytest.importorskip("onnxruntime")
    path = _tiny_onnx_classifier(tmp_path / "tiny.onnx")
    detector = OnnxDeepfakeDetector(
        DeepfakeDetectorConfig(backend="onnx", model_path=path, input_size=64, positive_index=1)
    )
    blue = np.zeros((80, 80, 3), dtype=np.uint8)
    blue[:, :, 2] = 255
    red = np.zeros((80, 80, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    assert detector.predict_image(blue).score > detector.predict_image(red).score


def test_onnx_adapter_respects_the_positive_index(tmp_path) -> None:
    """Reading the wrong output class silently inverts a detector."""
    pytest.importorskip("onnxruntime")
    path = _tiny_onnx_classifier(tmp_path / "tiny.onnx")
    blue = np.zeros((80, 80, 3), dtype=np.uint8)
    blue[:, :, 2] = 255

    first = OnnxDeepfakeDetector(
        DeepfakeDetectorConfig(model_path=path, input_size=64, positive_index=0)
    ).predict_image(blue).score
    second = OnnxDeepfakeDetector(
        DeepfakeDetectorConfig(model_path=path, input_size=64, positive_index=1)
    ).predict_image(blue).score
    assert first + second == pytest.approx(1.0, abs=1e-5)
    assert second > first


def test_onnx_results_carry_their_training_dataset(tmp_path) -> None:
    pytest.importorskip("onnxruntime")
    path = _tiny_onnx_classifier(tmp_path / "tiny.onnx")
    detector = OnnxDeepfakeDetector(
        DeepfakeDetectorConfig(
            model_path=path, input_size=64, training_dataset="unit-test corpus"
        )
    )
    result = detector.predict_image(np.zeros((80, 80, 3), dtype=np.uint8))
    assert result.model.training_dataset == "unit-test corpus"
    assert any("generalisation" in note for note in result.notes)
