"""Concrete deepfake detection backends.

Two real backends sit behind the Phase 0 adapter:

``spectral``
    A training-free frequency-artefact heuristic. Upsampling layers in many
    generative architectures leave periodic structure and an unusually flat or
    elevated high-frequency tail in the azimuthally averaged power spectrum.
    It needs no weights, which makes it useful as a baseline and as proof that
    the adapter boundary works - and it is weak. It responds to resizing,
    denoising and compression just as readily as to synthesis, and modern
    diffusion models were explicitly trained against this class of artefact.
    Its score is uncalibrated and must never be read as a verdict.

``onnx``
    Any ONNX image classifier the user supplies. This is the intended production
    path: a detector trained on a relevant dataset is dropped in through
    configuration, and its name, version and training dataset are recorded with
    every score so that results stay attributable after models change.

Both share the failure mode that motivates the whole multi-signal design: a
detector generalises poorly to generator families absent from its training data,
so its output is one piece of evidence rather than the answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from deepshield.config import DeepfakeDetectorConfig
from deepshield.detection.deepfake import DEEPFAKE_REGISTRY, DeepfakeDetector
from deepshield.exceptions import InvalidMediaError, ModelNotAvailableError
from deepshield.media import to_grayscale, validate_rgb
from deepshield.types import DeepfakeResult, ModelInfo

SPECTRAL_SIDE = 256
TRIMMED_FRACTION = 0.1


def azimuthal_power_spectrum(image: np.ndarray, side: int = SPECTRAL_SIDE) -> np.ndarray:
    """Return the radially averaged log power spectrum of an image.

    The 2-D FFT of the luminance plane is averaged over rings of constant radius,
    collapsing it into a 1-D profile from low to high spatial frequency. Natural
    photographs fall off smoothly with frequency; several synthesis pipelines
    leave a flatter tail or periodic bumps from their upsampling stages.
    """
    plane = to_grayscale(validate_rgb(image))
    resized = np.asarray(
        Image.fromarray(plane.astype(np.uint8)).resize(
            (side, side), Image.Resampling.LANCZOS
        ),
        dtype=np.float64,
    )
    window = np.hanning(side)
    windowed = resized * window[:, None] * window[None, :]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(windowed))) ** 2

    centre = side // 2
    yy, xx = np.mgrid[0:side, 0:side]
    radius = np.sqrt((yy - centre) ** 2 + (xx - centre) ** 2).astype(np.int32)
    totals = np.bincount(radius.ravel(), weights=spectrum.ravel())
    counts = np.bincount(radius.ravel())
    profile = totals[:centre] / np.maximum(counts[:centre], 1)
    return np.log1p(profile)


class SpectralArtifactDetector(DeepfakeDetector):
    """Training-free frequency-artefact heuristic.

    Scores the flatness of the high-frequency tail of the azimuthal power
    spectrum relative to its mid band. It is deterministic, needs no weights and
    runs in milliseconds. It is also uncalibrated and known to confuse ordinary
    image processing with synthesis, which is stated in every result it returns.
    """

    name = "spectral"

    def __init__(self, config: DeepfakeDetectorConfig | None = None) -> None:
        """Store detector configuration."""
        self.config = config or DeepfakeDetectorConfig()

    @property
    def model_info(self) -> ModelInfo:
        """Return metadata describing this heuristic."""
        return ModelInfo(
            name="spectral-artifact-heuristic",
            version="0.1.0",
            backend=self.name,
            training_dataset=None,
            input_size=SPECTRAL_SIDE,
        )

    def _score(self, image: np.ndarray) -> float:
        """Return the high-frequency flatness score in ``[0, 1]``."""
        profile = azimuthal_power_spectrum(image)
        usable = profile[4:]
        if usable.size < 16 or float(np.ptp(usable)) == 0.0:
            return 0.5
        split = usable.size // 2
        mid = float(np.mean(usable[:split]))
        high = float(np.mean(usable[split:]))
        denominator = abs(mid) + 1e-9
        ratio = high / denominator
        return float(np.clip(ratio, 0.0, 1.0))

    def predict_image(self, image: np.ndarray) -> DeepfakeResult:
        """Score one image from its frequency profile."""
        array = self.validate_image(image)
        return DeepfakeResult(
            score=self._score(array),
            model=self.model_info,
            notes=[
                "spectral heuristic: uncalibrated, no training data, "
                "responds to resizing and denoising as well as to synthesis",
            ],
        )

    def predict_video(self, frames: list[np.ndarray]) -> DeepfakeResult:
        """Score sampled frames and aggregate them."""
        if not frames:
            raise InvalidMediaError("no frames supplied to predict_video")
        scores = [self._score(self.validate_image(frame)) for frame in frames]
        return DeepfakeResult(
            score=aggregate_frame_scores(scores, self.config.frame_aggregation),
            model=self.model_info,
            per_frame_scores=scores,
            notes=[
                "spectral heuristic: uncalibrated, no training data, "
                "responds to resizing and denoising as well as to synthesis",
            ],
        )


class OnnxDeepfakeDetector(DeepfakeDetector):
    """Any ONNX image classifier supplied through configuration.

    Expects a model taking ``1 x 3 x S x S`` float input in ``[0, 1]`` RGB and
    producing either one logit or a two-class output. The class treated as
    "synthetic" is configurable, because dataset conventions differ and silently
    inverting a detector is an easy and expensive mistake.
    """

    name = "onnx"

    def __init__(self, config: DeepfakeDetectorConfig | None = None) -> None:
        """Create an inference session for the configured ONNX file."""
        self.config = config or DeepfakeDetectorConfig()
        try:
            import onnxruntime
        except ImportError as exc:
            raise ModelNotAvailableError(
                "onnxruntime is not installed; install the 'face' extra"
            ) from exc
        if self.config.model_path is None:
            raise ModelNotAvailableError(
                "the onnx deepfake backend needs detection.deepfake.model_path"
            )
        path = Path(self.config.model_path)
        if not path.is_file():
            raise ModelNotAvailableError(f"ONNX detector not found: {path}")
        self.model_path = path
        self._session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    @property
    def model_info(self) -> ModelInfo:
        """Return metadata naming the exact ONNX file and its training data."""
        return ModelInfo(
            name=self.config.model_name or self.model_path.stem,
            version=self.config.model_version,
            backend=self.name,
            training_dataset=self.config.training_dataset,
            input_size=self.config.input_size,
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize and scale an image into the model's expected tensor layout."""
        side = self.config.input_size
        resized = np.asarray(
            Image.fromarray(validate_rgb(image)).resize(
                (side, side), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        return np.transpose(resized / 255.0, (2, 0, 1))[None, ...]

    def _score(self, image: np.ndarray) -> float:
        """Run the model and reduce its output to a probability."""
        outputs = self._session.run(None, {self._input_name: self._preprocess(image)})
        logits = np.asarray(outputs[0], dtype=np.float64).ravel()
        if logits.size == 1:
            return float(1.0 / (1.0 + np.exp(-logits[0])))
        shifted = logits - logits.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        index = min(self.config.positive_index, probabilities.size - 1)
        return float(probabilities[index])

    def predict_image(self, image: np.ndarray) -> DeepfakeResult:
        """Score one image with the supplied model."""
        array = self.validate_image(image)
        return DeepfakeResult(
            score=self._score(array),
            model=self.model_info,
            notes=[
                "detector output is a likelihood, not a verdict; "
                "generalisation to unseen generators is unverified",
            ],
        )

    def predict_video(self, frames: list[np.ndarray]) -> DeepfakeResult:
        """Score sampled frames in batches and aggregate them."""
        if not frames:
            raise InvalidMediaError("no frames supplied to predict_video")
        scores = [self._score(self.validate_image(frame)) for frame in frames]
        return DeepfakeResult(
            score=aggregate_frame_scores(scores, self.config.frame_aggregation),
            model=self.model_info,
            per_frame_scores=scores,
            notes=[
                "detector output is a likelihood, not a verdict; "
                "generalisation to unseen generators is unverified",
            ],
        )


def aggregate_frame_scores(scores: list[float], strategy: str = "trimmed_mean") -> float:
    """Reduce per-frame scores to one video-level score.

    ``max`` is deliberately not the default: a single mis-scored frame in a
    thousand would pin every video at 1.0. A trimmed mean discards the extremes
    at both ends, which is the behaviour that survives contact with real footage.
    """
    if not scores:
        raise InvalidMediaError("cannot aggregate an empty score list")
    values = np.asarray(scores, dtype=np.float64)
    if strategy == "max":
        return float(values.max())
    if strategy == "mean":
        return float(values.mean())
    if strategy == "trimmed_mean":
        if values.size < 5:
            return float(values.mean())
        trim = max(1, int(values.size * TRIMMED_FRACTION))
        return float(np.sort(values)[trim:-trim].mean())
    raise ModelNotAvailableError(f"unsupported frame aggregation '{strategy}'")


DEEPFAKE_REGISTRY.register("spectral", SpectralArtifactDetector)
DEEPFAKE_REGISTRY.register("onnx", OnnxDeepfakeDetector)
