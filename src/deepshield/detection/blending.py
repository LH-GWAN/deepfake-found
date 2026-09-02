"""Blending-artefact detection for graphics-based face swaps.

Plain language: a swapped face is a piece of one photograph pasted into
another, and the paste leaves traces even when the seam is invisible.

The features below all compare the middle of a face crop against its border.
The pipeline hands this detector a crop with a margin, so the centre is the
possibly-replaced face and the border is untouched hair, neck and background.
Every measurement is a ratio between the two, which cancels out most of what
varies between photographs - exposure, camera, subject - and leaves the
inconsistency a paste introduces:

noise residual
    Warping and blending resample the pasted region, smoothing the fine sensor
    noise that the rest of the frame keeps.
sharpness
    The same resampling costs high-frequency detail, so the interior is softer
    than its surroundings.
blockiness
    JPEG leaves an 8-pixel grid. A warped face no longer aligns to the grid of
    the image it was pasted into, so the two regions have different blockiness.
spectral slope
    Interpolation attenuates the high-frequency tail of the pasted region.
colour statistics
    Even after colour matching, the interior and its surroundings rarely agree
    perfectly in mean and spread.
edge energy on the seam
    A ring around the face carries the blending boundary itself.

The classifier over these features is a logistic regression. It has eleven
coefficients, trains in milliseconds, and can be read and argued with, which
matters more here than the last few points of accuracy - especially since the
training family is narrow and honesty about what it covers is the whole point.

What this does not detect: GAN or diffusion synthesis, reenactment, or a swap
regenerated end to end by a neural network.

Measured result, which is the reason this backend is not the default: on
Poisson-blended swaps it reaches a held-out ROC-AUC of about 0.58, and only 0.64
even in sample. Seamless cloning solves the boundary discontinuity these
features are built to find, so there is little left for them to measure. Every
score therefore carries the model's own measured AUC in its notes, and the
deepfake threshold stays uncalibrated so the risk engine keeps excluding the
signal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deepshield.config import DeepfakeDetectorConfig
from deepshield.detection.deepfake import DEEPFAKE_REGISTRY, DeepfakeDetector
from deepshield.detection.deepfake_backends import aggregate_frame_scores
from deepshield.exceptions import InvalidMediaError, ModelNotAvailableError
from deepshield.logging_utils import get_logger
from deepshield.media import to_grayscale, validate_rgb
from deepshield.types import DeepfakeResult, ModelInfo

logger = get_logger(__name__)

FEATURE_NAMES = (
    "noise_ratio",
    "sharpness_ratio",
    "blockiness_ratio",
    "spectral_ratio",
    "mean_difference",
    "std_ratio",
    "seam_edge_ratio",
    "saturation_difference",
    "interior_noise",
    "interior_sharpness",
    "edge_density",
)
DEFAULT_MODEL_PATH = Path("models/blending_detector.json")
INTERIOR_FRACTION = 0.55
SEAM_INNER = 0.5
SEAM_OUTER = 0.75
EPSILON = 1e-6


def _region_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return interior, exterior and seam-ring masks for a face crop."""
    yy, xx = np.mgrid[0:height, 0:width]
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    radius = np.sqrt(((yy - cy) / max(cy, 1)) ** 2 + ((xx - cx) / max(cx, 1)) ** 2)
    interior = radius <= INTERIOR_FRACTION
    exterior = radius >= SEAM_OUTER
    seam = (radius > SEAM_INNER) & (radius < SEAM_OUTER)
    return interior, exterior, seam


def _high_pass(plane: np.ndarray) -> np.ndarray:
    """Return the absolute residual after a 3x3 box blur, a cheap noise proxy."""
    padded = np.pad(plane, 1, mode="reflect")
    accumulated = np.zeros_like(plane)
    for dy in range(3):
        for dx in range(3):
            accumulated += padded[dy : dy + plane.shape[0], dx : dx + plane.shape[1]]
    residual: np.ndarray = np.abs(plane - accumulated / 9.0)
    return residual


def _laplacian(plane: np.ndarray) -> np.ndarray:
    """Return the absolute Laplacian response of a plane."""
    padded = np.pad(plane, 1, mode="reflect")
    centre = padded[1:-1, 1:-1]
    response: np.ndarray = np.abs(
        padded[0:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, 0:-2] + padded[1:-1, 2:] - 4 * centre
    )
    return response


def _blockiness(plane: np.ndarray) -> float:
    """Return how much stronger differences are across 8-pixel boundaries.

    JPEG quantises 8x8 blocks independently, leaving a step at every block edge.
    A region warped from another image no longer aligns to that grid, so the
    step is weaker there than in untouched parts of the same picture.
    """
    if plane.shape[1] < 17 or plane.shape[0] < 17:
        return 0.0
    columns = np.abs(np.diff(plane, axis=1))
    rows = np.abs(np.diff(plane, axis=0))
    boundary_columns = columns[:, 7::8]
    boundary_rows = rows[7::8, :]
    interior_columns = np.delete(columns, np.s_[7::8], axis=1)
    interior_rows = np.delete(rows, np.s_[7::8], axis=0)
    boundary = float(boundary_columns.mean() + boundary_rows.mean())
    interior = float(interior_columns.mean() + interior_rows.mean()) + EPSILON
    return boundary / interior


def _spectral_high_ratio(plane: np.ndarray) -> float:
    """Return the share of spectral energy sitting in the high-frequency half."""
    if min(plane.shape) < 16:
        return 0.0
    window = np.outer(np.hanning(plane.shape[0]), np.hanning(plane.shape[1]))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(plane * window))) ** 2
    cy, cx = np.array(spectrum.shape) // 2
    yy, xx = np.mgrid[0 : spectrum.shape[0], 0 : spectrum.shape[1]]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    limit = float(radius.max())
    high = float(spectrum[radius > limit * 0.5].sum())
    total = float(spectrum.sum()) + EPSILON
    return high / total


def extract_features(image: np.ndarray) -> np.ndarray:
    """Return the blending-artefact feature vector of one face crop.

    Raises:
        InvalidMediaError: If the crop is too small to measure.

    """
    array = validate_rgb(image)
    height, width = array.shape[:2]
    if height < 32 or width < 32:
        raise InvalidMediaError("face crop is too small for blending analysis")

    plane = to_grayscale(array).astype(np.float64)
    interior, exterior, seam = _region_masks(height, width)
    if interior.sum() < 16 or exterior.sum() < 16:
        raise InvalidMediaError("face crop is too small for blending analysis")

    noise = _high_pass(plane)
    sharp = _laplacian(plane)

    interior_noise = float(noise[interior].mean())
    exterior_noise = float(noise[exterior].mean()) + EPSILON
    interior_sharp = float(sharp[interior].mean())
    exterior_sharp = float(sharp[exterior].mean()) + EPSILON

    inner_box = plane[
        int(height * 0.25) : int(height * 0.75), int(width * 0.25) : int(width * 0.75)
    ]
    outer_strip = plane[: max(1, int(height * 0.15)), :]

    channel_max = array.max(axis=2).astype(np.float64)
    channel_min = array.min(axis=2).astype(np.float64)
    saturation = (channel_max - channel_min) / (channel_max + EPSILON)

    features = np.array(
        [
            interior_noise / exterior_noise,
            interior_sharp / exterior_sharp,
            _blockiness(inner_box) / (_blockiness(outer_strip) + EPSILON),
            _spectral_high_ratio(inner_box) / (_spectral_high_ratio(outer_strip) + EPSILON),
            (float(plane[interior].mean()) - float(plane[exterior].mean())) / 255.0,
            (float(plane[interior].std()) + EPSILON) / (float(plane[exterior].std()) + EPSILON),
            float(sharp[seam].mean()) / exterior_sharp if seam.sum() > 0 else 1.0,
            float(saturation[interior].mean()) - float(saturation[exterior].mean()),
            interior_noise / 255.0,
            interior_sharp / 255.0,
            float((sharp > sharp.mean() * 3).mean()),
        ],
        dtype=np.float64,
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


class BlendingArtifactDetector(DeepfakeDetector):
    """Logistic regression over blending-artefact features.

    The model file holds the feature means, scales and coefficients produced by
    ``scripts/train_deepfake_detector.py``, along with the manipulation family it
    was fitted on. That family travels with every score, because it is the
    single best predictor of where the detector will fail.
    """

    name = "blending"

    def __init__(self, config: DeepfakeDetectorConfig | None = None) -> None:
        """Load the trained coefficients.

        Raises:
            ModelNotAvailableError: If no trained model file is present.

        """
        self.config = config or DeepfakeDetectorConfig()
        path = Path(self.config.model_path or DEFAULT_MODEL_PATH)
        if not path.is_file():
            raise ModelNotAvailableError(
                f"no trained blending detector at {path}; run "
                "scripts/train_deepfake_detector.py or select another backend"
            )
        model = json.loads(path.read_text(encoding="utf-8"))
        self.model_path = path
        self.mean = np.asarray(model["mean"], dtype=np.float64)
        self.scale = np.asarray(model["scale"], dtype=np.float64)
        self.coefficients = np.asarray(model["coefficients"], dtype=np.float64)
        self.intercept = float(model["intercept"])
        self.trained_on = model.get("trained_on", "unknown")
        self.version = model.get("version", "0.1.0")
        self.metrics = model.get("metrics", {})
        self.usable = bool(model.get("usable", False))
        if not self.usable:
            logger.warning(
                "blending detector loaded from %s did not reach its usability floor "
                "(held-out AUC %.3f); its scores are reported but should not be trusted",
                path,
                float(self.metrics.get("clean_auc", 0.0)),
            )
        if len(self.coefficients) != len(FEATURE_NAMES):
            raise ModelNotAvailableError(
                f"model at {path} has {len(self.coefficients)} coefficients, "
                f"expected {len(FEATURE_NAMES)}"
            )

    @property
    def model_info(self) -> ModelInfo:
        """Return metadata naming the manipulation family this was fitted on."""
        return ModelInfo(
            name="blending-artifact-logistic",
            version=self.version,
            backend=self.name,
            training_dataset=self.trained_on,
            input_size=None,
        )

    @property
    def notes(self) -> list[str]:
        """Return the caveats attached to every score this detector produces."""
        notes = [
            f"fitted on {self.trained_on}",
            "covers graphics-based face swapping; does not detect GAN or "
            "diffusion synthesis, or reenactment",
            "measures blending inconsistency between the middle and the border "
            "of the supplied crop, so it needs a crop with margin around the face",
        ]
        auc = float(self.metrics.get("clean_auc", 0.0))
        if auc:
            notes.append(
                f"held-out ROC-AUC on its own training family is {auc:.3f}"
                + ("" if self.usable else "; this is near chance and the score is not evidence")
            )
        return notes

    def score_features(self, features: np.ndarray) -> float:
        """Return the synthetic-media probability for a feature vector."""
        standardised = (features - self.mean) / np.maximum(self.scale, EPSILON)
        logit = float(np.dot(standardised, self.coefficients) + self.intercept)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def predict_image(self, image: np.ndarray) -> DeepfakeResult:
        """Score one face crop."""
        array = self.validate_image(image)
        try:
            features = extract_features(array)
        except InvalidMediaError as exc:
            logger.debug("blending features unavailable: %s", exc)
            return DeepfakeResult(
                score=0.5,
                model=self.model_info,
                notes=[*self.notes, f"crop unusable, returned a neutral score: {exc}"],
            )
        return DeepfakeResult(
            score=self.score_features(features), model=self.model_info, notes=self.notes
        )

    def predict_video(self, frames: list[np.ndarray]) -> DeepfakeResult:
        """Score sampled frames and aggregate them."""
        if not frames:
            raise InvalidMediaError("no frames supplied to predict_video")
        scores = [self.predict_image(frame).score for frame in frames]
        return DeepfakeResult(
            score=aggregate_frame_scores(scores, self.config.frame_aggregation),
            model=self.model_info,
            per_frame_scores=scores,
            notes=self.notes,
        )


DEEPFAKE_REGISTRY.register("blending", BlendingArtifactDetector)
