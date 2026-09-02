"""Transformation engine shared by every robustness experiment.

Real content never reaches an analyst untouched: platforms re-encode, users
screenshot, and attackers deliberately crop and rescale. Each transformation
here reproduces one of those steps so that face similarity, watermark recovery,
fingerprint matching and deepfake scoring can be measured after it.

The same engine is reused by all experiments so that a number from the watermark
benchmark and a number from the face-recognition benchmark refer to exactly the
same operation. Every transformation is deterministic given a seed, and its
parameters are recorded alongside its result.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from deepshield.exceptions import ConfigurationError
from deepshield.media import validate_rgb

TransformFn = Callable[..., np.ndarray]
TRANSFORMS: dict[str, TransformFn] = {}


def register_transform(name: str) -> Callable[[TransformFn], TransformFn]:
    """Return a decorator registering a transformation implementation."""

    def wrapper(function: TransformFn) -> TransformFn:
        TRANSFORMS[name] = function
        return function

    return wrapper


def _to_pil(image: np.ndarray) -> Image.Image:
    """Convert a validated RGB array into a PIL image."""
    return Image.fromarray(validate_rgb(image))


def _to_array(image: Image.Image) -> np.ndarray:
    """Convert a PIL image back into an RGB uint8 array."""
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _reencode(image: np.ndarray, image_format: str, quality: int) -> np.ndarray:
    """Round-trip an image through a lossy codec in memory."""
    buffer = io.BytesIO()
    _to_pil(image).save(buffer, format=image_format, quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return _to_array(decoded)


@register_transform("identity")
def identity(image: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return the image unchanged, used as the experiment control condition."""
    return validate_rgb(image).copy()


@register_transform("jpeg_compression")
def jpeg_compression(
    image: np.ndarray, rng: np.random.Generator | None = None, quality: int = 75
) -> np.ndarray:
    """Re-encode as JPEG, discarding high-frequency detail."""
    return _reencode(image, "JPEG", quality)


@register_transform("webp")
def webp(
    image: np.ndarray, rng: np.random.Generator | None = None, quality: int = 80
) -> np.ndarray:
    """Re-encode as WebP, the default upload format of several platforms."""
    return _reencode(image, "WEBP", quality)


@register_transform("resize")
def resize(
    image: np.ndarray, rng: np.random.Generator | None = None, scale: float = 0.5
) -> np.ndarray:
    """Downscale then restore the original size, destroying fine detail."""
    if scale <= 0:
        raise ConfigurationError("resize scale must be positive")
    array = validate_rgb(image)
    height, width = array.shape[:2]
    small = _to_pil(array).resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.LANCZOS,
    )
    return _to_array(small.resize((width, height), Image.Resampling.LANCZOS))


@register_transform("downscale")
def downscale(
    image: np.ndarray, rng: np.random.Generator | None = None, scale: float = 0.5
) -> np.ndarray:
    """Downscale without restoring the original resolution."""
    if scale <= 0:
        raise ConfigurationError("downscale scale must be positive")
    array = validate_rgb(image)
    height, width = array.shape[:2]
    target = (max(1, int(width * scale)), max(1, int(height * scale)))
    return _to_array(_to_pil(array).resize(target, Image.Resampling.LANCZOS))


@register_transform("crop")
def crop(
    image: np.ndarray, rng: np.random.Generator | None = None, ratio: float = 0.1
) -> np.ndarray:
    """Remove a border of ``ratio`` from each side and restore the original size.

    Cropping is the transformation that most reliably defeats block-aligned
    watermarks, because it destroys the synchronisation between encoder and
    decoder grids.
    """
    if not 0.0 <= ratio < 0.5:
        raise ConfigurationError("crop ratio must be in [0, 0.5)")
    array = validate_rgb(image)
    height, width = array.shape[:2]
    dx, dy = int(width * ratio), int(height * ratio)
    cropped = array[dy : height - dy, dx : width - dx]
    if cropped.size == 0:
        return array.copy()
    return _to_array(_to_pil(cropped).resize((width, height), Image.Resampling.LANCZOS))


@register_transform("rotation")
def rotation(
    image: np.ndarray, rng: np.random.Generator | None = None, degrees: float = 5.0
) -> np.ndarray:
    """Rotate about the centre, keeping the frame size and filling with edge pixels."""
    rotated = _to_pil(image).rotate(
        float(degrees), resample=Image.Resampling.BICUBIC, expand=False
    )
    return _to_array(rotated)


@register_transform("blur")
def blur(
    image: np.ndarray, rng: np.random.Generator | None = None, sigma: float = 1.0
) -> np.ndarray:
    """Apply a Gaussian blur, simulating soft focus or aggressive denoising."""
    return _to_array(_to_pil(image).filter(ImageFilter.GaussianBlur(radius=float(sigma))))


@register_transform("noise")
def noise(
    image: np.ndarray, rng: np.random.Generator | None = None, sigma: float = 5.0
) -> np.ndarray:
    """Add zero-mean Gaussian pixel noise."""
    generator = rng or np.random.default_rng()
    array = validate_rgb(image).astype(np.float32)
    noisy = array + generator.normal(0.0, float(sigma), size=array.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


@register_transform("brightness")
def brightness(
    image: np.ndarray, rng: np.random.Generator | None = None, factor: float = 1.2
) -> np.ndarray:
    """Scale brightness, as an auto-enhance filter would."""
    return _to_array(ImageEnhance.Brightness(_to_pil(image)).enhance(float(factor)))


@register_transform("contrast")
def contrast(
    image: np.ndarray, rng: np.random.Generator | None = None, factor: float = 0.8
) -> np.ndarray:
    """Scale contrast."""
    return _to_array(ImageEnhance.Contrast(_to_pil(image)).enhance(float(factor)))


@register_transform("color")
def color(
    image: np.ndarray, rng: np.random.Generator | None = None, factor: float = 1.2
) -> np.ndarray:
    """Scale colour saturation."""
    return _to_array(ImageEnhance.Color(_to_pil(image)).enhance(float(factor)))


@register_transform("screenshot_simulation")
def screenshot_simulation(
    image: np.ndarray,
    rng: np.random.Generator | None = None,
    scale: float = 0.9,
    quality: int = 85,
) -> np.ndarray:
    """Approximate a screenshot: rescale, resample back, then re-encode.

    A screenshot is the most common way protected content is laundered, and it
    combines resampling with a fresh lossy encode.
    """
    return jpeg_compression(resize(image, rng, scale=scale), rng, quality=quality)


@dataclass(frozen=True)
class Transformation:
    """One named, parameterised, reproducible transformation."""

    name: str
    type: str
    params: dict[str, Any] = field(default_factory=dict)

    def apply(self, image: np.ndarray, seed: int | None = None) -> np.ndarray:
        """Apply the transformation to an image.

        Args:
            image: ``H x W x 3`` RGB array.
            seed: Seed for the transformations that use randomness.

        Raises:
            ConfigurationError: If the transformation type is unknown.

        """
        function = TRANSFORMS.get(self.type)
        if function is None:
            available = ", ".join(sorted(TRANSFORMS))
            raise ConfigurationError(
                f"unknown transformation type '{self.type}'; available: {available}"
            )
        rng = np.random.default_rng(seed)
        return function(image, rng, **self.params)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable mapping used in experiment rows."""
        return {"name": self.name, "type": self.type, "params": dict(self.params)}


class TransformationPipeline:
    """An ordered, seeded collection of named transformations.

    Args:
        transformations: Transformations to apply, in order.
        seed: Base seed; each transformation derives its own stream from it so
            that reordering the pipeline does not silently change the results.

    """

    def __init__(self, transformations: Iterable[Transformation], seed: int = 42) -> None:
        """Store the transformation list and the base seed."""
        self.transformations = list(transformations)
        self.seed = seed

    @classmethod
    def from_config(
        cls, definitions: dict[str, dict[str, Any]], names: Iterable[str], seed: int = 42
    ) -> TransformationPipeline:
        """Build a pipeline from ``configs/experiments.yaml`` style definitions.

        Raises:
            ConfigurationError: If a requested name is not defined.

        """
        selected = []
        for name in names:
            definition = definitions.get(name)
            if definition is None:
                available = ", ".join(sorted(definitions))
                raise ConfigurationError(
                    f"transformation '{name}' is not defined; available: {available}"
                )
            params = {k: v for k, v in definition.items() if k != "type"}
            selected.append(Transformation(name=name, type=definition["type"], params=params))
        return cls(selected, seed=seed)

    def apply_all(self, image: np.ndarray) -> np.ndarray:
        """Apply every transformation in sequence and return the final image."""
        result = validate_rgb(image)
        for index, transformation in enumerate(self.transformations):
            result = transformation.apply(result, seed=self.seed + index)
        return result

    def apply_each(self, image: np.ndarray) -> list[tuple[Transformation, np.ndarray]]:
        """Apply each transformation independently to the original image.

        This is the mode benchmarks use: every transformation is measured against
        the same input rather than compounding with the previous ones.
        """
        source = validate_rgb(image)
        return [
            (transformation, transformation.apply(source, seed=self.seed + index))
            for index, transformation in enumerate(self.transformations)
        ]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the pipeline."""
        return {
            "seed": self.seed,
            "transformations": [t.to_dict() for t in self.transformations],
        }
