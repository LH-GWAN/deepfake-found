"""Phase 13 transformation engine: correctness, determinism and reproducibility."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from deepshield.exceptions import ConfigurationError
from deepshield.quality import psnr
from deepshield.transforms import (
    TRANSFORMS,
    Transformation,
    TransformationPipeline,
)

ALL_TYPES = sorted(TRANSFORMS)


@pytest.mark.parametrize("transform_type", ALL_TYPES)
def test_every_transform_preserves_rgb_uint8(transform_type: str, photo: np.ndarray) -> None:
    result = Transformation(transform_type, transform_type).apply(photo, seed=0)
    assert result.ndim == 3
    assert result.shape[2] == 3
    assert result.dtype == np.uint8


@pytest.mark.parametrize("transform_type", [t for t in ALL_TYPES if t != "downscale"])
def test_shape_preserving_transforms_keep_size(transform_type: str, photo: np.ndarray) -> None:
    assert Transformation("t", transform_type).apply(photo, seed=0).shape == photo.shape


def test_downscale_changes_resolution(photo: np.ndarray) -> None:
    result = Transformation("d", "downscale", {"scale": 0.5}).apply(photo)
    assert result.shape[0] == photo.shape[0] // 2


def test_identity_is_a_true_control(photo: np.ndarray) -> None:
    np.testing.assert_array_equal(Transformation("c", "identity").apply(photo), photo)


def test_transforms_are_deterministic_given_a_seed(photo: np.ndarray) -> None:
    transformation = Transformation("n", "noise", {"sigma": 10.0})
    np.testing.assert_array_equal(
        transformation.apply(photo, seed=7), transformation.apply(photo, seed=7)
    )


def test_noise_differs_across_seeds(photo: np.ndarray) -> None:
    transformation = Transformation("n", "noise", {"sigma": 10.0})
    assert not np.array_equal(transformation.apply(photo, seed=1), transformation.apply(photo, 2))


def test_stronger_jpeg_lowers_quality(photo: np.ndarray) -> None:
    high = Transformation("a", "jpeg_compression", {"quality": 95}).apply(photo)
    low = Transformation("b", "jpeg_compression", {"quality": 30}).apply(photo)
    assert psnr(photo, high) > psnr(photo, low)


def test_unknown_transform_type_is_rejected(photo: np.ndarray) -> None:
    with pytest.raises(ConfigurationError, match="unknown transformation"):
        Transformation("x", "does_not_exist").apply(photo)


def test_invalid_parameters_are_rejected(photo: np.ndarray) -> None:
    with pytest.raises(ConfigurationError, match="crop ratio"):
        Transformation("x", "crop", {"ratio": 0.9}).apply(photo)
    with pytest.raises(ConfigurationError, match="resize scale"):
        Transformation("x", "resize", {"scale": 0.0}).apply(photo)


def test_pipeline_from_shipped_experiment_config(project_root, photo: np.ndarray) -> None:
    definitions = yaml.safe_load(
        (project_root / "configs" / "experiments.yaml").read_text(encoding="utf-8")
    )["transformations"]
    pipeline = TransformationPipeline.from_config(definitions, list(definitions), seed=3)
    results = pipeline.apply_each(photo)
    assert len(results) == len(definitions)
    assert all(image.dtype == np.uint8 for _, image in results)


def test_pipeline_rejects_undefined_names() -> None:
    with pytest.raises(ConfigurationError, match="not defined"):
        TransformationPipeline.from_config({"a": {"type": "identity"}}, ["b"])


def test_apply_each_uses_the_original_image(photo: np.ndarray) -> None:
    pipeline = TransformationPipeline(
        [Transformation("c1", "crop", {"ratio": 0.1}), Transformation("c2", "identity")]
    )
    results = pipeline.apply_each(photo)
    identity_result = [image for t, image in results if t.type == "identity"]
    np.testing.assert_array_equal(identity_result[0], photo)
    assert len(results) == 2


def test_apply_all_compounds_transformations(photo: np.ndarray) -> None:
    pipeline = TransformationPipeline(
        [
            Transformation("j", "jpeg_compression", {"quality": 40}),
            Transformation("b", "blur", {"sigma": 2.0}),
        ]
    )
    compounded = pipeline.apply_all(photo)
    single = Transformation("j", "jpeg_compression", {"quality": 40}).apply(photo)
    assert psnr(photo, compounded) < psnr(photo, single)


def test_pipeline_records_its_parameters() -> None:
    pipeline = TransformationPipeline([Transformation("j", "jpeg_compression", {"quality": 70})])
    payload = pipeline.to_dict()
    assert payload["seed"] == 42
    assert payload["transformations"][0]["params"] == {"quality": 70}
