"""End-to-end wiring of the Phase 0 mock components.

This does not test recognition quality. It proves that configuration, the
registries and the component contracts compose into a runnable chain, so later
phases only have to swap backends.
"""

from __future__ import annotations

import numpy as np
import pytest

from deepshield.config import DeepShieldConfig, load_config
from deepshield.detection.deepfake import build_deepfake_detector
from deepshield.face.aligner import build_aligner
from deepshield.face.detector import build_detector
from deepshield.face.embedder import build_embedder
from deepshield.protection.watermark import build_watermarker
from deepshield.types import WatermarkPayload

pytestmark = pytest.mark.integration


@pytest.fixture
def loaded_config(project_root) -> DeepShieldConfig:
    return load_config(
        config_path=project_root / "configs" / "default.yaml",
        thresholds_path=project_root / "configs" / "thresholds.yaml",
        environ={},
    )


def test_detect_align_embed_chain(loaded_config: DeepShieldConfig, rgb_image: np.ndarray) -> None:
    """The chain must compose with mock backends, independent of any model download."""
    mock_config = loaded_config.model_copy(
        update={
            "face": loaded_config.face.model_copy(
                update={
                    "detector": loaded_config.face.detector.model_copy(
                        update={"backend": "mock"}
                    ),
                    "aligner": loaded_config.face.aligner.model_copy(
                        update={"backend": "mock"}
                    ),
                    "embedder": loaded_config.face.embedder.model_copy(
                        update={"backend": "mock"}
                    ),
                }
            )
        }
    )
    detector = build_detector(mock_config.face.detector)
    aligner = build_aligner(mock_config.face.aligner)
    embedder = build_embedder(mock_config.face.embedder)

    faces = detector.detect(rgb_image)
    assert faces

    aligned = aligner.align(rgb_image, faces[0])
    assert aligned.image.shape[0] == mock_config.face.aligner.output_size

    embedding = embedder.embed(aligned.image)
    assert embedding.dimension == mock_config.face.embedder.embedding_dimension
    assert float(np.linalg.norm(embedding.vector)) == pytest.approx(1.0, abs=1e-5)


def test_candidate_gating_order_is_expressible(
    loaded_config: DeepShieldConfig, rgb_image: np.ndarray
) -> None:
    detector = build_detector(
        loaded_config.face.detector.model_copy(update={"backend": "mock"})
    )
    deepfake = build_deepfake_detector(loaded_config.detection.deepfake)

    faces = detector.detect(rgb_image)
    threshold = loaded_config.thresholds.face_similarity.candidate_threshold
    similarity = 0.0

    if similarity >= threshold:
        result = deepfake.predict_image(rgb_image)
        assert 0.0 <= result.score <= 1.0
    else:
        assert faces


def test_protection_chain_round_trip(
    loaded_config: DeepShieldConfig, large_photo: np.ndarray
) -> None:
    watermarker = build_watermarker(loaded_config.protection.watermark)
    payload = WatermarkPayload(version=1, user_token="token", asset_id="asset-1")

    protected = watermarker.embed(large_photo, payload)
    assert protected.shape == large_photo.shape

    detection = watermarker.detect(protected)
    assert detection.detected is True
    assert detection.watermark_code == f"{payload.code():08x}"

    assert watermarker.detect(large_photo).detected is False


def test_every_selected_backend_is_registered(loaded_config: DeepShieldConfig) -> None:
    """Selecting a backend name in configuration must resolve to a factory.

    Instantiation may still need model weights, so only the lookup is asserted.
    """
    from deepshield.detection.deepfake import DEEPFAKE_REGISTRY
    from deepshield.face.aligner import ALIGNER_REGISTRY
    from deepshield.face.detector import DETECTOR_REGISTRY
    from deepshield.face.embedder import EMBEDDER_REGISTRY
    from deepshield.protection.watermark import WATERMARK_REGISTRY

    assert loaded_config.face.detector.backend in DETECTOR_REGISTRY
    assert loaded_config.face.aligner.backend in ALIGNER_REGISTRY
    assert loaded_config.face.embedder.backend in EMBEDDER_REGISTRY
    assert loaded_config.detection.deepfake.backend in DEEPFAKE_REGISTRY
    assert loaded_config.protection.watermark.backend in WATERMARK_REGISTRY
