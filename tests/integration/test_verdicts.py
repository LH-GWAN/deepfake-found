"""The verdict each real-world situation receives, end to end.

The mock embedder maps any pixel change to an unrelated vector, which makes a
watermarked copy of a photo look like a different person. These tests instead
embed a face crop by its coarse layout: a watermark barely moves it, a
different face moves it far. That is the property of a real recogniser the
verdicts depend on, without downloading one.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from deepshield.config import DeepShieldConfig, default_config
from deepshield.face.embedder import FaceEmbedder
from deepshield.media import load_image, save_image
from deepshield.pipeline.analysis_pipeline import DefaultAnalysisPipeline, subject_result
from deepshield.pipeline.protection_pipeline import DefaultProtectionPipeline
from deepshield.storage import build_identity_repository
from deepshield.types import (
    FaceEmbedding,
    IdentityProfile,
    ModelInfo,
    RiskLevel,
    SimilarityResult,
    Verdict,
)
from tests.conftest import synthetic_photo

pytestmark = pytest.mark.integration

GRID = 8


class CoarseLayoutEmbedder(FaceEmbedder):
    """Embeds a crop as its mean-removed 8x8 grey-level layout."""

    name = "coarse"

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(name="coarse-layout", version="1", backend=self.name)

    @property
    def dimension(self) -> int:
        return GRID * GRID

    def embed(self, face_image: np.ndarray) -> FaceEmbedding:
        self.validate_face_image(face_image)
        small = np.asarray(
            Image.fromarray(face_image)
            .convert("L")
            .resize((GRID, GRID), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ).ravel()
        vector = self.l2_normalize(small - small.mean())
        return FaceEmbedding(vector=vector, model=self.model_info)


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
                }
            ),
            "detection": base.detection.model_copy(
                update={"deepfake": base.detection.deepfake.model_copy(update={"backend": "mock"})}
            ),
            "protection": base.protection.model_copy(
                update={
                    "watermark": base.protection.watermark.model_copy(update={"backend": "dct"})
                }
            ),
        }
    )


@pytest.fixture
def pipeline(config: DeepShieldConfig) -> DefaultAnalysisPipeline:
    return DefaultAnalysisPipeline(config, embedder=CoarseLayoutEmbedder())


def face_region(image: np.ndarray, margin: float = 0.12) -> tuple[slice, slice]:
    """Return the mock detector's centred box, widened past the aligner's margin."""
    height, width = image.shape[:2]
    pad_y, pad_x = int(height / 2 * margin), int(width / 2 * margin)
    return (
        slice(height // 4 - pad_y, height // 4 + height // 2 + pad_y),
        slice(width // 4 - pad_x, width // 4 + width // 2 + pad_x),
    )


def replace_face(image: np.ndarray, donor: np.ndarray) -> np.ndarray:
    rows, cols = face_region(image)
    swapped = image.copy()
    swapped[rows, cols] = donor[rows, cols]
    return swapped


def jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    return np.asarray(Image.open(buffer).convert("RGB"))


def enroll(pipeline: DefaultAnalysisPipeline, config: DeepShieldConfig, image: np.ndarray) -> None:
    face = pipeline.detector.detect(image)[0]
    vector = pipeline.embedder.embed(pipeline.aligner.align(image, face).image).vector
    build_identity_repository(config).save(
        IdentityProfile(
            user_id="u1",
            reference_embeddings=vector[None, :],
            centroid_embedding=vector,
            image_count=1,
            model=pipeline.embedder.model_info,
            embedding_dimension=pipeline.embedder.dimension,
        )
    )


@pytest.fixture
def protected(
    config: DeepShieldConfig, pipeline: DefaultAnalysisPipeline, tmp_path: Path
) -> tuple[Path, dict]:
    source = synthetic_photo(seed=5, size=512)
    enroll(pipeline, config, source)
    report = DefaultProtectionPipeline(config).protect(
        save_image(source, tmp_path / "me.png"), "u1", "instagram"
    )
    return Path(report["protected_path"]), report


def analyse(pipeline: DefaultAnalysisPipeline, image: np.ndarray, path: Path):
    return pipeline.analyze_image(save_image(image, path), "u1")


def test_the_protected_file_itself_is_an_own_copy(pipeline, protected) -> None:
    path, _ = protected
    record = pipeline.analyze_image(path, "u1")
    assert record.risk.verdict is Verdict.OWN_COPY
    assert record.risk.risk_level is RiskLevel.LOW
    assert record.summary.startswith("This is a copy of a photograph registered by 'u1'.")


def test_a_recompressed_repost_is_an_own_copy(pipeline, protected, tmp_path: Path) -> None:
    path, report = protected
    record = analyse(pipeline, jpeg(load_image(path), 90), tmp_path / "repost.png")
    assert record.watermark_code == report["watermark"]["code"]
    assert record.risk.verdict is Verdict.OWN_COPY
    assert any("instagram" in line for line in record.risk.explanation)


def test_another_face_on_the_protected_photo_is_an_alteration(
    pipeline, protected, tmp_path: Path
) -> None:
    path, report = protected
    swapped = replace_face(load_image(path), synthetic_photo(seed=91, size=512))
    record = analyse(pipeline, swapped, tmp_path / "target_swap.png")
    assert record.matched_asset_id == report["asset_id"]
    assert record.risk.signals["owner_face_in_original"] is True
    assert record.risk.verdict is Verdict.OWN_ALTERED
    assert record.risk.risk_level is RiskLevel.HIGH


def test_an_alteration_is_unverified_once_the_protected_file_is_gone(
    pipeline, protected, tmp_path: Path
) -> None:
    path, _ = protected
    swapped = replace_face(load_image(path), synthetic_photo(seed=91, size=512))
    path.unlink()
    record = analyse(pipeline, swapped, tmp_path / "target_swap.png")
    assert record.risk.signals["owner_face_in_original"] is None
    assert record.risk.verdict is Verdict.OWN_UNVERIFIED
    assert record.risk.risk_level is RiskLevel.MEDIUM


def test_the_users_face_on_unregistered_content_is_an_identity_match(
    pipeline, protected, tmp_path: Path
) -> None:
    """The source direction: the user's face, someone else's picture."""
    source = synthetic_photo(seed=5, size=512)
    composite = replace_face(synthetic_photo(seed=200, size=512), source)
    record = analyse(pipeline, composite, tmp_path / "source_swap.png")
    assert record.matched_asset_id is None
    assert record.risk.verdict is Verdict.IDENTITY_MATCH
    assert record.risk.risk_level is RiskLevel.MEDIUM
    assert "genuine or synthetic is undetermined" in record.summary


def test_an_unrelated_image_is_unrelated(pipeline, protected, tmp_path: Path) -> None:
    record = analyse(pipeline, synthetic_photo(seed=91, size=512), tmp_path / "other.png")
    assert record.risk.verdict is Verdict.UNRELATED


def test_the_repost_now_ranks_below_the_face_match(pipeline, protected, tmp_path: Path) -> None:
    """The regression that motivated verdicts, measured through the whole pipeline."""
    path, _ = protected
    repost = pipeline.analyze_image(path, "u1")
    composite = replace_face(
        synthetic_photo(seed=200, size=512), synthetic_photo(seed=5, size=512)
    )
    match = analyse(pipeline, composite, tmp_path / "source_swap.png")
    order = list(RiskLevel)
    assert order.index(match.risk.risk_level) > order.index(repost.risk.risk_level)


def result(user: str, similarity: float, decision: str) -> SimilarityResult:
    return SimilarityResult(
        matched_user_id=user,
        similarity=similarity,
        aggregation="max",
        metric="cosine",
        decision=decision,
        is_candidate=decision != "no_match",
        is_high_confidence=decision == "high_confidence",
    )


def test_a_face_that_better_matches_someone_else_is_not_the_subjects() -> None:
    ranked = [result("u2", 0.8, "high_confidence"), result("u1", 0.6, "high_confidence")]
    best = subject_result([ranked], "u1")
    assert best is not None
    assert best.decision == "ambiguous"


def test_the_subjects_strongest_face_wins() -> None:
    faces = [
        [result("u1", 0.4, "candidate")],
        [result("u1", 0.7, "high_confidence")],
        [result("u1", 0.1, "no_match")],
    ]
    best = subject_result(faces, "u1")
    assert best is not None
    assert best.similarity == 0.7
    assert subject_result(faces, "u3") is None


def test_faces_that_could_not_be_compared_are_inconclusive(
    config, pipeline, tmp_path: Path
) -> None:
    """An identity enrolled with another embedder cannot be said to be absent."""
    other_model = np.ones(GRID, dtype=np.float32) / np.sqrt(GRID)
    build_identity_repository(config).save(
        IdentityProfile(
            user_id="u1",
            reference_embeddings=other_model[None, :],
            centroid_embedding=other_model,
            image_count=1,
            model=ModelInfo(name="other", version="1", backend="other"),
            embedding_dimension=GRID,
        )
    )
    record = analyse(pipeline, synthetic_photo(seed=5, size=512), tmp_path / "probe.png")
    assert record.faces
    assert record.risk.verdict is Verdict.INCONCLUSIVE


def shrink(image: np.ndarray, scale: float) -> np.ndarray:
    height, width = image.shape[:2]
    size = (int(round(width * scale)), int(round(height * scale)))
    return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS))


def test_a_shrunken_repost_is_read_at_its_registered_size(
    pipeline, protected, tmp_path: Path
) -> None:
    """A phone upload is smaller than the original; restoring the size restores the grid."""
    path, report = protected
    small = jpeg(shrink(load_image(path), 0.7), 90)
    assert not pipeline.watermarker.detect(small).detected
    record = analyse(pipeline, small, tmp_path / "upload.png")
    assert record.watermark_code == report["watermark"]["code"]
    assert record.risk.verdict is Verdict.OWN_COPY
    assert any("resized copy" in line for line in record.limitations)


def write_video(frames: list[np.ndarray], path: Path, crf: int = 18) -> Path:
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is needed to encode a test video")
    height, width = frames[0].shape[:2]
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", "10", "-i", "-", "-c:v", "libx264",
            "-crf", str(crf), "-pix_fmt", "yuv420p", str(path),
        ],
        input=b"".join(np.ascontiguousarray(frame).tobytes() for frame in frames),
        check=True,
    )
    return path


def letterboxed(image: np.ndarray, width: int, height: int, scale: float) -> np.ndarray:
    photo = shrink(image, scale)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    top, left = (height - photo.shape[0]) // 2, (width - photo.shape[1]) // 2
    frame[top : top + photo.shape[0], left : left + photo.shape[1]] = photo
    return frame


def test_a_registered_photo_shown_in_a_video_is_found_by_its_watermark(
    pipeline, config, protected, tmp_path: Path
) -> None:
    from deepshield.video.processor import DefaultVideoProcessor

    path, report = protected
    # The mock detector boxes the centre of the whole frame, so the frame fits
    # the photo closely; a real detector finds the face wherever it is.
    still = letterboxed(load_image(path), 362, 362, 0.7)
    video = write_video([still] * 20, tmp_path / "slideshow.mp4")
    record = DefaultVideoProcessor(config, analysis=pipeline).analyze(video, "u1")
    assert record.matched_asset_id == report["asset_id"]
    assert record.watermark_code == report["watermark"]["code"]
    assert record.risk.signals["asset_match_basis"] == "watermark code"
    assert record.risk.verdict is Verdict.OWN_COPY
    assert record.summary.startswith("This video shows a photograph registered by 'u1'.")
    assert any("resembles registered asset" in line for line in record.limitations)


def test_a_video_without_a_registered_photo_says_what_was_checked(
    pipeline, config, protected, tmp_path: Path
) -> None:
    from deepshield.video.processor import DefaultVideoProcessor

    other = letterboxed(synthetic_photo(seed=91, size=512), 640, 400, 0.7)
    video = write_video([other] * 20, tmp_path / "other.mp4")
    record = DefaultVideoProcessor(config, analysis=pipeline).analyze(video, "u1")
    assert record.matched_asset_id is None
    assert any("No sampled frame resembled" in line for line in record.limitations)
