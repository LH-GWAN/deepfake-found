"""Experiment framework: reproducible, machine-readable robustness benchmarks.

Every experiment writes one row per (image, transformation) pair containing the
measurement and everything needed to reproduce it: seed, git commit, Python
version, model names and versions, and the transformation parameters. A result
that cannot be traced back to the exact configuration that produced it is not a
result.

Three benchmarks are implemented against the shared transformation engine, so a
number from one is directly comparable with a number from another:

face recognition robustness
    Does the identity signal survive compression, rescaling and cropping?
watermark robustness
    Does source attribution survive the same treatment, and at what image cost?
combined protection
    Do watermarking and cloaking interfere with each other?

The recurring finding this framework exists to expose is that different signals
die under different transformations, which is the entire argument for combining
them rather than relying on any one.
"""

from __future__ import annotations

import csv
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from deepshield import __version__
from deepshield.config import DeepShieldConfig
from deepshield.logging_utils import get_logger
from deepshield.media import load_image, validate_rgb
from deepshield.quality import psnr, ssim
from deepshield.transforms import TransformationPipeline
from deepshield.types import WatermarkPayload

logger = get_logger(__name__)


def git_commit() -> str:
    """Return the current git commit, or a marker when not in a repository."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "not-a-git-repository"


def environment(config: DeepShieldConfig) -> dict[str, Any]:
    """Capture everything needed to reproduce a run."""
    return {
        "deepshield_version": __version__,
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "random_seed": config.runtime.random_seed,
        "device": config.runtime.device,
        "face_detector": config.face.detector.backend,
        "face_aligner": config.face.aligner.backend,
        "face_embedder": config.face.embedder.backend,
        "deepfake_detector": config.detection.deepfake.backend,
        "watermark_backend": config.protection.watermark.backend,
        "watermark_strength": config.protection.watermark.strength,
        "recorded_at": datetime.now(UTC).isoformat(),
    }


@dataclass
class ExperimentResult:
    """Rows plus the environment that produced them."""

    experiment_id: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)

    def write_csv(self, path: Path) -> Path:
        """Write the rows to CSV, with the environment flattened onto each row.

        Repeating the environment on every row is deliberate: rows get filtered,
        merged and pasted into other tools, and a provenance column that travels
        with the data survives that in a way that a separate header does not.
        """
        if not self.rows:
            raise ValueError(f"experiment '{self.experiment_id}' produced no rows")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        provenance = {f"env_{key}": value for key, value in self.environment.items()}
        enriched = [{**row, **provenance} for row in self.rows]
        fieldnames = list(enriched[0].keys())
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(enriched)
        logger.info("wrote %d rows to %s", len(enriched), destination)
        return destination

    def summary(self, value_column: str) -> dict[str, float]:
        """Return simple statistics for one numeric column."""
        values = [
            float(row[value_column])
            for row in self.rows
            if row.get(value_column) is not None
        ]
        if not values:
            return {}
        array = np.asarray(values)
        return {
            "count": float(array.size),
            "mean": float(array.mean()),
            "min": float(array.min()),
            "max": float(array.max()),
        }


def load_transformations(
    config_path: Path, names: list[str] | None = None, seed: int = 42
) -> TransformationPipeline:
    """Build a transformation pipeline from ``configs/experiments.yaml``."""
    import yaml

    definitions = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["transformations"]
    selected = names or list(definitions)
    return TransformationPipeline.from_config(definitions, selected, seed=seed)


class FaceRobustnessExperiment:
    """Measures how identity similarity survives image transformations.

    Independent variable: the transformation. Dependent variables: cosine
    similarity to the untransformed reference, and whether a face was detected
    at all. Controls: face model, aligner, reference image and thresholds, all
    recorded in the environment block.

    Detection failure is reported as a distinct outcome rather than as a
    similarity of zero, because "the pipeline never saw a face" and "the pipeline
    saw a different person" are different failures with different fixes.
    """

    experiment_id = "face_recognition_robustness"

    def __init__(self, config: DeepShieldConfig) -> None:
        """Build the face pipeline used for the measurements."""
        from deepshield.face.aligner import build_aligner
        from deepshield.face.detector import build_detector
        from deepshield.face.embedder import build_embedder

        self.config = config
        self.detector = build_detector(config.face.detector)
        self.aligner = build_aligner(config.face.aligner)
        self.embedder = build_embedder(config.face.embedder)

    def _embed(self, image: np.ndarray) -> np.ndarray | None:
        """Return the embedding of the most confident face, or ``None``."""
        faces = self.detector.detect(image)
        if not faces:
            return None
        return self.embedder.embed(self.aligner.align(image, faces[0]).image).vector

    def run(self, image_paths: list[Path], pipeline: TransformationPipeline) -> ExperimentResult:
        """Measure every image under every transformation."""
        rows: list[dict[str, Any]] = []
        threshold = self.config.thresholds.face_similarity.candidate_threshold

        for path in image_paths:
            image = validate_rgb(load_image(path))
            reference = self._embed(image)
            if reference is None:
                logger.warning("no face detected in reference image %s", path)
                continue

            for transformation, transformed in pipeline.apply_each(image):
                probe = self._embed(transformed)
                similarity = None if probe is None else float(np.dot(reference, probe))
                rows.append(
                    {
                        "experiment_id": self.experiment_id,
                        "image_id": path.name,
                        "transformation": transformation.name,
                        "transformation_type": transformation.type,
                        "parameters": str(transformation.params),
                        "face_detected": probe is not None,
                        "face_similarity": None if similarity is None else round(similarity, 6),
                        "recognition_success": (
                            None if similarity is None else similarity >= threshold
                        ),
                        "candidate_threshold": threshold,
                        "psnr": round(psnr(image, transformed), 4),
                        "ssim": round(ssim(image, transformed), 6),
                    }
                )
        return ExperimentResult(self.experiment_id, rows, environment(self.config))


class WatermarkRobustnessExperiment:
    """Measures whether source attribution survives image transformations.

    Independent variable: the transformation. Dependent variables: detection,
    decoder confidence, bit accuracy and whether the recovered code is correct.

    Bit accuracy is reported alongside the binary detection flag because it
    degrades smoothly under compression and shows how much margin remains, while
    detection is a cliff. It is measured on the unshifted block grid, so under a
    geometric transformation it sits at chance even when the decoder
    resynchronises and recovers the code exactly - the two columns have to be
    read together.

    A wrong recovered code is tracked separately and is the one outcome that
    would be worse than failure, since it would attribute a leak to the wrong
    channel.
    """

    experiment_id = "watermark_robustness"

    def __init__(self, config: DeepShieldConfig) -> None:
        """Build the watermark backend used for the measurements."""
        from deepshield.protection.watermark import build_watermarker

        self.config = config
        self.watermarker = build_watermarker(config.protection.watermark)

    def run(self, image_paths: list[Path], pipeline: TransformationPipeline) -> ExperimentResult:
        """Watermark every image, then measure recovery under each transformation."""
        from deepshield.protection.watermark import CODE_BITS, DctWatermarker

        rows: list[dict[str, Any]] = []
        for index, path in enumerate(image_paths):
            original = validate_rgb(load_image(path))
            payload = WatermarkPayload(
                version=1,
                user_token="benchmark-token",
                asset_id=f"benchmark-{index}",
                distribution_id="benchmark",
            )
            expected_code = f"{payload.code(CODE_BITS):08x}"
            try:
                marked = self.watermarker.embed(original, payload)
            except Exception as exc:
                logger.warning("could not watermark %s: %s", path.name, exc)
                continue

            embed_psnr = psnr(original, marked)
            embed_ssim = ssim(original, marked)

            for transformation, transformed in pipeline.apply_each(marked):
                result = self.watermarker.detect(transformed)
                accuracy = (
                    self.watermarker.bit_accuracy(transformed, payload)
                    if isinstance(self.watermarker, DctWatermarker)
                    else None
                )
                rows.append(
                    {
                        "experiment_id": self.experiment_id,
                        "image_id": path.name,
                        "transformation": transformation.name,
                        "transformation_type": transformation.type,
                        "parameters": str(transformation.params),
                        "watermark_detected": result.detected,
                        "watermark_confidence": round(result.confidence, 6),
                        "bit_accuracy": None if accuracy is None else round(accuracy, 6),
                        "recovered_code": result.watermark_code,
                        "code_correct": result.watermark_code == expected_code,
                        "false_attribution": bool(
                            result.detected and result.watermark_code != expected_code
                        ),
                        "embed_psnr": round(embed_psnr, 4),
                        "embed_ssim": round(embed_ssim, 6),
                        "psnr_after_transform": round(psnr(marked, transformed), 4),
                    }
                )
        return ExperimentResult(self.experiment_id, rows, environment(self.config))


class CombinedProtectionExperiment:
    """Compares the four protection conditions on the same images.

    Conditions: original, watermark only, adversarial only, and both. The point
    is to find out whether the two layers interfere - whether cloaking noise
    destroys the watermark, or watermark embedding undoes the cloak - rather
    than assuming they compose.
    """

    experiment_id = "combined_protection"

    def __init__(self, config: DeepShieldConfig) -> None:
        """Build the watermark, face and cloaking components."""
        from deepshield.face.aligner import build_aligner
        from deepshield.face.detector import build_detector
        from deepshield.face.embedder import build_embedder
        from deepshield.protection.adversarial import SpsaAdversarialProtector
        from deepshield.protection.watermark import build_watermarker

        self.config = config
        self.watermarker = build_watermarker(config.protection.watermark)
        self.detector = build_detector(config.face.detector)
        self.aligner = build_aligner(config.face.aligner)
        self.embedder = build_embedder(config.face.embedder)
        self.protector = SpsaAdversarialProtector(config.protection.adversarial)

    def _similarity(self, reference: np.ndarray | None, image: np.ndarray) -> float | None:
        """Return the identity similarity of an image against a reference vector."""
        if reference is None:
            return None
        faces = self.detector.detect(image)
        if not faces:
            return None
        probe = self.embedder.embed(self.aligner.align(image, faces[0]).image).vector
        return float(np.dot(reference, probe))

    def run(self, image_paths: list[Path]) -> ExperimentResult:
        """Build all four conditions per image and measure each."""
        from deepshield.protection.watermark import CODE_BITS

        rows: list[dict[str, Any]] = []
        for index, path in enumerate(image_paths):
            original = validate_rgb(load_image(path))
            faces = self.detector.detect(original)
            reference = (
                self.embedder.embed(self.aligner.align(original, faces[0]).image).vector
                if faces
                else None
            )
            payload = WatermarkPayload(
                version=1,
                user_token="combined-token",
                asset_id=f"combined-{index}",
                distribution_id="benchmark",
            )
            expected = f"{payload.code(CODE_BITS):08x}"

            cloaked = self.protector.protect(original, [self.embedder])
            conditions = {
                "original": original,
                "watermark_only": self.watermarker.embed(original, payload),
                "adversarial_only": cloaked,
                "watermark_and_adversarial": self.watermarker.embed(cloaked, payload),
            }

            for name, variant in conditions.items():
                detection = self.watermarker.detect(variant)
                rows.append(
                    {
                        "experiment_id": self.experiment_id,
                        "image_id": path.name,
                        "condition": name,
                        "psnr": None
                        if psnr(original, variant) == float("inf")
                        else round(psnr(original, variant), 4),
                        "ssim": round(ssim(original, variant), 6),
                        "face_similarity": self._similarity(reference, variant),
                        "watermark_detected": detection.detected,
                        "watermark_confidence": round(detection.confidence, 6),
                        "code_correct": detection.watermark_code == expected,
                    }
                )
        return ExperimentResult(self.experiment_id, rows, environment(self.config))
