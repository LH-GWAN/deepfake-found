"""Real face detection and embedding backends.

Three families are wired in behind the Phase 0 interfaces:

``opencv_yunet`` / ``opencv_sface``
    A small ONNX detector and embedder from the OpenCV model zoo. They download
    in seconds, run on CPU, and need no extra Python package beyond OpenCV, so
    they are the default working configuration.
``insightface``
    SCRFD detection plus ArcFace embeddings from the InsightFace package. Higher
    accuracy and the reference implementation this field is benchmarked against,
    at the cost of a large install and a several-hundred-megabyte model pack.
``onnx_arcface``
    Any ArcFace-compatible ONNX file the user supplies, loaded directly through
    ONNX Runtime, so a newer or differently licensed model can be dropped in
    without waiting for this project to add support for it.

Both embedder families output an L2-normalised vector, which makes cosine
similarity a plain dot product. Their dimensionalities differ - SFace produces
128 numbers, ArcFace 512 - so embeddings from different models are never
comparable, and the model name and version travel with every vector to make an
accidental cross-model comparison detectable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from deepshield.config import FaceAlignerConfig, FaceDetectorConfig, FaceEmbedderConfig
from deepshield.exceptions import ModelNotAvailableError
from deepshield.face.aligner import ALIGNER_REGISTRY, FaceAligner
from deepshield.face.detector import DETECTOR_REGISTRY, FaceDetector
from deepshield.face.embedder import EMBEDDER_REGISTRY, FaceEmbedder
from deepshield.logging_utils import get_logger
from deepshield.media import validate_rgb
from deepshield.models import resolve_model
from deepshield.types import AlignedFace, BoundingBox, DetectedFace, FaceEmbedding, ModelInfo

logger = get_logger(__name__)

ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _require(module: str, extra: str) -> Any:
    """Import an optional dependency or raise a clear installation hint."""
    try:
        return __import__(module)
    except ImportError as exc:
        raise ModelNotAvailableError(
            f"'{module}' is not installed; install the '{extra}' extra: "
            f"pip install -e '.[{extra}]'"
        ) from exc


class YuNetFaceDetector(FaceDetector):
    """OpenCV YuNet ONNX detector producing boxes, scores and five landmarks.

    YuNet is a compact anchor-free detector. It is far weaker than SCRFD on
    small, rotated or occluded faces, which is exactly the kind of degradation
    the Phase 13 robustness benchmark is meant to expose rather than hide.
    """

    name = "opencv_yunet"

    def __init__(
        self, config: FaceDetectorConfig | None = None, model_dir: Path | None = None
    ) -> None:
        """Load the YuNet model, downloading it when necessary."""
        self.config = config or FaceDetectorConfig()
        self.cv2 = _require("cv2", "face")
        self.model_path = resolve_model(
            "yunet",
            Path(model_dir or "models"),
            explicit_path=self.config.model_path,
            allow_download=self.config.allow_download,
        )
        self._detector = self.cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (320, 320),
            self.config.detection_confidence_threshold,
            self.config.nms_threshold,
            self.config.max_faces,
        )

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect faces and return them sorted by descending confidence."""
        original = self.validate_image(image)
        array, scale = self.downscale_for_detection(original, self.config.max_detection_side)
        height, width = array.shape[:2]
        self._detector.setInputSize((width, height))
        bgr = array[:, :, ::-1]
        _, raw = self._detector.detect(np.ascontiguousarray(bgr))
        if raw is None:
            return []
        min_side = self.config.min_face_size * scale

        faces: list[DetectedFace] = []
        for row in raw:
            x, y, box_w, box_h = (float(v) for v in row[:4])
            confidence = float(row[14])
            if confidence < self.config.detection_confidence_threshold:
                continue
            if min(box_w, box_h) < min_side:
                continue
            landmarks = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
            faces.append(
                self.rescale_face(
                    DetectedFace(
                        bbox=BoundingBox(x, y, x + box_w, y + box_h),
                        detection_confidence=confidence,
                        landmarks=landmarks,
                    ),
                    scale,
                )
            )
        faces.sort(key=lambda face: face.detection_confidence, reverse=True)
        return faces[: self.config.max_faces]


class InsightFaceDetector(FaceDetector):
    """SCRFD detection from the InsightFace model pack."""

    name = "insightface"

    def __init__(
        self, config: FaceDetectorConfig | None = None, model_dir: Path | None = None
    ) -> None:
        """Prepare an InsightFace analysis app restricted to detection."""
        self.config = config or FaceDetectorConfig()
        insightface = _require("insightface", "face")
        _require("onnxruntime", "face")
        self._app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=str(Path(model_dir or "models") / "insightface"),
            allowed_modules=["detection"],
        )
        self._app.prepare(ctx_id=-1, det_size=(640, 640))

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect faces with SCRFD and return them sorted by confidence."""
        original = self.validate_image(image)
        array, scale = self.downscale_for_detection(original, self.config.max_detection_side)
        detections = self._app.get(np.ascontiguousarray(array[:, :, ::-1]))
        min_side = self.config.min_face_size * scale
        faces: list[DetectedFace] = []
        for detection in detections:
            confidence = float(detection.det_score)
            x1, y1, x2, y2 = (float(v) for v in detection.bbox)
            if confidence < self.config.detection_confidence_threshold:
                continue
            if min(x2 - x1, y2 - y1) < min_side:
                continue
            landmarks = getattr(detection, "kps", None)
            faces.append(
                self.rescale_face(
                    DetectedFace(
                        bbox=BoundingBox(x1, y1, x2, y2),
                        detection_confidence=confidence,
                        landmarks=None
                        if landmarks is None
                        else np.asarray(landmarks, np.float32),
                    ),
                    scale,
                )
            )
        faces.sort(key=lambda face: face.detection_confidence, reverse=True)
        return faces[: self.config.max_faces]


class LandmarkAligner(FaceAligner):
    """Similarity-transform alignment onto the canonical ArcFace template.

    Five landmarks - both eyes, nose tip and both mouth corners - are mapped onto
    fixed coordinates with a rotation, uniform scale and translation. This
    removes in-plane rotation and scale differences that would otherwise show up
    as a drop in similarity between two photos of the same person.

    Falls back to a padded crop when a detector supplied no landmarks, which is
    weaker and is logged rather than silently accepted.
    """

    name = "landmark"

    def __init__(self, config: FaceAlignerConfig | None = None) -> None:
        """Store alignment configuration and load OpenCV."""
        self.config = config or FaceAlignerConfig()
        self.cv2 = _require("cv2", "face")

    def _template(self) -> np.ndarray:
        """Return the canonical landmark template scaled to the output size."""
        return ARCFACE_TEMPLATE_112 * (self.config.output_size / 112.0)

    def align(self, image: np.ndarray, face: DetectedFace) -> AlignedFace:
        """Warp the face onto the canonical template using its landmarks."""
        array = validate_rgb(image)
        size = self.config.output_size

        if face.landmarks is None or len(face.landmarks) < 5:
            from deepshield.face.aligner import SimpleCropAligner

            logger.debug("no landmarks available, falling back to crop alignment")
            return SimpleCropAligner(self.config, margin=self.config.margin).align(array, face)

        source = np.asarray(face.landmarks[:5], dtype=np.float32)
        matrix, _ = self.cv2.estimateAffinePartial2D(
            source, self._template(), method=self.cv2.LMEDS
        )
        if matrix is None:
            from deepshield.face.aligner import SimpleCropAligner

            logger.debug("landmark transform could not be estimated, falling back to crop")
            return SimpleCropAligner(self.config, margin=self.config.margin).align(array, face)

        warped = self.cv2.warpAffine(
            array, matrix, (size, size), borderValue=(0, 0, 0)
        )
        return AlignedFace(image=np.asarray(warped, dtype=np.uint8), source=face, output_size=size)


class SFaceEmbedder(FaceEmbedder):
    """OpenCV SFace ONNX embedder producing 128-dimensional vectors."""

    name = "opencv_sface"
    dimension_value = 128

    def __init__(
        self, config: FaceEmbedderConfig | None = None, model_dir: Path | None = None
    ) -> None:
        """Load the SFace model, downloading it when necessary."""
        self.config = config or FaceEmbedderConfig()
        self.cv2 = _require("cv2", "face")
        self.model_path = resolve_model(
            "sface",
            Path(model_dir or "models"),
            explicit_path=self.config.model_path,
            allow_download=self.config.allow_download,
        )
        self._recognizer = self.cv2.FaceRecognizerSF.create(str(self.model_path), "")

    @property
    def model_info(self) -> ModelInfo:
        """Return the SFace model metadata."""
        return ModelInfo(
            name="sface",
            version="2021dec",
            backend=self.name,
            training_dataset="unknown",
            input_size=112,
        )

    @property
    def dimension(self) -> int:
        """SFace always produces 128 dimensions regardless of configuration."""
        return self.dimension_value

    def embed(self, face_image: np.ndarray) -> FaceEmbedding:
        """Embed an aligned 112x112 face crop."""
        array = self.validate_face_image(face_image)
        if array.shape[0] != 112 or array.shape[1] != 112:
            array = np.asarray(
                self.cv2.resize(array, (112, 112), interpolation=self.cv2.INTER_LINEAR),
                dtype=np.uint8,
            )
        bgr = np.ascontiguousarray(array[:, :, ::-1])
        vector = np.asarray(self._recognizer.feature(bgr), dtype=np.float32).ravel()
        if self.config.normalize:
            vector = self.l2_normalize(vector)
        return FaceEmbedding(vector=vector, model=self.model_info, normalized=self.config.normalize)


class InsightFaceEmbedder(FaceEmbedder):
    """ArcFace embeddings from the InsightFace recognition model."""

    name = "insightface"
    dimension_value = 512

    RECOGNITION_FILES = ("w600k_r50.onnx", "w600k_mbf.onnx", "glintr100.onnx")

    def __init__(
        self, config: FaceEmbedderConfig | None = None, model_dir: Path | None = None
    ) -> None:
        """Load the ArcFace recognition model, fetching the pack if necessary.

        The recognition network is loaded directly rather than through
        InsightFace's ``FaceAnalysis`` helper, which insists on also loading a
        detector. Embedding and detection are separate stages here, and pairing
        an ArcFace embedder with a different detector is exactly the kind of
        substitution the registry exists to allow.
        """
        self.config = config or FaceEmbedderConfig()
        insightface = _require("insightface", "face")
        _require("onnxruntime", "face")
        self._pack = self.config.insightface_pack
        root = Path(model_dir or "models") / "insightface"
        pack_dir = root / "models" / self._pack

        model_file: Path | None
        if self.config.model_path is not None:
            model_file = Path(self.config.model_path)
        else:
            model_file = self._find_recognition_model(pack_dir)
            if model_file is None:
                self._download_pack(insightface, root)
                model_file = self._find_recognition_model(pack_dir)
        if model_file is None or not model_file.is_file():
            raise ModelNotAvailableError(
                f"no InsightFace recognition model found under {pack_dir}; "
                "run 'deepshield download-models --insightface' or set "
                "face.embedder.model_path"
            )

        self.model_path = model_file
        self._model = insightface.model_zoo.get_model(str(model_file))
        self._model.prepare(ctx_id=-1)

    @classmethod
    def _find_recognition_model(cls, pack_dir: Path) -> Path | None:
        """Return the recognition ONNX file inside a downloaded model pack."""
        for filename in cls.RECOGNITION_FILES:
            candidate = pack_dir / filename
            if candidate.is_file():
                return candidate
        return None

    def _download_pack(self, insightface: Any, root: Path) -> None:
        """Trigger InsightFace's own downloader for the configured model pack."""
        logger.info("downloading InsightFace model pack %s", self._pack)
        try:
            insightface.app.FaceAnalysis(name=self._pack, root=str(root))
        except Exception as exc:
            raise ModelNotAvailableError(
                f"could not download InsightFace pack '{self._pack}': {exc}"
            ) from exc

    @property
    def model_info(self) -> ModelInfo:
        """Return the ArcFace model metadata."""
        return ModelInfo(
            name=f"arcface-{self._pack}",
            version="insightface",
            backend=self.name,
            training_dataset="glint360k or webface600k depending on pack",
            input_size=112,
        )

    @property
    def dimension(self) -> int:
        """ArcFace packs used here produce 512 dimensions."""
        return self.dimension_value

    def embed(self, face_image: np.ndarray) -> FaceEmbedding:
        """Embed an aligned 112x112 face crop."""
        array = self.validate_face_image(face_image)
        bgr = np.ascontiguousarray(array[:, :, ::-1])
        vector = np.asarray(self._model.get_feat(bgr), dtype=np.float32).ravel()
        if self.config.normalize:
            vector = self.l2_normalize(vector)
        return FaceEmbedding(vector=vector, model=self.model_info, normalized=self.config.normalize)


class OnnxArcFaceEmbedder(FaceEmbedder):
    """Any ArcFace-compatible ONNX model loaded directly through ONNX Runtime.

    Expects a model taking ``1 x 3 x 112 x 112`` input normalised to
    ``[-1, 1]`` in RGB order, which is the ArcFace convention. Supplying a model
    trained under a different convention silently degrades similarity, so the
    model name recorded on every embedding is the only defence against comparing
    vectors that were never comparable.
    """

    name = "onnx_arcface"

    def __init__(
        self, config: FaceEmbedderConfig | None = None, model_dir: Path | None = None
    ) -> None:
        """Create an inference session for the configured ONNX file."""
        self.config = config or FaceEmbedderConfig()
        onnxruntime = _require("onnxruntime", "face")
        if self.config.model_path is None:
            raise ModelNotAvailableError(
                "the onnx_arcface backend needs face.embedder.model_path in configuration"
            )
        path = Path(self.config.model_path)
        if not path.is_file():
            raise ModelNotAvailableError(f"ONNX model not found: {path}")
        self.model_path = path
        self._session = onnxruntime.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._dimension = int(self._session.get_outputs()[0].shape[-1])

    @property
    def model_info(self) -> ModelInfo:
        """Return metadata naming the exact ONNX file in use."""
        return ModelInfo(
            name=self.config.model_name or self.model_path.stem,
            version=self.config.model_version,
            backend=self.name,
            training_dataset=None,
            input_size=112,
        )

    @property
    def dimension(self) -> int:
        """Return the output width reported by the ONNX graph."""
        return self._dimension

    def embed(self, face_image: np.ndarray) -> FaceEmbedding:
        """Embed an aligned face crop through the ONNX session."""
        array = self.validate_face_image(face_image)
        from PIL import Image

        if array.shape[0] != 112 or array.shape[1] != 112:
            array = np.asarray(
                Image.fromarray(array).resize((112, 112), Image.Resampling.BILINEAR),
                dtype=np.uint8,
            )
        tensor = (array.astype(np.float32) - 127.5) / 127.5
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        outputs = self._session.run(None, {self._input_name: tensor})
        vector = np.asarray(outputs[0], dtype=np.float32).ravel()
        if self.config.normalize:
            vector = self.l2_normalize(vector)
        return FaceEmbedding(vector=vector, model=self.model_info, normalized=self.config.normalize)


DETECTOR_REGISTRY.register("opencv_yunet", YuNetFaceDetector)
DETECTOR_REGISTRY.register("insightface", InsightFaceDetector)
ALIGNER_REGISTRY.register("landmark", LandmarkAligner)
EMBEDDER_REGISTRY.register("opencv_sface", SFaceEmbedder)
EMBEDDER_REGISTRY.register("insightface", InsightFaceEmbedder)
EMBEDDER_REGISTRY.register("onnx_arcface", OnnxArcFaceEmbedder)
