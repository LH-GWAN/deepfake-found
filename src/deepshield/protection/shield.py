"""Swap shield: a perturbation that stops face swaps from carrying the person.

A GAN face swap takes one thing from the source photograph, the identity
embedding of the recogniser it was built around. ``inswapper_128`` reads the
ArcFace model of the ``buffalo_l`` pack (``w600k_r50``), whose weights are
public. The shield therefore attacks that encoder itself rather than a
stand-in, which is what made the difference in measurement: the same 8/255
budget aimed at a surrogate (FaceNet) left 20 of 20 faces recognised, while
aimed at the swapper's own encoder it left 0 of 30 swaps carrying the person,
through JPEG re-saves and halving (README, "노이즈 방어").

The encoder runs in PyTorch by executing its ONNX graph directly. The graph
uses six operator types, so no conversion package is needed; the result is
checked against onnxruntime in the tests.

The attack is projected gradient descent on the cosine similarity between
each face's embedding and its clean embedding, optimised in the photograph
itself through a differentiable warp onto the 112-pixel aligned frame, with
an expectation over small alignment jitter, resampling, blur and noise,
because the swapper re-detects and re-aligns the face on its own.

What it costs and what it does not do:

- The shielded photo no longer matches its owner by face either, in this
  pipeline or anywhere that uses the same recogniser. Tracking keeps the clean
  original for enrollment and finds reposts of the shielded file by its
  watermark and perceptual hash, which survive the perturbation.
- It was measured against one swapper. Swappers built on another encoder,
  deliberate purification (denoising, face restoration, regeneration) and
  LoRA fine-tuning were either not measured or not stopped.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from deepshield.config import ShieldConfig
from deepshield.exceptions import ModelNotAvailableError
from deepshield.face.backends import ARCFACE_TEMPLATE_112
from deepshield.logging_utils import get_logger
from deepshield.media import validate_rgb
from deepshield.types import DetectedFace

logger = get_logger(__name__)

FRAME = 112
SUPPORTED_OPERATORS = frozenset({"Conv", "BatchNormalization", "PRelu", "Add", "Flatten", "Gemm"})


def _torch() -> Any:
    """Return the torch module, or explain how to install it."""
    try:
        import torch
    except ImportError as exc:
        raise ModelNotAvailableError(
            "the swap shield needs PyTorch; install the 'torch' extra"
        ) from exc
    return torch


def pick_device() -> str:
    """Return the fastest available torch device."""
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def similarity_matrix(landmarks: np.ndarray) -> np.ndarray:
    """Return the least-squares similarity transform of five landmarks onto the ArcFace frame.

    This is Umeyama's closed form, the estimate insightface's ``norm_crop``
    uses, so the frame the shield optimises in is the frame the swapper reads.
    """
    source = np.asarray(landmarks, dtype=np.float64)[:5]
    target = ARCFACE_TEMPLATE_112.astype(np.float64)
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centred, target_centred = source - source_mean, target - target_mean
    covariance = target_centred.T @ source_centred / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1.0
    rotation = u @ sign @ vt
    variance = source_centred.var(axis=0).sum()
    scale = float(np.trace(np.diag(singular) @ sign) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return np.hstack([scale * rotation, translation[:, None]])


class OnnxGraph:
    """Executes a Conv/BatchNorm/PReLU/Add/Flatten/Gemm ONNX graph with torch operations."""

    def __init__(self, path: Path, device: str = "cpu") -> None:
        """Load the graph and its weights as frozen tensors on ``device``.

        Raises:
            ModelNotAvailableError: If the file, onnx or torch is missing, or the
                graph uses an operator this executor does not implement.

        """
        torch = _torch()
        try:
            import onnx
            from onnx import numpy_helper
        except ImportError as exc:
            raise ModelNotAvailableError("the swap shield needs the 'onnx' package") from exc
        if not Path(path).is_file():
            raise ModelNotAvailableError(
                f"missing {path}; run 'deepshield download-models' for the buffalo_l pack"
            )
        model = onnx.load(str(path))
        unknown = {node.op_type for node in model.graph.node} - SUPPORTED_OPERATORS
        if unknown:
            raise ModelNotAvailableError(f"unsupported ONNX operators: {sorted(unknown)}")
        self.nodes = list(model.graph.node)
        self.attributes = [
            {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            for node in self.nodes
        ]
        self.input_name = model.graph.input[0].name
        self.output_name = model.graph.output[0].name
        self.weights = {
            init.name: torch.from_numpy(numpy_helper.to_array(init).copy()).to(device)
            for init in model.graph.initializer
        }

    def __call__(self, x: Any) -> Any:
        """Run the graph on an NCHW batch and return its output."""
        functional = _torch().nn.functional
        values: dict[str, Any] = {self.input_name: x, **self.weights}
        for node, attrs in zip(self.nodes, self.attributes, strict=True):
            args = [values[name] for name in node.input]
            if node.op_type == "Conv":
                pads = attrs.get("pads", [0, 0, 0, 0])
                out = functional.conv2d(
                    args[0], args[1], args[2] if len(args) > 2 else None,
                    stride=attrs.get("strides", [1, 1]), padding=(pads[0], pads[1]),
                    dilation=attrs.get("dilations", [1, 1]), groups=attrs.get("group", 1),
                )
            elif node.op_type == "BatchNormalization":
                out = functional.batch_norm(
                    args[0], args[3], args[4], args[1], args[2],
                    training=False, eps=attrs.get("epsilon", 1e-5),
                )
            elif node.op_type == "PRelu":
                out = functional.prelu(args[0], args[1].reshape(-1))
            elif node.op_type == "Add":
                out = args[0] + args[1]
            elif node.op_type == "Flatten":
                out = args[0].flatten(attrs.get("axis", 1))
            else:
                out = functional.linear(args[0], args[1], args[2])
            values[node.output[0]] = out
        return values[self.output_name]


def sampling_grid(matrix: np.ndarray, height: int, width: int, jitter: Any) -> Any:
    """Return the ``grid_sample`` grid that reads the aligned frame out of the photo.

    ``jitter`` is a batch of (scale, rotation, dx, dy) perturbations of the
    alignment, standing in for the swapper's own re-detection.
    """
    torch = _torch()
    device = jitter.device
    full = np.vstack([matrix, [0.0, 0.0, 1.0]])
    inverse = torch.tensor(np.linalg.inv(full)[:2], dtype=torch.float32, device=device)
    batch = jitter.shape[0]
    ys, xs = torch.meshgrid(
        torch.arange(FRAME, device=device, dtype=torch.float32),
        torch.arange(FRAME, device=device, dtype=torch.float32),
        indexing="ij",
    )
    points = torch.stack([xs, ys], dim=-1).reshape(1, -1, 2).repeat(batch, 1, 1)
    centre = FRAME / 2.0
    scale, angle, dx, dy = jitter[:, 0:1], jitter[:, 1:2], jitter[:, 2:3], jitter[:, 3:4]
    px, py = points[..., 0] - centre, points[..., 1] - centre
    cos, sin = torch.cos(angle), torch.sin(angle)
    jx = (cos * px - sin * py) * scale + centre + dx
    jy = (sin * px + cos * py) * scale + centre + dy
    source_x = inverse[0, 0] * jx + inverse[0, 1] * jy + inverse[0, 2]
    source_y = inverse[1, 0] * jx + inverse[1, 1] * jy + inverse[1, 2]
    grid = torch.stack(
        [source_x / (width - 1) * 2 - 1, source_y / (height - 1) * 2 - 1], dim=-1
    )
    return grid.reshape(batch, FRAME, FRAME, 2)


def _blur(frames: Any, sigma: float) -> Any:
    """Return a separable Gaussian blur of an NCHW batch; ``sigma`` 0 is a no-op."""
    torch = _torch()
    if sigma <= 0.05:
        return frames
    radius = max(1, int(round(sigma * 3)))
    offsets = torch.arange(-radius, radius + 1, dtype=frames.dtype, device=frames.device)
    kernel = torch.exp(-(offsets**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = frames.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    vertical = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    padded = torch.nn.functional.pad(frames, (radius, radius, 0, 0), mode="reflect")
    frames = torch.nn.functional.conv2d(padded, horizontal, groups=channels)
    padded = torch.nn.functional.pad(frames, (0, 0, radius, radius), mode="reflect")
    return torch.nn.functional.conv2d(padded, vertical, groups=channels)


class SwapShield:
    """Perturbs every face in a photo against the encoder face swappers condition on."""

    def __init__(
        self,
        config: ShieldConfig,
        model_dir: Path,
        encoder: Any = None,
        device: str | None = None,
    ) -> None:
        """Load the encoder, or accept one (any callable NCHW -> embeddings) for tests."""
        self.config = config
        self.device = device or pick_device()
        self.encoder = encoder or OnnxGraph(Path(model_dir) / config.encoder_model, self.device)

    def _embed(self, frames: Any) -> Any:
        """Return L2-normalised embeddings of RGB frames in [0, 1]."""
        torch = _torch()
        return torch.nn.functional.normalize(self.encoder(frames * 2.0 - 1.0), dim=1)

    def shield(
        self, image: np.ndarray, faces: list[DetectedFace]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Return the shielded photo and a report of what was done.

        Every face with five landmarks is shielded at once; faces without them
        are left alone. With no such face the photo is returned unchanged and
        the report says so.
        """
        torch = _torch()
        functional = torch.nn.functional
        array = validate_rgb(image)
        started = time.perf_counter()
        points = [
            np.asarray(face.landmarks, dtype=np.float64)
            for face in faces
            if face.landmarks is not None and len(face.landmarks) >= 5
        ]
        report: dict[str, Any] = {
            "applied": False,
            "faces": len(points),
            "epsilon": self.config.epsilon,
            "steps": self.config.steps,
            "attacked_model": str(self.config.encoder_model),
        }
        if not points:
            report["reason"] = "no face with landmarks was detected, so nothing was shielded"
            return array, report

        generator = torch.Generator().manual_seed(self.config.seed)
        height, width = array.shape[:2]
        photo = (
            torch.from_numpy(np.array(array, copy=True))
            .permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        )
        matrices = [similarity_matrix(landmarks) for landmarks in points]
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        with torch.no_grad():
            anchors = [
                self._embed(
                    functional.grid_sample(
                        photo, sampling_grid(m, height, width, identity), align_corners=True
                    )
                )
                for m in matrices
            ]

        epsilon = float(self.config.epsilon)
        step = epsilon / 8.0
        batch = int(self.config.eot_samples)
        delta = torch.zeros_like(photo, requires_grad=True)
        for _ in range(int(self.config.steps)):
            jitter = torch.stack(
                [
                    torch.empty(batch).uniform_(0.95, 1.05, generator=generator),
                    torch.empty(batch).uniform_(-0.05, 0.05, generator=generator),
                    torch.empty(batch).uniform_(-2.0, 2.0, generator=generator),
                    torch.empty(batch).uniform_(-2.0, 2.0, generator=generator),
                ],
                dim=1,
            ).to(self.device)
            side = int(torch.randint(72, FRAME + 1, (1,), generator=generator).item())
            sigma = float(torch.empty(1).uniform_(0.0, 0.7, generator=generator).item())
            shielded = (photo + delta).clamp(0, 1).repeat(batch, 1, 1, 1)
            loss = 0.0
            for matrix, anchor in zip(matrices, anchors, strict=True):
                frames = functional.grid_sample(
                    shielded, sampling_grid(matrix, height, width, jitter), align_corners=True
                )
                frames = functional.interpolate(
                    functional.interpolate(
                        frames, size=(side, side), mode="bilinear", align_corners=False
                    ),
                    size=(FRAME, FRAME), mode="bilinear", align_corners=False,
                )
                frames = _blur(frames, sigma)
                noise = torch.randn(frames.shape, generator=generator).to(self.device) * 0.01
                frames = (frames + noise).clamp(0, 1)
                loss = loss + functional.cosine_similarity(self._embed(frames), anchor).mean()
            (gradient,) = torch.autograd.grad(loss, delta)
            with torch.no_grad():
                delta -= step * gradient.sign()
                delta.clamp_(-epsilon, epsilon)
                delta.copy_((photo + delta).clamp(0, 1) - photo)

        result = (photo + delta).clamp(0, 1).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        shielded_image = np.round(result * 255).astype(np.uint8)
        with torch.no_grad():
            after = (
                torch.from_numpy(shielded_image).permute(2, 0, 1).float().div(255)
                .unsqueeze(0).to(self.device)
            )
            similarities = [
                float(
                    functional.cosine_similarity(
                        self._embed(
                            functional.grid_sample(
                                after, sampling_grid(m, height, width, identity),
                                align_corners=True,
                            )
                        ),
                        anchor,
                    ).item()
                )
                for m, anchor in zip(matrices, anchors, strict=True)
            ]
        report.update(
            {
                "applied": True,
                "similarity_to_clean_face": [round(s, 4) for s in similarities],
                "seconds": round(time.perf_counter() - started, 2),
                "device": self.device,
            }
        )
        logger.info(
            "shielded %d face(s) in %.1fs, similarity to the clean face now %s",
            len(points), report["seconds"], report["similarity_to_clean_face"],
        )
        return shielded_image, report
