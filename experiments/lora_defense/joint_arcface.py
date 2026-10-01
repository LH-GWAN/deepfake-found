"""The swap shield's objective as an extra step inside Mist v2's PGD, so one perturbation does both.

Mist v2 and ``protect --mode shield`` each work alone, but stacking the shield on a Mist
photograph undid most of Mist (README, "LoRA 방어 3차"). This adds the shield's loss to
Mist's own optimisation instead: at every PGD step Mist takes its signed step on the
Stable Diffusion objective, ``ArcFaceTerm.step`` adds a signed step toward lower cosine
similarity between each face's ArcFace embedding and its clean embedding, and the sum
goes through Mist's one projection onto the budget. The similarity is averaged over the
alignment jitter, resampling, blur and noise the shield uses, because the swapper
re-detects the face on its own. Once that average is at or below ``tau`` the step is
zero and the rest of the budget goes to Mist; shielding harder than needed would take
pixels from Mist for nothing.

Mist runs on Kaggle in its own environment (Python 3.10, torch 2.0.1) without this
repository's package, so this file carries a copy of what it needs from
``src/deepshield/protection/shield.py``: the ONNX-graph executor for the encoder
``inswapper_128`` conditions on (buffalo_l ``w600k_r50``), Umeyama's frame, the jittered
sampling grid and the blur. ``check_joint_arcface.py`` compares the copy with the shipped
module locally. ``kaggle_mist.py`` patches Mist to call this when ``MIST_JOINT_ONNX`` is
set; see ``from_environment`` for the other variables.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

FRAME = 112
# deepshield.face.backends.ARCFACE_TEMPLATE_112, insightface's ``arcface_dst``.
ARCFACE_TEMPLATE_112 = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)
SUPPORTED_OPERATORS = frozenset({"Conv", "BatchNormalization", "PRelu", "Add", "Flatten", "Gemm"})
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def similarity_matrix(landmarks: np.ndarray) -> np.ndarray:
    """Return the least-squares similarity transform of five landmarks onto the ArcFace frame."""
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

    def __init__(self, path: str, device: str = "cpu") -> None:
        """Load the graph and its weights as frozen tensors on ``device``."""
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(path))
        unknown = {node.op_type for node in model.graph.node} - SUPPORTED_OPERATORS
        if unknown:
            raise RuntimeError(f"unsupported ONNX operators: {sorted(unknown)}")
        self.nodes = list(model.graph.node)
        self.attributes = [
            {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            for node in self.nodes
        ]
        for node, attrs in zip(self.nodes, self.attributes):
            if node.op_type == "Conv":
                pads = list(attrs.get("pads", [0, 0, 0, 0]))
                if attrs.get("auto_pad", b"NOTSET") not in (b"NOTSET", "NOTSET") or \
                        len(pads) != 4 or pads[0] != pads[2] or pads[1] != pads[3]:
                    raise RuntimeError(f"unsupported Conv node {node.name}")
            if node.op_type == "Gemm" and (
                    attrs.get("transA", 0) != 0 or attrs.get("transB", 0) != 1
                    or attrs.get("alpha", 1.0) != 1.0 or attrs.get("beta", 1.0) != 1.0):
                raise RuntimeError(f"unsupported Gemm node {node.name}")
        self.input_name = model.graph.input[0].name
        self.output_name = model.graph.output[0].name
        self.weights = {
            init.name: torch.from_numpy(numpy_helper.to_array(init).copy()).to(device)
            for init in model.graph.initializer
        }

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Run the graph on an NCHW batch and return its output."""
        values: dict[str, Any] = {self.input_name: x, **self.weights}
        for node, attrs in zip(self.nodes, self.attributes):
            args = [values[name] for name in node.input]
            if node.op_type == "Conv":
                pads = attrs.get("pads", [0, 0, 0, 0])
                out = F.conv2d(
                    args[0], args[1], args[2] if len(args) > 2 else None,
                    stride=attrs.get("strides", [1, 1]), padding=(pads[0], pads[1]),
                    dilation=attrs.get("dilations", [1, 1]), groups=attrs.get("group", 1),
                )
            elif node.op_type == "BatchNormalization":
                out = F.batch_norm(
                    args[0], args[3], args[4], args[1], args[2],
                    training=False, eps=attrs.get("epsilon", 1e-5),
                )
            elif node.op_type == "PRelu":
                out = F.prelu(args[0], args[1].reshape(-1))
            elif node.op_type == "Add":
                out = args[0] + args[1]
            elif node.op_type == "Flatten":
                out = args[0].flatten(attrs.get("axis", 1))
            else:
                out = F.linear(args[0], args[1], args[2])
            values[node.output[0]] = out
        return values[self.output_name]


def sampling_grid(matrix: np.ndarray, height: int, width: int, jitter: torch.Tensor) -> torch.Tensor:
    """Return the ``grid_sample`` grid that reads the jittered aligned frame out of the photo."""
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


def blur(frames: torch.Tensor, sigma: float) -> torch.Tensor:
    """Return a separable Gaussian blur of an NCHW batch; ``sigma`` 0 is a no-op."""
    if sigma <= 0:
        return frames
    radius = max(1, int(2 * sigma))
    offsets = torch.arange(-radius, radius + 1, dtype=frames.dtype, device=frames.device)
    kernel = torch.exp(-(offsets**2) / (2 * sigma * sigma))
    kernel = kernel / kernel.sum()
    channels = frames.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    frames = F.conv2d(frames, horizontal, padding=(0, radius), groups=channels)
    return F.conv2d(frames, vertical, padding=(radius, 0), groups=channels)


class ArcFaceTerm:
    """The shield's loss for each image Mist perturbs, as one more signed PGD step."""

    def __init__(
        self,
        encoder: Any,
        faces: list[list[list[list[float]]]],
        originals: torch.Tensor,
        device: str,
        tau: float,
        step: float,
        eot_samples: int = 4,
        seed: int = 0,
    ) -> None:
        """Embed every face of every original, in Mist's [-1, 1] images.

        ``faces[i]`` holds the five-point landmarks of each face in image ``i``, in its
        pixels. The clean embeddings are taken from ``originals``, the images before
        any noise, the way the shield takes them from the photo it is given.
        """
        self.encoder = encoder
        self.device = device
        self.tau = float(tau)
        self.step_size = float(step)
        self.batch = int(eot_samples)
        self.generator = torch.Generator().manual_seed(seed)
        self.height, self.width = originals.shape[-2:]
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        self.frames: list[list[tuple[np.ndarray, torch.Tensor]]] = []
        with torch.no_grad():
            for image, points in zip(originals, faces):
                photo = (image[None].float().to(device) + 1) / 2
                entries = []
                for landmarks in points:
                    matrix = similarity_matrix(np.asarray(landmarks))
                    grid = sampling_grid(matrix, self.height, self.width, identity)
                    frame = F.grid_sample(photo, grid, align_corners=True)
                    entries.append((matrix, self._embed(frame)))
                self.frames.append(entries)
        self.active = [0] * len(self.frames)
        self.taken = [0] * len(self.frames)
        self.latest: list[float | None] = [None] * len(self.frames)

    def _embed(self, frames: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised embeddings of RGB frames in [0, 1]."""
        return F.normalize(self.encoder(frames * 2.0 - 1.0), dim=1)

    def similarities(self, image: torch.Tensor, index: int) -> list[float]:
        """Return each face's similarity to its clean embedding, without jitter."""
        photo = (image.float().to(self.device) + 1) / 2
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device)
        with torch.no_grad():
            return [
                float(F.cosine_similarity(self._embed(F.grid_sample(
                    photo, sampling_grid(matrix, self.height, self.width, identity),
                    align_corners=True)), anchor).item())
                for matrix, anchor in self.frames[index]
            ]

    def step(self, image: torch.Tensor, index: int) -> torch.Tensor:
        """Return the signed step toward lower similarity for image ``index``.

        ``image`` is 1x3xHxW in [-1, 1]; the step comes back on its device and dtype. It
        is zero when every face's expected similarity is already at or below ``tau``.
        The random draws follow the shield's order: jitter, resampling size, blur, then
        noise per face.
        """
        generator = self.generator
        batch = self.batch
        photo = ((image.detach().float().to(self.device) + 1) / 2).requires_grad_(True)
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
        repeated = photo.repeat(batch, 1, 1, 1)
        similarities = []
        for matrix, anchor in self.frames[index]:
            frames = F.grid_sample(
                repeated, sampling_grid(matrix, self.height, self.width, jitter),
                align_corners=True,
            )
            frames = F.interpolate(
                F.interpolate(frames, size=(side, side), mode="bilinear", align_corners=False),
                size=(FRAME, FRAME), mode="bilinear", align_corners=False,
            )
            frames = blur(frames, sigma)
            noise = torch.randn(frames.shape, generator=generator).to(self.device) * 0.01
            frames = (frames + noise).clamp(0, 1)
            similarities.append(F.cosine_similarity(self._embed(frames), anchor).mean())
        self.taken[index] += 1
        self.latest[index] = max(float(s.item()) for s in similarities)
        above = [s for s in similarities if float(s.item()) > self.tau]
        if not above:
            return torch.zeros_like(image)
        self.active[index] += 1
        (gradient,) = torch.autograd.grad(sum(above), photo)
        return (-self.step_size * gradient.sign()).to(image.device, image.dtype)

    def report(self, index: int) -> None:
        """Print how the last round went for image ``index`` and start counting afresh."""
        print(f"joint image {index}: expected similarity {self.latest[index]:.3f}, "
              f"shield steps {self.active[index]}/{self.taken[index]}", flush=True)
        self.active[index] = self.taken[index] = 0


def from_environment(instance_dir: str, originals: torch.Tensor) -> ArcFaceTerm:
    """Build the term for the images Mist loaded from ``instance_dir``.

    ``MIST_JOINT_ONNX`` is the encoder, ``MIST_JOINT_LANDMARKS`` a JSON file of
    ``{identity: {file name: [five-point landmarks per face]}}`` and
    ``MIST_JOINT_IDENTITY`` the key in it; ``MIST_JOINT_TAU`` and ``MIST_JOINT_STEP``
    (in Mist's [-1, 1] units) set the stopping similarity and the step. The images are
    matched to Mist's by listing the folder the way its ``load_data`` does, and must
    already be Mist's resolution, or the landmarks would not fall on the faces.
    """
    names = [n for n in os.listdir(instance_dir) if n.lower().endswith(IMAGE_SUFFIXES)]
    with open(os.environ["MIST_JOINT_LANDMARKS"]) as handle:
        faces = json.load(handle)[os.environ["MIST_JOINT_IDENTITY"]]
    height, width = originals.shape[-2:]
    for name in names:
        with Image.open(os.path.join(instance_dir, name)) as image:
            if image.size != (width, height):
                raise RuntimeError(f"{name} is {image.size}, not Mist's {(width, height)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tau = float(os.environ["MIST_JOINT_TAU"])
    step = float(os.environ["MIST_JOINT_STEP"])
    term = ArcFaceTerm(OnnxGraph(os.environ["MIST_JOINT_ONNX"], device),
                       [faces[name] for name in names], originals, device, tau, step)
    start = [term.similarities(originals[i:i + 1], i) for i in range(len(names))]
    print(f"joint: ArcFace term on {len(names)} images ({names}), tau {tau}, step {step}, "
          f"similarity at the start {start}", flush=True)
    return term
