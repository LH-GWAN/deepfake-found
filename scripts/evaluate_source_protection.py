r"""Measure whether a perturbation aimed at the swapper's own encoder stops tracking.

A GAN face swap takes exactly one thing from the source photograph: the
identity embedding of the recogniser it was built around. ``inswapper_128``
reads ``buffalo_l``'s ArcFace (``w600k_r50``), whose weights ship with the
model pack. The earlier cloaking measurement attacked a different model
(FaceNet) and lost most of its effect in transfer; this one attacks the
encoder the swapper actually uses, so there is no transfer to lose on the
swap itself.

What it answers, per perturbation budget:

protection
    Does a swap made from the protected photograph still carry the person?
    Measured as the donor's rank among 30 enrolled identities and the share
    above the high-confidence threshold, with ArcFace (the attacked model, as
    the pipeline uses it) and SFace (a recogniser nobody attacked).
tracking
    The same numbers are what the tracking side relies on: identity matching
    finds a swap's donor at 169/169 on clean sources. Protection that works
    is expected to take tracking with it, and this measures by how much.
the photo itself
    Is the protected photograph still recognised as its owner? That decides
    whether reposts of it can still be matched by face.
an attacker who re-saves
    The protected photograph is JPEG-compressed at quality 85 and 70, or
    halved in size, before the swap, as a re-upload would do. Deliberate
    purification (denoising, diffusion regeneration) is not measured here.

The attack is PGD on cosine similarity to the clean embedding, optimised in
the photograph itself through a differentiable warp onto the 112-pixel
aligned frame, with an expectation over small alignment jitter, resampling,
blur and noise, since the swapper re-detects and re-aligns the face.

ArcFace runs in PyTorch by executing its ONNX graph directly: the graph
uses six operator types, so no conversion package is needed, and the
outputs are checked against onnxruntime before anything is measured.

Usage:
    python scripts/evaluate_source_protection.py --identities 30
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_verdicts import GanSwapper
from learned_watermark import gaussian_blur, pick_device

from deepshield.config import FaceEmbedderConfig, load_config
from deepshield.face.aligner import build_aligner
from deepshield.face.backends import ARCFACE_TEMPLATE_112
from deepshield.face.detector import build_detector
from deepshield.face.embedder import build_embedder
from deepshield.media import load_image
from deepshield.quality import psnr, ssim

ARCFACE = Path("models/insightface/models/buffalo_l/w600k_r50.onnx")
SIZE = 112
STEPS = 200
EOT_BATCH = 4


class OnnxGraph(torch.nn.Module):
    """Executes a Conv/BatchNorm/PReLU/Add/Flatten/Gemm ONNX graph with torch ops."""

    def __init__(self, path: Path) -> None:
        """Load the graph and its weights as frozen tensors."""
        import onnx
        from onnx import numpy_helper

        super().__init__()
        model = onnx.load(str(path))
        self.nodes = list(model.graph.node)
        self.input_name = model.graph.input[0].name
        self.output_name = model.graph.output[0].name
        self.weights = {
            init.name: torch.from_numpy(numpy_helper.to_array(init).copy())
            for init in model.graph.initializer
        }
        self.attributes = [
            {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            for node in self.nodes
        ]
        supported = {"Conv", "BatchNormalization", "PRelu", "Add", "Flatten", "Gemm"}
        unknown = {node.op_type for node in self.nodes} - supported
        if unknown:
            raise SystemExit(f"unsupported ONNX operators: {sorted(unknown)}")

    def to(self, device: str) -> OnnxGraph:  # type: ignore[override]
        """Move every weight to ``device``."""
        self.weights = {name: tensor.to(device) for name, tensor in self.weights.items()}
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the graph on an NCHW batch."""
        values: dict[str, torch.Tensor] = {self.input_name: x, **self.weights}
        for node, attrs in zip(self.nodes, self.attributes, strict=True):
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
                slope = args[1].reshape(-1)
                out = F.prelu(args[0], slope)
            elif node.op_type == "Add":
                out = args[0] + args[1]
            elif node.op_type == "Flatten":
                out = torch.flatten(args[0], attrs.get("axis", 1))
            else:  # Gemm with transB=1
                out = F.linear(args[0], args[1], args[2])
            values[node.output[0]] = out
        return values[self.output_name]


def check_against_onnxruntime(model: OnnxGraph, device: str) -> float:
    """Return the largest absolute difference from onnxruntime on random input."""
    import onnxruntime

    session = onnxruntime.InferenceSession(str(ARCFACE), providers=["CPUExecutionProvider"])
    sample = np.random.default_rng(0).uniform(-1, 1, (2, 3, SIZE, SIZE)).astype(np.float32)
    reference = session.run(None, {session.get_inputs()[0].name: sample})[0]
    with torch.no_grad():
        ours = model(torch.from_numpy(sample).to(device)).cpu().numpy()
    return float(np.abs(reference - ours).max())


def norm_matrix(kps: np.ndarray) -> np.ndarray:
    """Return insightface's similarity transform from the photo onto the 112 frame."""
    from skimage import transform as trans

    tform = trans.SimilarityTransform()
    tform.estimate(np.asarray(kps, dtype=np.float32), ARCFACE_TEMPLATE_112)
    return np.asarray(tform.params[:2], dtype=np.float64)


def sampling_grid(
    matrix: np.ndarray, height: int, width: int, jitter: torch.Tensor
) -> torch.Tensor:
    """Return the grid_sample grid mapping the 112 frame back into the photo.

    ``jitter`` is a batch of (scale, rotation, dx, dy) perturbations of the
    alignment, standing in for the swapper's own re-detection.
    """
    full = np.vstack([matrix, [0.0, 0.0, 1.0]])
    inverse = torch.tensor(np.linalg.inv(full)[:2], dtype=torch.float32, device=jitter.device)
    batch = jitter.shape[0]
    ys, xs = torch.meshgrid(
        torch.arange(SIZE, device=jitter.device, dtype=torch.float32),
        torch.arange(SIZE, device=jitter.device, dtype=torch.float32),
        indexing="ij",
    )
    points = torch.stack([xs, ys], dim=-1).reshape(1, -1, 2).repeat(batch, 1, 1)
    centre = SIZE / 2.0
    scale, angle, dx, dy = jitter[:, 0:1], jitter[:, 1:2], jitter[:, 2:3], jitter[:, 3:4]
    px, py = points[..., 0] - centre, points[..., 1] - centre
    cos, sin = torch.cos(angle), torch.sin(angle)
    jx = (cos * px - sin * py) * scale + centre + dx
    jy = (sin * px + cos * py) * scale + centre + dy
    source_x = inverse[0, 0] * jx + inverse[0, 1] * jy + inverse[0, 2]
    source_y = inverse[1, 0] * jx + inverse[1, 1] * jy + inverse[1, 2]
    grid = torch.stack([source_x / (width - 1) * 2 - 1, source_y / (height - 1) * 2 - 1], dim=-1)
    return grid.reshape(batch, SIZE, SIZE, 2)


def embed_frames(model: OnnxGraph, frames: torch.Tensor) -> torch.Tensor:
    """Return L2-normalised ArcFace embeddings of RGB frames in [0, 1]."""
    return F.normalize(model(frames * 2.0 - 1.0), dim=1)


def protect(
    model: OnnxGraph, image: np.ndarray, kps: np.ndarray, epsilon: float, device: str
) -> np.ndarray:
    """Return the photo with an L-infinity perturbation that moves its ArcFace identity."""
    height, width = image.shape[:2]
    matrix = norm_matrix(kps)
    photo = torch.from_numpy(image).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    exact = sampling_grid(matrix, height, width, identity)
    with torch.no_grad():
        anchor = embed_frames(model, F.grid_sample(photo, exact, align_corners=True))
    delta = torch.zeros_like(photo, requires_grad=True)
    step = epsilon / 8.0
    for _ in range(STEPS):
        jitter = torch.stack(
            [
                torch.empty(EOT_BATCH).uniform_(0.95, 1.05),
                torch.empty(EOT_BATCH).uniform_(-0.05, 0.05),
                torch.empty(EOT_BATCH).uniform_(-2.0, 2.0),
                torch.empty(EOT_BATCH).uniform_(-2.0, 2.0),
            ],
            dim=1,
        ).to(device)
        grid = sampling_grid(matrix, height, width, jitter)
        protected = (photo + delta).clamp(0, 1).repeat(EOT_BATCH, 1, 1, 1)
        frames = F.grid_sample(protected, grid, align_corners=True)
        side = int(torch.randint(72, SIZE + 1, (1,)).item())
        frames = F.interpolate(
            F.interpolate(frames, size=(side, side), mode="bilinear", align_corners=False),
            size=(SIZE, SIZE), mode="bilinear", align_corners=False,
        )
        frames = gaussian_blur(frames, float(torch.empty(1).uniform_(0.0, 0.7).item()))
        frames = (frames + torch.randn_like(frames) * 0.01).clamp(0, 1)
        loss = F.cosine_similarity(embed_frames(model, frames), anchor).mean()
        (gradient,) = torch.autograd.grad(loss, delta)
        with torch.no_grad():
            delta -= step * gradient.sign()
            delta.clamp_(-epsilon, epsilon)
            delta.copy_((photo + delta).clamp(0, 1) - photo)
    result = (photo + delta).clamp(0, 1).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    return np.round(result * 255).astype(np.uint8)


def halve(image: np.ndarray) -> np.ndarray:
    """Return the image at half its width and height."""
    height, width = image.shape[:2]
    return np.asarray(
        Image.fromarray(image).resize((width // 2, height // 2), Image.Resampling.LANCZOS)
    )


RESAVES = {
    "jpeg85": lambda image: jpeg(image, 85),
    "jpeg70": lambda image: jpeg(image, 70),
    "half": halve,
}


def jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    """Round-trip an RGB array through JPEG."""
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=quality)
    return np.asarray(Image.open(buffer).convert("RGB"))


class Recognisers:
    """The pipeline's ArcFace (with its flip averaging) and an unattacked SFace."""

    def __init__(self) -> None:
        """Build the production face stack plus SFace."""
        config = load_config()
        self.detector = build_detector(config.face.detector)
        self.aligner = build_aligner(config.face.aligner)
        self.embedders = {
            "arcface": build_embedder(config.face.embedder),
            "sface": build_embedder(
                FaceEmbedderConfig(backend="opencv_sface", model_name="sface", flip_tta=False)
            ),
        }
        thresholds = config.thresholds.face_similarity
        self.high = thresholds.high_confidence_threshold
        self.candidate = thresholds.candidate_threshold

    def __call__(self, image: np.ndarray) -> dict[str, np.ndarray] | None:
        """Return every recogniser's embedding of the most confident face."""
        faces = self.detector.detect(image)
        if not faces:
            return None
        face = max(faces, key=lambda f: f.detection_confidence)
        aligned = self.aligner.align(image, face).image
        return {name: e.embed(aligned).vector for name, e in self.embedders.items()}


def score(
    probe: dict[str, np.ndarray] | None,
    galleries: dict[str, dict[str, dict[str, np.ndarray]]],
    donor: str,
    exclude: str,
) -> dict[str, dict[str, Any]] | None:
    """Return, per recogniser, the donor's similarity and whether it ranks first."""
    if probe is None:
        return None
    result: dict[str, dict[str, Any]] = {}
    for model in probe:
        best = {
            identity: max(
                float(probe[model] @ vectors[model])
                for name, vectors in gallery.items()
                if name != exclude
            )
            for identity, gallery in galleries.items()
        }
        result[model] = {
            "donor_similarity": best[donor],
            "donor_first": max(best, key=lambda k: best[k]) == donor,
        }
    return result


def main(argv: list[str] | None = None) -> int:
    """Protect one photo per identity at each budget and measure what survives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument("--identities", type=int, default=30)
    parser.add_argument("--epsilon", type=int, nargs="+", default=[2, 4, 8, 16],
                        help="L-infinity budgets in 1/255 units")
    parser.add_argument("--models", type=Path, default=Path("models/insightface"))
    parser.add_argument(
        "--inswapper", type=Path, default=Path("models/inswapper/inswapper_128.onnx")
    )
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    args = parser.parse_args(argv)

    device = pick_device()
    arcface = OnnxGraph(ARCFACE).to(device)
    mismatch = check_against_onnxruntime(arcface, device)
    print(f"torch ArcFace vs onnxruntime: max |diff| = {mismatch:.2e}", flush=True)
    if mismatch > 1e-2:
        raise SystemExit("the torch execution of ArcFace does not match onnxruntime")

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(args.faces.glob("*.png")):
        grouped[path.stem.rsplit("_", 1)[0]].append(path)
    identities = sorted(name for name, paths in grouped.items() if len(paths) >= 3)
    identities = identities[: args.identities]

    recognise = Recognisers()
    swapper = GanSwapper(args.models, args.inswapper)
    galleries: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for identity in identities:
        galleries[identity] = {}
        for path in grouped[identity]:
            vectors = recognise(load_image(path))
            if vectors is not None:
                galleries[identity][path.name] = vectors

    conditions = ["clean"] + [
        f"eps{e}{suffix}" for e in args.epsilon for suffix in ["", *(f"_{r}" for r in RESAVES)]
    ]
    rows: dict[str, list[dict[str, Any]]] = {c: [] for c in conditions}
    photo_rows: dict[str, list[dict[str, Any]]] = {c: [] for c in conditions if c != "clean"}
    quality: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for index, donor in enumerate(identities):
        source_path = grouped[donor][0]
        target_identity = identities[(index + 1) % len(identities)]
        target = load_image(grouped[target_identity][0])
        source = load_image(source_path)
        face = swapper._largest(source)
        if face is None:
            continue
        variants: dict[str, np.ndarray] = {"clean": source}
        for epsilon in args.epsilon:
            protected = protect(arcface, source, face.kps, epsilon / 255.0, device)
            variants[f"eps{epsilon}"] = protected
            for name, resave in RESAVES.items():
                variants[f"eps{epsilon}_{name}"] = resave(protected)
            quality[f"eps{epsilon}"].append((psnr(source, protected), ssim(source, protected)))

        for condition, photo in variants.items():
            if condition != "clean":
                photo_rows[condition].append(
                    {"score": score(recognise(photo), galleries, donor, source_path.name)}
                )
            swapped = swapper(photo, target)
            scored = score(
                None if swapped is None else recognise(swapped),
                galleries, donor, source_path.name,
            )
            rows[condition].append({"score": scored})
        print(f"{index + 1}/{len(identities)} {donor}", flush=True)

    def summarise(entries: list[dict[str, Any]]) -> dict[str, Any]:
        scored = [e["score"] for e in entries if e["score"] is not None]
        out: dict[str, Any] = {"scored": len(scored), "no_face": len(entries) - len(scored)}
        for model in ("arcface", "sface"):
            sims = [s[model]["donor_similarity"] for s in scored]
            out[model] = {
                "donor_first": int(sum(s[model]["donor_first"] for s in scored)),
                "median_donor_similarity": round(float(np.median(sims)), 4) if sims else None,
            }
            if model == "arcface" and sims:
                out[model]["above_high_confidence"] = int(sum(v >= recognise.high for v in sims))
                out[model]["above_candidate"] = int(sum(v >= recognise.candidate for v in sims))
        return out

    report: dict[str, Any] = {
        "question": "does a perturbation on the swapper's own encoder stop a swap carrying "
        "the person, and does tracking survive it?",
        "attacked_model": "buffalo_l w600k_r50 (the ArcFace inswapper conditions on)",
        "attack": f"PGD {STEPS} steps, L-infinity, EOT over alignment jitter, resampling, "
        "blur and noise",
        "swapper": "inswapper_128",
        "identities": len(identities),
        "arcface_thresholds": {"candidate": recognise.candidate, "high": recognise.high},
        "torch_vs_onnxruntime_max_diff": mismatch,
        "swaps": {c: summarise(v) for c, v in rows.items()},
        "protected_photo_itself": {c: summarise(v) for c, v in photo_rows.items()},
        "image_quality": {
            c: {
                "psnr": round(float(np.mean([q[0] for q in v])), 2),
                "ssim": round(float(np.mean([q[1] for q in v])), 4),
            }
            for c, v in quality.items()
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "source_protection.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
