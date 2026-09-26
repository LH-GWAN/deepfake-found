"""Download a published deepfake detector and export it to ONNX.

The project already has an ONNX adapter for synthetic-media detection, so a
model is brought in by converting it rather than by adding a second inference
stack. Exporting also freezes the preprocessing into the graph: the normalisation
a published checkpoint expects is part of that checkpoint, and rediscovering it
at inference time is a reliable way to run a model on inputs it was never
trained for and then blame the model.

The exported graph therefore takes a plain ``1 x 3 x S x S`` tensor in ``[0, 1]``
RGB, exactly what the ``onnx`` backend produces, and performs the checkpoint's
own resizing and normalisation internally.

PyTorch is needed only here. Inference uses ONNX Runtime alone.

Usage:
    python scripts/fetch_deepfake_detector.py --model dima806/deepfake_vs_real_image_detection
    python scripts/fetch_deepfake_detector.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CANDIDATES: dict[str, str] = {
    "dima806": "dima806/deepfake_vs_real_image_detection",
    "wvolf": "Wvolf/ViT_Deepfake_Detection",
    "prithiv": "prithivMLmods/Deep-Fake-Detector-Model",
    "hemg": "Hemg/Deepfake-Detection",
}
FAKE_LABEL_HINTS = ("fake", "deepfake", "ai", "synthetic", "generated", "manipulated")


def resolve_fake_index(id2label: dict[Any, Any]) -> tuple[int, str]:
    """Return which output index means 'synthetic', and the label that said so.

    Getting this backwards silently inverts the detector, which looks like a
    model that is confidently wrong rather than one that is wired wrong, so the
    decision is made explicitly and recorded.

    Raises:
        SystemExit: If no label can be read as the synthetic class.

    """
    for key, label in id2label.items():
        lowered = str(label).lower()
        if any(hint in lowered for hint in FAKE_LABEL_HINTS) and "real" not in lowered:
            return int(key), str(label)
    raise SystemExit(f"could not tell which label means synthetic from {id2label}")


GEND_REPOSITORY = "yermandy/GenD_CLIP_L_14"
GEND_BACKBONE = "openai/clip-vit-large-patch14"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def legacy_exporter(torch: Any) -> dict[str, Any]:
    """Return the keyword that keeps ``torch.onnx.export`` on the TorchScript exporter.

    ``dynamo`` exists from torch 2.5 and defaults to True from 2.9; older
    releases, which ``pyproject.toml`` still admits, reject the keyword and
    only have the TorchScript exporter anyway.
    """
    import inspect

    return {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}


def export_gend(output: Path, input_size: int) -> int:
    """Export GenD (WACV 2026), a CLIP ViT-L/14 whose normalised features feed a linear head.

    The checkpoint's own loader pins transformers 4.56 and downloads the whole
    CLIP model only to overwrite it; the vision tower and the head are all the
    detector is, so they are rebuilt here from CLIP's configuration and GenD's
    weights, which hold every vision parameter. Index 1 is "fake": the authors'
    training code labels every source whose name contains "real" 0 and the rest 1.
    """
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import CLIPVisionConfig, CLIPVisionModel
    except ImportError:
        raise SystemExit(
            "exporting GenD needs torch, transformers, safetensors and huggingface_hub"
        ) from None

    print(f"downloading {GEND_REPOSITORY}", flush=True)
    weights = load_file(hf_hub_download(GEND_REPOSITORY, "model.safetensors"))
    prefix = "feature_extractor.vision_model."
    clip = CLIPVisionModel(CLIPVisionConfig.from_pretrained(GEND_BACKBONE))
    tower = getattr(clip, "vision_model", clip)
    tower.load_state_dict(
        {name[len(prefix):]: value for name, value in weights.items() if name.startswith(prefix)},
        strict=True,
    )

    class Wrapped(torch.nn.Module):
        """CLIP normalisation, the vision tower, L2 normalisation and GenD's linear head."""

        def __init__(self) -> None:
            super().__init__()
            self.tower = tower.eval()
            self.head = torch.nn.Linear(weights["model.linear.weight"].shape[1], 2)
            self.head.load_state_dict(
                {"weight": weights["model.linear.weight"], "bias": weights["model.linear.bias"]}
            )
            self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1))

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            """Take ``[0, 1]`` RGB and return real/fake logits.

            The bicubic resize only matters for an ``--input-size`` other than
            224. At the default it is an identity, and the resampling a face
            crop really gets is the ``onnx`` adapter's bilinear resize to 224.
            """
            resized = torch.nn.functional.interpolate(
                pixels, size=(224, 224), mode="bicubic", align_corners=False
            )
            features = self.tower(pixel_values=(resized - self.mean) / self.std).pooler_output
            return self.head(torch.nn.functional.normalize(features, p=2, dim=1))

    wrapped = Wrapped().eval()
    dummy = torch.rand(1, 3, input_size, input_size)
    with torch.no_grad():
        print(f"smoke test output shape {tuple(wrapped(dummy).shape)}")
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "deepfake_gend.onnx"
    torch.onnx.export(
        wrapped,
        dummy,
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        **legacy_exporter(torch),
    )
    metadata = {
        "repo": GEND_REPOSITORY,
        "alias": "gend",
        "input_size": input_size,
        "native_size": 224,
        "positive_index": 1,
        "labels": {"0": "Real", "1": "Fake"},
        "training_data": "FaceForensics++ c23 (the GenD paper, Tab. 2)",
        "preprocessing": (
            "CLIP normalisation is baked into the graph; its bicubic resize to 224 is an "
            f"identity at input_size {input_size}" if input_size == 224 else
            f"the graph resizes {input_size} to 224 bicubically and applies CLIP normalisation"
        ) + "; the onnx adapter resizes each crop to input_size bilinearly first",
    }
    (output / "deepfake_gend.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {onnx_path} and {output / 'deepfake_gend.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Download one checkpoint, wrap its preprocessing, and export it to ONNX."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dima806")
    parser.add_argument("--output", type=Path, default=Path("models"))
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    if args.list:
        for alias, repo in CANDIDATES.items():
            print(f"  {alias:10s} {repo}")
        print(f"  {'gend':10s} {GEND_REPOSITORY} (CLIP ViT-L/14, exported by its own path)")
        return 0
    if args.model in ("gend", GEND_REPOSITORY):
        return export_gend(args.output, args.input_size)

    repo = CANDIDATES.get(args.model, args.model)
    alias = next((a for a, r in CANDIDATES.items() if r == repo), repo.split("/")[-1])

    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
    except ImportError:
        raise SystemExit(
            "PyTorch and transformers are required to export a model; "
            "install them with: pip install torch transformers"
        ) from None

    print(f"downloading {repo}", flush=True)
    processor = AutoImageProcessor.from_pretrained(repo)
    model = AutoModelForImageClassification.from_pretrained(repo).eval()

    id2label = model.config.id2label
    fake_index, fake_label = resolve_fake_index(id2label)
    print(f"labels: {id2label}")
    print(f"synthetic class is index {fake_index} ({fake_label})")

    mean = torch.tensor(getattr(processor, "image_mean", [0.5, 0.5, 0.5])).view(1, 3, 1, 1)
    std = torch.tensor(getattr(processor, "image_std", [0.5, 0.5, 0.5])).view(1, 3, 1, 1)
    size = getattr(processor, "size", {}) or {}
    native = int(size.get("height") or size.get("shortest_edge") or args.input_size)
    print(f"checkpoint expects {native}px, mean={mean.flatten().tolist()}, "
          f"std={std.flatten().tolist()}")

    class Wrapped(torch.nn.Module):
        """Folds the checkpoint's own resizing and normalisation into the graph."""

        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
            self.native = native

        def forward(self, pixels: torch.Tensor) -> torch.Tensor:
            """Take ``[0, 1]`` RGB and return class logits."""
            resized = torch.nn.functional.interpolate(
                pixels, size=(self.native, self.native), mode="bilinear", align_corners=False
            )
            return self.model(pixel_values=(resized - self.mean) / self.std).logits

    wrapped = Wrapped().eval()
    dummy = torch.rand(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        logits = wrapped(dummy)
    print(f"smoke test output shape {tuple(logits.shape)}")

    args.output.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output / f"deepfake_{alias}.onnx"
    torch.onnx.export(
        wrapped,
        dummy,
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=14,
    )

    metadata = {
        "repo": repo,
        "alias": alias,
        "input_size": args.input_size,
        "native_size": native,
        "positive_index": fake_index,
        "labels": {str(k): str(v) for k, v in id2label.items()},
        "preprocessing": "resize and normalise are baked into the graph; feed [0,1] RGB",
    }
    (args.output / f"deepfake_{alias}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"\nwrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {args.output / f'deepfake_{alias}.json'}")
    print(
        "\nThis checkpoint's accuracy on its own training family says nothing about "
        "its accuracy here. Run scripts/evaluate_deepfake_detectors.py before "
        "trusting it."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
