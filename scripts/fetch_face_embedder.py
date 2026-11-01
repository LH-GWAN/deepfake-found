r"""Download a face recogniser and export it to ONNX for the ``onnx_arcface`` backend.

The pipeline recognises faces with ArcFace (buffalo_l ``w600k_r50``). On a face
shrunk to about 55 pixels and re-encoded at crf 35 it puts only 44% of genuine
probes above the high-confidence threshold, and five cheaper fixes made that
worse. AdaFace (Kim et al., CVPR 2022) was trained for exactly that regime: its
margin scales with each training image's quality, so low-quality faces are not
forced into the same geometry as clean ones. This script brings it in as a
second recogniser that ``scripts/compare_face_embedders.py`` can screen.

The export takes ``1 x 3 x 112 x 112`` RGB in ``[-1, 1]`` and returns the
512-dimensional feature before normalisation, which is the ``onnx_arcface``
convention, so no new backend is needed. The weights are the author's CVLFace
port of the IR-101 WebFace12M model, whose ``model.yaml`` reads RGB normalised
by ``(x - 0.5) / 0.5``: the same convention, so the graph is the network alone.
They are read from ``model.safetensors``, never from the repository's pickle, and
the network is rebuilt here from the published iresnet definition rather than by
importing the repository's code. Every weight must load with its shape and
nothing may be left over; only BatchNorm's training counters may be absent.

WebFace12M, the training data, is licensed for non-commercial research. Like
every model weight in this repository, the export is not committed.

PyTorch is needed only here. Inference uses ONNX Runtime alone.

Usage:
    python scripts/fetch_face_embedder.py --model adaface --output models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ADAFACE_REPOSITORY = "minchul/cvlface_adaface_ir101_webface12m"
IR101_UNITS = ((64, 64, 3), (64, 128, 13), (128, 256, 30), (256, 512, 3))


def build_ir101() -> Any:
    """Return the IR-101 backbone AdaFace uses, as published in CVLFace's iresnet.

    Module order and nesting follow the original exactly, because the state
    dict's keys are positional (``body.12.res_layer.3.weight``).
    """
    from torch import nn

    class Flatten(nn.Module):
        def forward(self, x: Any) -> Any:
            return x.view(x.size(0), -1)

    class BasicBlockIR(nn.Module):
        def __init__(self, in_channel: int, depth: int, stride: int) -> None:
            super().__init__()
            if in_channel == depth:
                self.shortcut_layer: nn.Module = nn.MaxPool2d(1, stride)
            else:
                self.shortcut_layer = nn.Sequential(
                    nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                    nn.BatchNorm2d(depth),
                )
            self.res_layer = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
                nn.BatchNorm2d(depth),
                nn.PReLU(depth),
                nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
                nn.BatchNorm2d(depth),
            )

        def forward(self, x: Any) -> Any:
            return self.res_layer(x) + self.shortcut_layer(x)

    class Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_layer = nn.Sequential(
                nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.PReLU(64)
            )
            units = []
            for in_channel, depth, count in IR101_UNITS:
                units.append(BasicBlockIR(in_channel, depth, 2))
                units += [BasicBlockIR(depth, depth, 1) for _ in range(count - 1)]
            self.body = nn.Sequential(*units)
            self.output_layer = nn.Sequential(
                nn.BatchNorm2d(512), nn.Dropout(0.4), Flatten(),
                nn.Linear(512 * 7 * 7, 512), nn.BatchNorm1d(512, affine=False),
            )

        def forward(self, x: Any) -> Any:
            return self.output_layer(self.body(self.input_layer(x)))

    return Backbone()


def strip_to_network(weights: dict[str, Any]) -> dict[str, Any]:
    """Return the backbone's own keys from a checkpoint that wraps it.

    The HF wrapper stores the network as ``model.net.<key>``; the backbone
    itself knows it as ``<key>``. Anything outside the network (a training
    head, for instance) is dropped.
    """
    marker = "net."
    stripped: dict[str, Any] = {}
    for key, value in weights.items():
        if key.startswith(marker):
            stripped[key[len(marker):]] = value
        elif f".{marker}" in key:
            stripped[key.split(f".{marker}", 1)[1]] = value
    return stripped


def check_keys(result: Any) -> None:
    """Fail unless every weight of the network was loaded and nothing was left over.

    Only BatchNorm's ``num_batches_tracked`` counters may be absent: they are
    bookkeeping for training, some exports drop them, and inference never
    reads them.

    Raises:
        SystemExit: If a real parameter is missing or the checkpoint holds
            keys the network does not have.

    """
    missing = [key for key in result.missing_keys if not key.endswith("num_batches_tracked")]
    if missing or result.unexpected_keys:
        raise SystemExit(
            f"checkpoint does not match IR-101: missing {missing[:5]} ({len(missing)}), "
            f"unexpected {list(result.unexpected_keys)[:5]} ({len(result.unexpected_keys)})"
        )


def export_adaface(output: Path) -> int:
    """Download AdaFace IR-101, load it strictly, export it and check the export."""
    try:
        import numpy as np
        import onnxruntime
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
    except ImportError:
        raise SystemExit(
            "exporting AdaFace needs torch, onnxruntime, safetensors and huggingface_hub"
        ) from None

    print(f"downloading {ADAFACE_REPOSITORY}", flush=True)
    network = strip_to_network(load_file(hf_hub_download(ADAFACE_REPOSITORY, "model.safetensors")))
    model = build_ir101().eval()
    check_keys(model.load_state_dict(network, strict=False))
    print(f"loaded {len(network)} tensors into IR-101 "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters)")

    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "adaface_ir101.onnx"
    probe = torch.rand(2, 3, 112, 112) * 2 - 1
    torch.onnx.export(
        model,
        probe[:1],
        str(onnx_path),
        input_names=["pixels"],
        output_names=["features"],
        dynamic_axes={"pixels": {0: "batch"}, "features": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = model(probe).numpy()
    got = session.run(None, {"pixels": probe.numpy()})[0]
    difference = float(np.abs(expected - got).max())
    print(f"onnx vs torch max difference {difference:.2e}")
    if difference > 1e-3:
        raise SystemExit("the ONNX export does not reproduce the network")

    metadata = {
        "repo": ADAFACE_REPOSITORY,
        "alias": "adaface",
        "architecture": "IR-101",
        "paper": "AdaFace: Quality Adaptive Margin for Face Recognition, CVPR 2022, "
                 "arXiv:2204.00964",
        "training_data": "WebFace12M (licensed for non-commercial research)",
        "input": "1 x 3 x 112 x 112 RGB in [-1, 1], aligned to the ArcFace 5-point template",
        "output": "512-d feature before L2 normalisation",
        "backend": "onnx_arcface",
        "onnx_vs_torch_max_difference": difference,
    }
    (output / "adaface_ir101.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.0f} MB) and adaface_ir101.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Export the requested recogniser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("adaface",), default="adaface")
    parser.add_argument("--output", type=Path, default=Path("models"))
    args = parser.parse_args(argv)
    return export_adaface(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
