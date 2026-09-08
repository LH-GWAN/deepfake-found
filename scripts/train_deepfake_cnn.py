r"""Train a CNN deepfake detector on this repository's own fakes, family by family.

Four detectors have failed on this material: hand-made blending features and
three public checkpoints, all at chance on graphics swaps and again on GAN
swaps. The README's remaining move was the one this repository can make
without a data-use agreement: build fakes from three different families
(graphics blending, GAN swap, diffusion regeneration), train on some, and
measure on the others. The number that matters is not in-family accuracy,
which any classifier reaches by memorising one generator's artefacts, but
cross-family transfer on identities it never saw.

Protocol:

identity-disjoint split
    A fixed fraction of identities is held out before anything is trained, and
    the same held-out set is used for every family, so cross-family numbers
    are never inflated by a face the network has seen.
one model per training recipe
    ``--train`` names the families to train on; every family is evaluated on
    its held-out identities, clean and after JPEG q50, and the real photographs
    are shared, so the false-positive rate is one number per model.
the same crop the pipeline makes
    Faces are located by the project's detector and cropped with the same
    margin the analysis pipeline uses, then the network sees them at 224
    pixels. Resize and ImageNet normalisation are exported inside the ONNX
    graph, so the ``onnx`` backend feeds it [0, 1] RGB exactly as it feeds the
    published checkpoints, and ``evaluate_deepfake_detectors.py`` can survey
    it and refuse or adopt it by the same bar.

Usage:
    python scripts/train_deepfake_cnn.py --manifest data/test/manipulated/manifest.json \\
        data/test/manipulated_inswapper/manifest.json \\
        data/test/manipulated_diffusion/manifest.json \\
        --train inswapper diffusion --tag cnn_gan_diffusion
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learned_watermark import pick_device

from deepshield.config import load_config
from deepshield.face.detector import build_detector
from deepshield.media import load_image
from deepshield.pipeline.analysis_pipeline import DEEPFAKE_CROP_MARGIN, crop_with_margin
from deepshield.risk.calibration import precision_recall_at, roc_curve
from deepshield.transforms import Transformation

SIDE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MIN_USEFUL_AUC = 0.75
MAX_TOLERABLE_FPR = 0.10


def family_of(manifest: Path, payload: dict[str, Any]) -> str:
    """Name a manifest's family from its swapper field or its directory."""
    swapper = payload.get("swapper")
    if swapper:
        return str(swapper)
    name = manifest.parent.name
    return name.replace("manipulated_", "") if "_" in name else "graphics"


def load_manifests(
    paths: list[Path],
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]]]:
    """Return fake records per family and the shared real records, deduplicated."""
    fakes: dict[str, list[dict[str, str]]] = {}
    reals: dict[str, dict[str, str]] = {}
    for manifest in paths:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        family = family_of(manifest, payload)
        for record in payload["records"]:
            if record["label"] == "fake":
                fakes.setdefault(family, []).append({**record, "family": family})
            else:
                reals.setdefault(record["source"], {**record, "family": "real"})
    return fakes, list(reals.values())


def face_crop(detector: Any, image: np.ndarray) -> np.ndarray | None:
    """Return the pipeline's loose face crop, resized for the network."""
    faces = detector.detect(image)
    if not faces:
        return None
    crop = crop_with_margin(image, faces[0], DEEPFAKE_CROP_MARGIN)
    return np.asarray(Image.fromarray(crop).resize((SIDE, SIDE), Image.Resampling.BILINEAR))


def jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    """Re-encode an image as JPEG at the given quality."""
    return Transformation("jpeg", "jpeg_compression", {"quality": quality}).apply(image, seed=1)


class Detector(nn.Module):
    """EfficientNet-B0 with a two-way head and the preprocessing inside the graph."""

    def __init__(self) -> None:
        """Build the backbone from ImageNet weights."""
        super().__init__()
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(features, 2)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        """Take [0, 1] RGB at any size and return two logits (real, fake)."""
        resized = F.interpolate(pixels, size=(SIDE, SIDE), mode="bilinear", align_corners=False)
        logits: torch.Tensor = self.backbone((resized - self.mean) / self.std)
        return logits


def augment(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Flip, jitter and re-encode so the network cannot key on one encoder."""
    out = image
    if rng.random() < 0.5:
        out = out[:, ::-1]
    if rng.random() < 0.5:
        out = jpeg(np.ascontiguousarray(out), rng.randint(50, 95))
    if rng.random() < 0.5:
        gain = rng.uniform(0.85, 1.15)
        out = np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def to_tensor(images: list[np.ndarray], device: str) -> torch.Tensor:
    """Stack uint8 crops into a [0, 1] batch."""
    array = np.stack(images).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(0, 3, 1, 2).to(device)


def scores_for(model: Detector, crops: list[np.ndarray], device: str) -> np.ndarray:
    """Return the fake probability of every crop."""
    model.eval()
    out: list[float] = []
    with torch.no_grad():
        for start in range(0, len(crops), 32):
            batch = to_tensor(crops[start : start + 32], device)
            out.extend(torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy().tolist())
    return np.asarray(out)


def evaluate(scores: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Return AUC, EER, and the operating point at the network's own 0.5.

    A family whose held-out half is empty on either side gets ``None`` metrics
    rather than an exception, so a small run still reports what it can.
    """
    if labels.size == 0 or labels.min() == labels.max():
        return {"n": int(labels.size), "auc": None, "eer": None,
                "recall_at_0.5": None, "false_positive_rate_at_0.5": None}
    curve = roc_curve(scores, labels)
    point = precision_recall_at(scores, labels, 0.5)
    return {
        "n": int(labels.size),
        "auc": round(float(curve.auc), 4),
        "eer": round(float(curve.eer), 4),
        "recall_at_0.5": round(float(point["recall"]), 4),
        "false_positive_rate_at_0.5": round(float(point["false_positive_rate"]), 4),
    }


def main(argv: list[str] | None = None) -> int:
    """Train on the named families and report every family held out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--train", nargs="+", required=True, help="families to train on")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    parser.add_argument("--holdout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="cap records per family")
    args = parser.parse_args(argv)

    fakes, reals = load_manifests(args.manifest)
    missing = [family for family in args.train if family not in fakes]
    if missing:
        raise SystemExit(f"no fakes for {missing}; families present: {sorted(fakes)}")

    rng = random.Random(args.seed)
    identities = sorted({record["identity"] for record in reals})
    rng.shuffle(identities)
    held = set(identities[: max(1, int(round(len(identities) * args.holdout)))])
    print(f"families: {sorted(fakes)}; identities {len(identities)}, held out {len(held)}")

    config = load_config()
    detector = build_detector(config.face.detector)
    device = pick_device()

    def crops_of(records: list[dict[str, str]]) -> list[tuple[np.ndarray, str, str]]:
        found = []
        for record in records[: args.limit] if args.limit else records:
            path = Path(record["path"])
            if not path.is_file():
                continue
            crop = face_crop(detector, load_image(path))
            if crop is not None:
                found.append((crop, record["identity"], record["family"]))
        return found

    real_crops = crops_of(reals)
    fake_crops = {family: crops_of(records) for family, records in fakes.items()}
    counts = ", ".join(f"{k} {len(v)}" for k, v in fake_crops.items())
    print(f"crops: real {len(real_crops)}, {counts}")

    train_items: list[tuple[np.ndarray, int]] = [
        (crop, 0) for crop, identity, _ in real_crops if identity not in held
    ]
    for family in args.train:
        train_items += [
            (crop, 1) for crop, identity, _ in fake_crops[family] if identity not in held
        ]
    positives = sum(label for _, label in train_items)
    print(f"training on {len(train_items)} crops ({positives} fake) from {args.train}")

    torch.manual_seed(args.seed)
    model = Detector().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    weight = torch.tensor(
        [positives / max(1, len(train_items) - positives), 1.0], device=device
    ).float()
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_items)
        total = 0.0
        for start in range(0, len(train_items), args.batch):
            chunk = train_items[start : start + args.batch]
            batch = to_tensor([augment(crop, rng) for crop, _ in chunk], device)
            labels = torch.tensor([label for _, label in chunk], device=device)
            loss = F.cross_entropy(model(batch), labels, weight=weight)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.item()) * len(chunk)
        mean_loss = total / max(1, len(train_items))
        print(f"epoch {epoch:2d} loss {mean_loss:.4f} {time.time() - started:.0f}s", flush=True)

    held_real = [crop for crop, identity, _ in real_crops if identity in held]
    report: dict[str, Any] = {
        "tag": args.tag,
        "trained_on": args.train,
        "holdout_identities": len(held),
        "train_crops": len(train_items),
        "epochs": args.epochs,
        "held_out_real": len(held_real),
        "per_family": {},
        "criteria": {"min_auc": MIN_USEFUL_AUC, "max_false_positive_rate": MAX_TOLERABLE_FPR},
    }
    real_scores = {"clean": scores_for(model, held_real, device),
                   "jpeg50": scores_for(model, [jpeg(c, 50) for c in held_real], device)}
    for family, crops in fake_crops.items():
        held_fake = [crop for crop, identity, _ in crops if identity in held]
        entry: dict[str, Any] = {
            "seen_in_training": family in args.train,
            "held_out_fake": len(held_fake),
        }
        for condition, transform in (("clean", None), ("jpeg50", 50)):
            fake_scores = scores_for(
                model, [jpeg(c, transform) if transform else c for c in held_fake], device
            )
            scores = np.concatenate([real_scores[condition], fake_scores])
            labels = np.concatenate([np.zeros(len(held_real)), np.ones(len(held_fake))])
            entry[condition] = evaluate(scores, labels)
        clean_auc = entry["clean"]["auc"]
        clean_fpr = entry["clean"]["false_positive_rate_at_0.5"]
        entry["usable_by_this_project's_bar"] = bool(
            clean_auc is not None
            and clean_auc >= MIN_USEFUL_AUC
            and clean_fpr <= MAX_TOLERABLE_FPR
        )
        report["per_family"][family] = entry
        seen = "seen" if family in args.train else "unseen"
        print(f"{family:10s} ({seen}) clean {entry['clean']}", flush=True)
        print(f"{'':10s} {'':8s} jpeg50 {entry['jpeg50']}", flush=True)

    args.models.mkdir(parents=True, exist_ok=True)
    onnx_path = args.models / f"deepfake_{args.tag}.onnx"
    model.eval().cpu()
    torch.onnx.export(
        model,
        torch.rand(1, 3, SIDE, SIDE),
        str(onnx_path),
        input_names=["pixels"],
        output_names=["logits"],
        dynamic_axes={"pixels": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    metadata = {
        "repo": f"deepshield/{args.tag}",
        "alias": args.tag,
        "input_size": SIDE,
        "native_size": SIDE,
        "positive_index": 1,
        "labels": {"0": "Real", "1": "Fake"},
        "preprocessing": "resize and normalise are baked into the graph; feed [0,1] RGB",
        "trained_on": args.train,
        "training_families_manifests": [str(m) for m in args.manifest],
        "holdout_fraction": args.holdout,
        "note": (
            "trained on fakes this repository generated itself; in-family numbers say "
            "nothing about deployment, only the unseen-family rows do"
        ),
    }
    (args.models / f"deepfake_{args.tag}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / f"deepfake_cnn_{args.tag}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {destination} and {onnx_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
