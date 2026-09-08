r"""Measure detection and attribution of the learned watermark under one attack.

Bit accuracy answers "how many bits survived"; it does not answer either question
the product actually asks. Those are:

detection
    Does this deepfake carry a DeepShield mark at all? Without this, an unmarked
    picture still gets attributed to *someone*, and that someone is innocent.
attribution
    Which registered user's photograph did it come from? A closed-set question
    over a codebook of registered codes.

Both are scored from the decoder's logits by soft correlation with every
codebook entry: the detection score is the best correlation, attribution is its
argmax. Hard bit decisions throw the reliability away and were measured to cost
0.15 of top-1 at 72 registrants.

The control is the same photograph, unmarked, through the same attack with the
same seed. Every threshold is fitted on a calibration half and every rate is
reported on the other half, because a threshold measured on the data it was fitted
to is optimistic.

One attack per run, selected with ``--attack``:

swap, swap_target
    The graphics swapper from ``build_manipulation_set.py`` (106-point Delaunay
    warp, colour matching, Poisson blending), with the marked photograph as the
    face source or as the frame the other face lands on.
inswapper, inswapper_target
    The GAN swapper ``inswapper_128``. Its source input is a 512-d ArcFace
    embedding, not pixels, so in the source direction no carrier pixel reaches
    the output at all; measuring it settles whether any residual leaks through
    the identity vector. ``--inswapper`` points at the ONNX file.
diffusion
    Stable Diffusion v1.5 img2img on the marked photograph, at each
    ``--strength``. Nothing here was in the training graph; the training used
    only the VAE round trip.
none
    No attack, as a ceiling.

Usage:
    python scripts/evaluate_learned_watermark_attribution.py --attack swap \\
        --checkpoint models/learned_watermark_v2.pt
    python scripts/evaluate_learned_watermark_attribution.py --attack diffusion \\
        --checkpoint models/learned_watermark_v3.pt --strength 0.1 0.2 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_manipulation_set as swaps
from evaluate_learned_watermark import SIZE, align_matrix, as_tensor
from learned_watermark import BITS, Decoder, Encoder, pick_device

from deepshield.media import load_image

SUBSETS = (2, 8, 32, 72)
SUBSET_DRAWS = 20
DIFFUSION_REPOSITORY = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DIFFUSION_SIZE = 256
DIFFUSION_STEPS = 30
DIFFUSION_PROMPT = "a photograph of a person"
DIFFUSION_GUIDANCE = 5.0


def identity_pairs(faces: Path) -> list[tuple[Path, Path]]:
    """Pair each identity with the next so no pair shares a person.

    Accepts the flat ``name_index.jpg`` layout of the evaluation set and the
    nested ``name/name_index.jpg`` layout of LFW itself.
    """
    by_identity: dict[str, list[Path]] = {}
    for path in sorted(faces.rglob("*.jpg")):
        identity = path.parent.name if path.parent != faces else path.stem.rsplit("_", 1)[0]
        by_identity.setdefault(identity, []).append(path)
    names = sorted(by_identity)
    return [
        (by_identity[names[index]][0], by_identity[names[index + 1]][0])
        for index in range(0, len(names) - 1, 2)
    ]


def rank_auc(positives: np.ndarray, negatives: np.ndarray) -> float:
    """Return the ROC-AUC as the probability a positive outscores a negative."""
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    wins = (positives[:, None] > negatives[None, :]).sum()
    ties = (positives[:, None] == negatives[None, :]).sum()
    return float((wins + 0.5 * ties) / (positives.size * negatives.size))


class Model:
    """The encoder-decoder pair plus the coordinate round trip into a photograph."""

    def __init__(self, checkpoint: Path, device: str, orientation: str | None) -> None:
        """Load the checkpoint and decide the decoder's orientation."""
        state = torch.load(checkpoint, map_location=device)
        self.device = device
        self.encoder, self.decoder = Encoder().to(device), Decoder().to(device)
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
        self.encoder.eval()
        self.decoder.eval()
        self.scale = float(state["res_scale"])
        self.orientation = orientation or state.get("orientation", "upright")
        self.transposed = self.orientation == "transposed"

    def mark(
        self, cv2: Any, app: Any, image: np.ndarray, message: np.ndarray
    ) -> tuple[np.ndarray, float] | None:
        """Return the photograph carrying ``message`` in its aligned face, and PSNR."""
        matrix = align_matrix(cv2, app, image)
        if matrix is None:
            return None
        crop = cv2.warpAffine(image, matrix, (SIZE, SIZE))
        bound = torch.from_numpy(message[None].astype(np.float32)).to(self.device)
        with torch.no_grad():
            residual = self.encoder(as_tensor(crop, self.device, False), bound) * self.scale
        shift = residual.squeeze(0).permute(1, 2, 0).cpu().numpy() * 127.5
        spread = cv2.warpAffine(
            shift, cv2.invertAffineTransform(matrix), (image.shape[1], image.shape[0])
        )
        marked = np.clip(image.astype(np.float32) + spread, 0, 255).astype(np.uint8)
        error = ((marked.astype(float) - image.astype(float)) ** 2).mean()
        return marked, float(10 * np.log10(255.0**2 / max(error, 1e-9)))

    def logits(self, cv2: Any, app: Any, image: np.ndarray) -> np.ndarray | None:
        """Re-detect, re-align and return the decoder's per-bit logits."""
        matrix = align_matrix(cv2, app, image)
        if matrix is None:
            return None
        aligned = cv2.warpAffine(image, matrix, (SIZE, SIZE))
        with torch.no_grad():
            out = self.decoder(as_tensor(aligned, self.device, self.transposed))
        return out.squeeze(0).cpu().numpy().astype(np.float64)


class Attacks:
    """Every attack the benchmark can apply, each built only when first used."""

    def __init__(self, models: Path, inswapper: Path, device: str) -> None:
        """Remember where the heavy models live without loading them yet."""
        self.models = models
        self.inswapper_path = inswapper
        self.device = device
        self._recogniser: Any = None
        self._swapper: Any = None
        self._pipeline: Any = None

    def recogniser(self) -> Any:
        """Return an InsightFace app that also produces identity embeddings."""
        if self._recogniser is None:
            import insightface

            app = insightface.app.FaceAnalysis(
                name="buffalo_l",
                root=str(self.models / "insightface"),
                allowed_modules=["detection", "recognition"],
            )
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self._recogniser = app
        return self._recogniser

    def swapper(self) -> Any:
        """Return the inswapper model."""
        if self._swapper is None:
            if not self.inswapper_path.is_file():
                raise SystemExit(f"missing {self.inswapper_path}; see the README for the source")
            from insightface.model_zoo import get_model

            self._swapper = get_model(
                str(self.inswapper_path), providers=["CPUExecutionProvider"]
            )
        return self._swapper

    def pipeline(self) -> Any:
        """Return the Stable Diffusion img2img pipeline."""
        if self._pipeline is None:
            try:
                from diffusers import StableDiffusionImg2ImgPipeline
            except ImportError as exc:
                raise SystemExit("--attack diffusion needs diffusers and transformers") from exc
            dtype = torch.float16 if self.device != "cpu" else torch.float32
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
                DIFFUSION_REPOSITORY,
                torch_dtype=dtype,
                variant="fp16",
                safety_checker=None,
                requires_safety_checker=False,
            ).to(self.device)
            pipeline.set_progress_bar_config(disable=True)
            self._pipeline = pipeline
        return self._pipeline

    @staticmethod
    def _largest(faces: list[Any]) -> Any:
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def inswap(self, source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
        """Put the identity of ``source`` onto the face in ``target``."""
        app = self.recogniser()
        source_faces = app.get(np.ascontiguousarray(source[:, :, ::-1]))
        target_faces = app.get(np.ascontiguousarray(target[:, :, ::-1]))
        if not source_faces or not target_faces:
            return None
        swapped = self.swapper().get(
            np.ascontiguousarray(target[:, :, ::-1]),
            self._largest(target_faces),
            self._largest(source_faces),
            paste_back=True,
        )
        return np.ascontiguousarray(swapped[:, :, ::-1])

    def diffuse(self, image: np.ndarray, strength: float, seed: int) -> np.ndarray:
        """Regenerate the photograph with img2img at one denoising strength."""
        from PIL import Image

        height, width = image.shape[:2]
        pil = Image.fromarray(image).resize(
            (DIFFUSION_SIZE, DIFFUSION_SIZE), Image.Resampling.LANCZOS
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        out = self.pipeline()(
            prompt=DIFFUSION_PROMPT,
            image=pil,
            strength=strength,
            num_inference_steps=DIFFUSION_STEPS,
            guidance_scale=DIFFUSION_GUIDANCE,
            generator=generator,
        ).images[0]
        return np.asarray(out.resize((width, height), Image.Resampling.LANCZOS), dtype=np.uint8)

    def apply(
        self,
        cv2: Any,
        app: Any,
        name: str,
        probe: np.ndarray,
        other: np.ndarray,
        strength: float,
        seed: int,
    ) -> np.ndarray | None:
        """Apply the named attack to ``probe``."""
        if name == "none":
            return probe
        if name == "swap":
            return swaps.swap_face(cv2, probe, other, app)
        if name == "swap_target":
            return swaps.swap_face(cv2, other, probe, app)
        if name == "inswapper":
            return self.inswap(probe, other)
        if name == "inswapper_target":
            return self.inswap(other, probe)
        if name == "diffusion":
            return self.diffuse(probe, strength, seed)
        raise ValueError(f"unknown attack {name}")


def score(logits: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Return the soft correlation of the logits with every codebook entry."""
    signs = 2.0 * codebook.astype(np.float64) - 1.0
    correlation: np.ndarray = signs @ logits
    return correlation


def closed_set_top1(
    scores: np.ndarray, truth: int, size: int, rng: np.random.Generator
) -> float:
    """Return how often the true code wins among a random subset of ``size`` codes."""
    if size >= scores.size:
        return float(int(np.argmax(scores)) == truth)
    others = np.delete(np.arange(scores.size), truth)
    wins = 0
    for _ in range(SUBSET_DRAWS):
        drawn = rng.choice(others, size - 1, replace=False)
        wins += int(scores[truth] > scores[drawn].max())
    return wins / SUBSET_DRAWS


def summarise(
    marked: list[dict[str, Any]], unmarked: list[dict[str, Any]], seed: int
) -> dict[str, Any]:
    """Fit thresholds on one half, report every rate on the other."""
    rng = np.random.default_rng(seed)
    count = min(len(marked), len(unmarked))
    order = rng.permutation(count)
    calibration, test = order[: count // 2], order[count // 2 :]

    marked_scores = np.array([row["detection_score"] for row in marked])[:count]
    unmarked_scores = np.array([row["detection_score"] for row in unmarked])[:count]
    top1 = np.array([row["top1"] for row in marked])[:count]
    subsets = {
        size: np.array([row["closed_set"][str(size)] for row in marked])[:count]
        for size in SUBSETS
    }

    threshold_zero = float(unmarked_scores[calibration].max()) + 1e-9
    threshold_one = float(np.quantile(unmarked_scores[calibration], 0.99))
    return {
        "samples": {"marked": len(marked), "unmarked": len(unmarked), "paired": count},
        "split": {"calibration": int(calibration.size), "test": int(test.size)},
        "detection": {
            "threshold_zero_fp_on_calibration": round(threshold_zero, 4),
            "test_detection_rate": round(
                float((marked_scores[test] >= threshold_zero).mean()), 4
            ),
            "test_false_positive_rate": round(
                float((unmarked_scores[test] >= threshold_zero).mean()), 4
            ),
            "threshold_1pct_fp_on_calibration": round(threshold_one, 4),
            "test_detection_rate_at_1pct": round(
                float((marked_scores[test] >= threshold_one).mean()), 4
            ),
            "test_false_positive_rate_at_1pct": round(
                float((unmarked_scores[test] >= threshold_one).mean()), 4
            ),
            "test_auc": round(rank_auc(marked_scores[test], unmarked_scores[test]), 4),
            "all_auc": round(rank_auc(marked_scores, unmarked_scores), 4),
            "marked_score_mean": round(float(marked_scores.mean()), 4),
            "unmarked_score_mean": round(float(unmarked_scores.mean()), 4),
        },
        "attribution_test_half": {
            f"top1_of_{size}": round(float(subsets[size][test].mean()), 4) for size in SUBSETS
        },
        "attribution_all": {"top1_of_codebook": round(float(top1.mean()), 4)},
    }


def main(argv: list[str] | None = None) -> int:
    """Run one attack over the identity pairs and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--faces", type=Path, default=Path("data/sklearn/lfw_home/lfw_funneled"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument(
        "--inswapper", type=Path, default=Path("models/inswapper/inswapper_128.onnx")
    )
    parser.add_argument("--output", type=Path, default=Path("data/results"))
    parser.add_argument("--pairs", type=int, default=150)
    parser.add_argument("--codebook", type=int, default=72)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--attack",
        choices=("none", "swap", "swap_target", "inswapper", "inswapper_target", "diffusion"),
        default="swap",
    )
    parser.add_argument("--strength", type=float, nargs="+", default=[0.1, 0.2, 0.3])
    parser.add_argument("--orientation", choices=("transposed", "upright"), default=None)
    parser.add_argument("--tag", default=None, help="name for the output file")
    args = parser.parse_args(argv)

    if not args.checkpoint.is_file():
        raise SystemExit(f"missing {args.checkpoint}")
    pairs = identity_pairs(args.faces)
    if not pairs:
        raise SystemExit(f"no identity pairs found in {args.faces}")

    import cv2

    device = pick_device()
    model = Model(args.checkpoint, device, args.orientation)
    app = swaps.build_analyzer(args.models)
    attacks = Attacks(args.models, args.inswapper, device)

    rng = np.random.default_rng(args.seed)
    codebook = rng.integers(0, 2, (args.codebook, BITS)).astype(np.uint8)
    strengths = args.strength if args.attack == "diffusion" else [0.0]
    cells: dict[float, dict[str, list[dict[str, Any]]]] = {
        strength: {"marked": [], "unmarked": []} for strength in strengths
    }
    quality: list[float] = []
    samples = 0

    for index, (own_path, other_path) in enumerate(pairs):
        if samples >= args.pairs:
            break
        own, other = load_image(own_path), load_image(other_path)
        truth = index % args.codebook
        prepared = model.mark(cv2, app, own, codebook[truth])
        if prepared is None or align_matrix(cv2, app, other) is None:
            continue
        marked, psnr = prepared
        quality.append(psnr)
        samples += 1
        subset_rng = np.random.default_rng(args.seed * 1000 + index)

        for strength in strengths:
            for label, probe in (("marked", marked), ("unmarked", own)):
                attacked = attacks.apply(
                    cv2, app, args.attack, probe, other, strength, args.seed * 1000 + index
                )
                if attacked is None:
                    continue
                logits = model.logits(cv2, app, attacked)
                if logits is None:
                    continue
                scores = score(logits, codebook)
                cells[strength][label].append(
                    {
                        "pair": index,
                        "truth": truth,
                        "detection_score": float(scores.max()),
                        "top1": float(int(np.argmax(scores)) == truth),
                        "closed_set": {
                            str(size): closed_set_top1(scores, truth, size, subset_rng)
                            for size in SUBSETS
                        },
                    }
                )
        if samples % 25 == 0:
            print(f"{samples} samples", flush=True)

    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "orientation": model.orientation,
        "attack": args.attack,
        "faces": str(args.faces),
        "pairs_requested": args.pairs,
        "pairs_marked": samples,
        "codebook": args.codebook,
        "seed": args.seed,
        "psnr_db": round(float(np.mean(quality)), 2) if quality else None,
        "scoring": "soft correlation of decoder logits with every codebook entry",
        "results": {},
    }
    if args.attack == "diffusion":
        report["diffusion"] = {
            "repository": DIFFUSION_REPOSITORY,
            "size": DIFFUSION_SIZE,
            "steps": DIFFUSION_STEPS,
            "guidance": DIFFUSION_GUIDANCE,
            "prompt": DIFFUSION_PROMPT,
        }
    for strength, cell in cells.items():
        key = f"strength_{strength:.2f}" if args.attack == "diffusion" else args.attack
        if not cell["marked"] or not cell["unmarked"]:
            report["results"][key] = {
                "samples": {"marked": len(cell["marked"]), "unmarked": len(cell["unmarked"])}
            }
            continue
        report["results"][key] = summarise(cell["marked"], cell["unmarked"], args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.checkpoint.stem.replace('learned_watermark_', '')}_{args.attack}"
    destination = args.output / f"learned_watermark_attribution_{tag}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
