r"""Check the joint term locally before it goes to Kaggle, and choose where it may stop.

1. The copy in ``joint_arcface.py`` against the shipped ``protection/shield.py``: the same
   frame matrix, sampling grid, blur and embedding for the same photograph.
2. The term alone, driven the way Mist drives it: Mist's step (0.005 of [-1, 1]) and
   budget (16/255 of [-1, 1], 8/255 of [0, 1]), five rounds of 30 steps from the
   watermarked 512-pixel photograph (``wm512``), saved with Mist's quantisation. This is
   run for several stopping similarities ``tau`` on the pilot people's photographs, and
   each result is face-swapped with inswapper onto another person's photograph, as saved
   and after JPEG 85, JPEG 70 and halving, then scored against the whole gallery like
   evaluate_mist_shield.py. The watermark alone (``wm512``) and the shipped shield
   (``shield512``) are the two ends.

Inside the joint run Mist pulls the similarity back up, so it hovers around ``tau``
instead of settling below it; the stopping point chosen here needs a margin. Results go
to ``<work>/protected/check_<tau>/`` and ``<output>/joint_arcface_check.json``. No
model is trained; the shield and inswapper run forward as in the other local scripts.

    python experiments/lora_defense/check_joint_arcface.py --work experiments/lora_defense/v3 \
        --clean experiments/lora_defense/v3/clean --gallery experiments/lora_defense/v3/gallery \
        --output data/results/lora_defense_v3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joint_arcface as joint
from evaluate_source_protection import RESAVES, Recognisers, score
from evaluate_verdicts import GanSwapper, photographs
from prepare_joint_inputs import PILOT, enlarge

from deepshield.config import load_config
from deepshield.media import load_image, save_image
from deepshield.protection import shield as shipped
from deepshield.quality import psnr, ssim

MIST_STEP = 0.005
MIST_EPS = 16 / 255
ROUNDS, STEPS = 5, 30


def require_local_models() -> None:
    """Refuse to start unless ./models holds what the recognisers load.

    They look for their models under ./models and download whatever is missing there
    (YuNet, SFace, the whole buffalo_l pack), so run from the checkout that has them.
    """
    missing = [path.resolve() for path in (
        Path("models/face_detection_yunet_2023mar.onnx"),
        Path("models/face_recognition_sface_2021dec.onnx"),
        Path("models/insightface/models/buffalo_l/w600k_r50.onnx"),
    ) if not path.is_file()]
    if missing:
        raise SystemExit(f"run from the checkout that holds the models; missing {missing}")


def to_mist(image: np.ndarray) -> Any:
    """Return an RGB array as Mist holds it: 1x3xHxW in [-1, 1]."""
    import torch

    return torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].float().div(127.5).sub(1)


def from_mist(x: Any) -> np.ndarray:
    """Return Mist's image the way Mist saves it (``* 127.5 + 128``, clamped, truncated)."""
    import torch

    return (x[0] * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def compare_with_shipped(encoder_path: Path, photo: np.ndarray, landmarks: list, device: str,
                         ours: Any) -> dict[str, float]:
    """Return the largest differences between the copy and the shipped module."""
    import torch

    theirs = shipped.OnnxGraph(encoder_path, device)
    matrix_ours = joint.similarity_matrix(np.asarray(landmarks))
    matrix_theirs = shipped.similarity_matrix(np.asarray(landmarks))
    jitter = torch.tensor([[1.02, 0.03, 1.5, -1.0], [0.97, -0.02, -0.5, 2.0]], device=device)
    height, width = photo.shape[:2]
    grid_ours = joint.sampling_grid(matrix_ours, height, width, jitter)
    grid_theirs = shipped.sampling_grid(matrix_theirs, height, width, jitter)
    tensor = torch.from_numpy(np.array(photo)).permute(2, 0, 1)[None].float().div(255).to(device)
    frames = torch.nn.functional.grid_sample(
        tensor.repeat(2, 1, 1, 1), grid_theirs, align_corners=True)
    with torch.no_grad():
        embedded_ours = ours(joint.blur(frames, 0.6) * 2 - 1)
        embedded_theirs = theirs(shipped._blur(frames, 0.6) * 2 - 1)
    return {
        "matrix": float(np.abs(matrix_ours - matrix_theirs).max()),
        "grid": float((grid_ours - grid_theirs).abs().max()),
        "embedding": float((embedded_ours - embedded_theirs).abs().max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--clean", type=Path, required=True, help="<identity>/<n>.png")
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--inswapper", type=Path,
                        default=ROOT / "models/inswapper/inswapper_128.onnx")
    parser.add_argument("--people", default=",".join(PILOT))
    parser.add_argument("--photos", type=int, default=2, help="per person, from 0.png")
    parser.add_argument("--taus", default="-0.2,-0.4,-0.6,none",
                        help="stopping similarities; none never stops")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    require_local_models()
    config = load_config()
    model_dir = Path(config.runtime.model_dir)
    encoder_path = model_dir / config.protection.shield.encoder_model
    device = shipped.pick_device()
    ours = joint.OnnxGraph(str(encoder_path), device)
    landmarks = json.loads((args.work / "joint_inputs" / "landmarks.json").read_text("utf-8"))
    people = args.people.split(",")
    names = [f"{n}.png" for n in range(args.photos)]
    first = load_image(args.work / "protected" / "wm512" / people[0] / names[0])
    report: dict[str, Any] = {
        "question": "does the joint term reproduce the shield, and how low must it push "
                    "the similarity for inswapper to stop carrying the person?",
        "copy_vs_shipped_max_difference": compare_with_shipped(
            encoder_path, first, landmarks[people[0]][names[0]][0], device, ours),
        "mist_step": MIST_STEP, "mist_eps_of_minus1_1": MIST_EPS,
        "rounds_x_steps": [ROUNDS, STEPS],
    }
    print(json.dumps(report["copy_vs_shipped_max_difference"]), flush=True)

    taus = {tag: (float(tag) if tag != "none" else float("-inf"))
            for tag in args.taus.split(",")}
    quality: dict[str, list[tuple[float, float]]] = defaultdict(list)
    similarity: dict[str, list[float]] = defaultdict(list)
    seconds: dict[str, list[float]] = defaultdict(list)
    for person in people:
        for name in names:
            marked = load_image(args.work / "protected" / "wm512" / person / name)
            reference = np.asarray(enlarge(args.clean / person / name))
            start = to_mist(marked)
            for tag, tau in taus.items():
                target = args.work / "protected" / f"check_{tag}" / person / name
                term = joint.ArcFaceTerm(ours, [landmarks[person][name]], start, device,
                                         tau, MIST_STEP)
                if not target.exists():
                    began = time.perf_counter()
                    x = start.clone()
                    for _ in range(ROUNDS * STEPS):
                        x = x + term.step(x, 0)
                        x = (start + (x - start).clamp(-MIST_EPS, MIST_EPS)).clamp(-1, 1)
                    save_image(from_mist(x), target)
                    seconds[tag].append(time.perf_counter() - began)
                saved = load_image(target)
                similarity[tag].append(term.similarities(to_mist(saved), 0)[0])
                quality[tag].append((psnr(reference, saved), ssim(reference, saved)))
            for variant in ("wm512", "shield512"):
                saved = load_image(args.work / "protected" / variant / person / name)
                quality[variant].append((psnr(reference, saved), ssim(reference, saved)))
            print(f"{person}/{name}: " + ", ".join(
                f"{tag} {similarity[tag][-1]:+.3f}" for tag in taus), flush=True)

    everyone = sorted(p.name for p in args.clean.iterdir() if len(list(p.glob("*.png"))) >= 8)
    grouped = photographs(args.gallery)
    recognise = Recognisers()
    galleries = {
        identity: {p.name: v for p in paths if (v := recognise(load_image(p))) is not None}
        for identity, paths in grouped.items()
    }
    swapper = GanSwapper(model_dir / "insightface", args.inswapper)
    swaps: dict[str, list[dict[str, Any] | None]] = defaultdict(list)
    variants = ["wm512", "shield512"] + [f"check_{tag}" for tag in taus]
    for person in people:
        other = everyone[(everyone.index(person) + 1) % len(everyone)]
        picture = load_image(grouped[other][0])
        for name in names:
            for variant in variants:
                photo = load_image(args.work / "protected" / variant / person / name)
                conditions = {variant: photo}
                conditions.update({f"{variant}_{k}": f(photo) for k, f in RESAVES.items()})
                for condition, image in conditions.items():
                    swapped = swapper(image, picture)
                    swaps[condition].append(score(
                        None if swapped is None else recognise(swapped),
                        galleries, person, f"{person}_{name}"))
        print(f"swapped {person}", flush=True)

    def summarise(entries: list[dict[str, Any] | None]) -> dict[str, Any]:
        scored = [e for e in entries if e is not None]
        sims = [e["arcface"]["donor_similarity"] for e in scored]
        return {
            "swaps": len(entries), "with_a_face": len(scored),
            "arcface_owner_first": int(sum(e["arcface"]["donor_first"] for e in scored)),
            "sface_owner_first": int(sum(e["sface"]["donor_first"] for e in scored)),
            "median_owner_similarity": round(float(np.median(sims)), 4) if sims else None,
            "above_high_confidence": int(sum(v >= recognise.high for v in sims)),
        }

    report.update({
        "people": people, "photos_per_person": args.photos,
        "gallery_identities": len(galleries),
        "similarity_to_clean_face_after_saving": {
            tag: {"median": round(float(np.median(v)), 4),
                  "max": round(float(np.max(v)), 4)} for tag, v in similarity.items()},
        "seconds_per_photo": {tag: round(float(np.median(v)), 1) for tag, v in seconds.items()},
        "quality_vs_enlarged_clean": {
            k: {"psnr": round(float(np.mean([q[0] for q in v])), 2),
                "ssim": round(float(np.mean([q[1] for q in v])), 4)} for k, v in quality.items()},
        "swaps": {c: summarise(e) for c, e in swaps.items()},
    })
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "joint_arcface_check.json").write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps({k: report[k] for k in (
        "similarity_to_clean_face_after_saving", "quality_vs_enlarged_clean", "swaps")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
