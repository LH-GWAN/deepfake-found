r"""Build the upload bundle for the LoRA-defence experiment run on Colab.

The heavy half of the experiment (perturbations against Stable Diffusion,
LoRA fine-tuning, generation) runs on a CUDA GPU; this script does the light
half that needs the project's own models, so the notebook needs none of them.

For each chosen identity it writes the eight evaluation photographs, and the
same photographs carrying the swap-defence perturbation
(``evaluate_source_protection.protect`` at 8/255 against ArcFace). Training a
LoRA on the latter answers whether the noise that stops a face swap also
does anything to fine-tuning, which it was not designed for.

Layout inside the zip::

    clean/<identity>/<n>.png
    arcface8/<identity>/<n>.png

Usage:
    python scripts/prepare_lora_defense.py --identities tom_hanks jennifer_lopez hugo_chavez
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_source_protection import ARCFACE, OnnxGraph, protect
from evaluate_verdicts import GanSwapper
from learned_watermark import pick_device

from deepshield.media import load_image
from deepshield.quality import psnr, ssim

DEFAULT_IDENTITIES = ["tom_hanks", "jennifer_lopez", "hugo_chavez"]


def main(argv: list[str] | None = None) -> int:
    """Write clean and swap-protected photographs of each identity into one zip."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faces", type=Path, default=Path("data/test/eval_faces"))
    parser.add_argument("--identities", nargs="+", default=DEFAULT_IDENTITIES)
    parser.add_argument("--models", type=Path, default=Path("models/insightface"))
    parser.add_argument(
        "--inswapper", type=Path, default=Path("models/inswapper/inswapper_128.onnx")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    device = pick_device()
    arcface = OnnxGraph(ARCFACE).to(device)
    detector = GanSwapper(args.models, args.inswapper)
    qualities: list[tuple[float, float]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as bundle:
        for identity in args.identities:
            paths = sorted(args.faces.glob(f"{identity}_*.png"))
            if not paths:
                raise SystemExit(f"no photographs of {identity} under {args.faces}")
            for index, path in enumerate(paths):
                image = load_image(path)
                face = detector._largest(image)
                protected = (
                    image if face is None
                    else protect(arcface, image, face.kps, 8 / 255.0, device)
                )
                if face is not None:
                    qualities.append((psnr(image, protected), ssim(image, protected)))
                for variant, pixels in (("clean", image), ("arcface8", protected)):
                    target = Path(args.output.parent / f"_{variant}.png")
                    Image.fromarray(np.asarray(pixels)).save(target)
                    bundle.write(target, f"{variant}/{identity}/{index}.png")
                    target.unlink()
            print(f"{identity}: {len(paths)} photographs", flush=True)
    if qualities:
        print(
            f"arcface8 quality: PSNR {np.mean([q[0] for q in qualities]):.2f} dB, "
            f"SSIM {np.mean([q[1] for q in qualities]):.4f}"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
