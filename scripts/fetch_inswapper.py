r"""Fetch the inswapper_128 GAN face swapper, pinned by URL and SHA-256.

The graphics swapper this repository ships moves pixels along a landmark mesh.
GAN swappers work differently: ``inswapper_128`` takes the target frame and a
512-d ArcFace embedding of the source identity, and regenerates a 128-pixel
aligned face from the two. The source photograph contributes an embedding and
nothing else. Measuring the watermark against it is what settles whether any
of the mark leaks through the identity vector, and the README records that it
does not.

The weights are not part of ``download-models``: they are 554 MB, licensed for
non-commercial research by their authors, and nothing in the pipeline uses
them. They serve one evaluation script. The digest below is the one the
InsightFace community publishes for this file, and the download is refused if
the bytes do not match it.

Usage:
    python scripts/fetch_inswapper.py
    python scripts/evaluate_learned_watermark_attribution.py --attack inswapper_target \\
        --checkpoint models/learned_watermark_v2.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepshield.models import ModelAsset, download_asset

INSWAPPER = ModelAsset(
    key="inswapper_128",
    filename="inswapper_128.onnx",
    url="https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    sha256="e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
    description="inswapper_128 GAN face swapper, evaluation only",
)


def main(argv: list[str] | None = None) -> int:
    """Download the swapper into ``models/inswapper`` and verify its digest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/inswapper"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    path = download_asset(INSWAPPER, args.model_dir, force=args.force)
    print(f"ready {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
