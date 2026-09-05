"""Train the learned watermark against the swap-shaped distortion layer.

Two settings here were chosen by measurement rather than taste.

The distortion strength is ramped from zero over the first part of training. A
first run applied it at full strength from step one and stalled at 0.614 bit
accuracy by step 1000; with the ramp the same step reached 0.623 and kept
climbing to 0.809. Starting hard leaves the model in a shallow solution.

The message and image losses are weighted against each other because they pull in
opposite directions: the residual has to be strong enough to survive resampling
and faint enough to stay invisible. ``--img-weight`` is the knob, and the PSNR
reported each validation is what it buys.

Validation reports bit accuracy at three distortion strengths. The clean column
is the model's own ceiling - what it can do with nothing done to the carrier -
and no amount of robustness work raises it.

Usage:
    python scripts/train_learned_watermark.py --data data/results/face_crops_128.npy \
        --out models/learned_watermark.pt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from learned_watermark import BITS, Decoder, Encoder, distort

VALIDATION = 512
WARMUP_FRACTION = 0.4


def peak_snr(marked: torch.Tensor, original: torch.Tensor) -> float:
    """Return PSNR in decibels for tensors scaled to [-1, 1]."""
    error = ((marked - original) ** 2).mean().item()
    return float(10 * np.log10(4.0 / max(error, 1e-12)))


def main(argv: list[str] | None = None) -> int:
    """Train the encoder and decoder jointly and save both."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=9000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--img-weight", type=float, default=1.5)
    parser.add_argument("--res-scale", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args(argv)

    if not args.data.is_file():
        raise SystemExit(f"missing {args.data}; run scripts/prepare_face_crops.py first")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    crops = np.load(args.data)
    if crops.shape[0] <= VALIDATION:
        raise SystemExit(
            f"{args.data} holds {crops.shape[0]} crops, too few to hold out {VALIDATION}"
        )
    held_out = torch.from_numpy(crops[:VALIDATION]).permute(0, 3, 1, 2)
    held_out = held_out.float().div(127.5).sub(1).to(device)
    training = torch.from_numpy(crops[VALIDATION:]).permute(0, 3, 1, 2)
    print(f"device={device} train={tuple(training.shape)} val={tuple(held_out.shape)}", flush=True)

    encoder, decoder = Encoder().to(device), Decoder().to(device)
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimiser = torch.optim.Adam(parameters, lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.steps)

    started = time.time()
    for step in range(1, args.steps + 1):
        warmed = min(1.0, step / (args.steps * WARMUP_FRACTION))
        batch = training[torch.randint(0, training.size(0), (args.batch,))]
        images = batch.float().div(127.5).sub(1).to(device)
        message = torch.randint(0, 2, (args.batch, BITS), device=device).float()
        marked = (images + encoder(images, message) * args.res_scale).clamp(-1, 1)
        logits = decoder(distort(marked, warmed))
        message_loss = F.binary_cross_entropy_with_logits(logits, message)
        image_loss = F.mse_loss(marked, images)
        loss = message_loss + args.img_weight * image_loss
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        schedule.step()

        if step % 250 == 0 or step == 1:
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                probe = torch.randint(0, 2, (VALIDATION, BITS), device=device).float()
                candidate = (
                    held_out + encoder(held_out, probe) * args.res_scale
                ).clamp(-1, 1)
                accuracies = []
                for level in (0.0, 0.5, 1.0):
                    read = (decoder(distort(candidate, level)) > 0).float()
                    accuracies.append((read == probe).float().mean().item())
            print(
                f"step {step:5d} loss {loss.item():.4f} msg {message_loss.item():.4f} "
                f"psnr {peak_snr(candidate, held_out):.1f} "
                f"acc[clean/mid/full] {accuracies[0]:.3f}/{accuracies[1]:.3f}/{accuracies[2]:.3f} "
                f"warm {warmed:.2f} {time.time() - started:.0f}s",
                flush=True,
            )
            encoder.train()
            decoder.train()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "bits": BITS,
            "res_scale": args.res_scale,
        },
        args.out,
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
