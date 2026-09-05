"""A learned watermark, trained to survive the transform a face swap applies.

The signal-processing watermark in :mod:`deepshield.protection.watermark` reads a
globally coherent 8x8 block grid, and a face swap does not preserve one: the
source face is resampled triangle by triangle onto another person's landmarks.
Measured end to end that watermark lands at chance in the source direction, and
moving it into a landmark-normalised frame does not rescue it, because the
normalising warp and the swap's warp use different meshes. Both measurements are
in the README.

What remains is to stop requiring a grid. This module learns the carrier instead:
an encoder writes a residual conditioned on the message, a decoder reads the
message back, and between them sits a distortion layer standing in for the swap.
The three parts of that layer are not arbitrary augmentations - they are the
three mechanisms the earlier measurements identified as destructive:

non-rigid resampling
    A smooth low-frequency displacement field applied with ``grid_sample``, the
    shape a warp onto different facial geometry has.
colour statistics shift
    Per-channel gain and bias, standing in for the blend's colour matching.
low-pass loss
    Gaussian blur and noise, standing in for two bilinear resampling passes.

Each is applied at a random magnitude so the decoder learns a range rather than
one operating point, and training ramps the magnitude from zero (see
``scripts/train_learned_watermark.py``) because starting at full strength leaves
the model stuck near chance.

This lives in ``scripts/`` rather than under ``src/`` on purpose. Every module in
the package must import without torch installed - ``test_package_imports`` asserts
it, so a minimal install can still run the CLI - and these classes need torch at
definition time. It is research code either way: nothing in the pipeline
scores its output, and at the measured bit accuracy a 32-bit code cannot be read back
exactly. It is kept, and its numbers published, because it is the first thing in
this project that carries a recoverable signal through a real face swap at all.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

BITS = 32


def _block(inputs: int, outputs: int, stride: int = 1) -> nn.Sequential:
    """Return the convolution, normalisation and activation used throughout."""
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, stride, 1),
        nn.BatchNorm2d(outputs),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """Write a message into an image as an additive residual."""

    def __init__(self, bits: int = BITS, width: int = 64) -> None:
        """Build the encoder trunk and residual head."""
        super().__init__()
        self.pre = nn.Sequential(_block(3, width), _block(width, width), _block(width, width))
        self.post = nn.Sequential(_block(width + 3 + bits, width), _block(width, width))
        self.out = nn.Conv2d(width, 3, 1)

    def forward(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        """Return a residual in [-1, 1] carrying ``message``."""
        hidden = self.pre(image)
        spread = message.view(message.size(0), -1, 1, 1)
        spread = spread.expand(-1, -1, image.size(2), image.size(3))
        hidden = self.post(torch.cat([hidden, image, spread], dim=1))
        return torch.tanh(self.out(hidden))


class Decoder(nn.Module):
    """Read a message back out of a distorted image."""

    def __init__(self, bits: int = BITS, width: int = 96) -> None:
        """Build the decoder trunk and message head."""
        super().__init__()
        self.net = nn.Sequential(
            _block(3, width), _block(width, width, 2), _block(width, width),
            _block(width, width, 2), _block(width, width), _block(width, width, 2),
            _block(width, width),
        )
        self.head = nn.Linear(width, bits)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return per-bit logits."""
        hidden = self.net(image)
        pooled = F.adaptive_avg_pool2d(hidden, 1).flatten(1)
        logits: torch.Tensor = self.head(pooled)
        return logits


def displacement_field(
    count: int, size: int, cells: int, amplitude: torch.Tensor, device: torch.device | str
) -> torch.Tensor:
    """Return a smooth random displacement field, the shape a swap warp has."""
    field = torch.randn(count, 2, cells, cells, device=device)
    field = F.interpolate(field, size=(size, size), mode="bicubic", align_corners=True)
    return field * amplitude


def gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur separably, standing in for the loss of two resampling passes."""
    if sigma <= 0:
        return image
    radius = max(1, int(2 * sigma))
    width = 2 * radius + 1
    offsets = torch.arange(width, device=image.device, dtype=image.dtype) - radius
    kernel = torch.exp(-(offsets**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    rowwise = kernel.view(1, 1, 1, width).expand(3, 1, 1, width)
    columnwise = kernel.view(1, 1, width, 1).expand(3, 1, width, 1)
    image = F.conv2d(image, rowwise, padding=(0, radius), groups=3)
    return F.conv2d(image, columnwise, padding=(radius, 0), groups=3)


def distort(image: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
    """Apply a differentiable stand-in for what a face swap does to the carrier.

    ``strength`` scales all three mechanisms together and is ramped during
    training. At zero this is the identity, so the same call site can measure a
    clean baseline.
    """
    count, _, height, width = image.shape
    device = image.device
    rows, columns = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing="ij",
    )
    base = torch.stack([columns, rows], dim=-1).unsqueeze(0).expand(count, -1, -1, -1)
    amplitude = (0.02 + 0.10 * torch.rand(count, 1, 1, 1, device=device)) * strength
    field = displacement_field(count, height, 5, amplitude, device).permute(0, 2, 3, 1)
    warped = F.grid_sample(
        image, (base + field).clamp(-1, 1), align_corners=True, padding_mode="border"
    )
    gain = 1.0 + (torch.rand(count, 3, 1, 1, device=device) - 0.5) * 0.30 * strength
    bias = (torch.rand(count, 3, 1, 1, device=device) - 0.5) * 0.30 * strength
    warped = warped * gain + bias
    if strength > 0:
        warped = gaussian_blur(warped, float(torch.rand(1).item()) * 1.2 * strength)
        warped = warped + torch.randn_like(warped) * 0.02 * strength
    return warped.clamp(-1, 1)
