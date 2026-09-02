"""Watermark extraction interface.

Extraction is separated from embedding because the two run in different places:
embedding happens once when the user protects an image, extraction runs on every
piece of suspect content.

A negative result carries little information. An attacker can crop, re-encode or
regenerate an image and destroy the mark, so ``detected=False`` is treated as
neutral by the risk engine while ``detected=True`` is strong positive evidence.

Implementation lands in Phase 9 alongside the embedder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from deepshield.types import WatermarkDetectionResult


class WatermarkDetector(ABC):
    """Contract for recovering a watermark payload from an image."""

    name: str = "abstract"

    @abstractmethod
    def detect(self, image: np.ndarray) -> WatermarkDetectionResult:
        """Attempt to recover a payload from ``image``.

        Returns:
            A result whose ``confidence`` reflects decoder certainty, and whose
            ``detected=False`` case must be read as inconclusive.

        """
