"""Generic image-manipulation forensics interface.

Distinct from deepfake detection: this covers classical tampering evidence such
as JPEG ghosts, error level analysis, resampling traces, copy-move regions and
noise inconsistency. Those signals fire on ordinary photo edits too, which is
exactly why the score is kept separate from the synthetic-media score instead of
being folded into it.

Implementation is not scheduled inside the MVP; the interface exists so the risk
engine can already reserve a slot for the signal.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from deepshield.types import ModelInfo


class ManipulationDetector(ABC):
    """Contract for classical image tampering analysis."""

    name: str = "abstract"

    @property
    @abstractmethod
    def model_info(self) -> ModelInfo:
        """Identity and version of the analysis method."""

    @abstractmethod
    def analyze(self, image: np.ndarray) -> dict[str, float]:
        """Return named manipulation signals in ``[0, 1]`` for one image."""
