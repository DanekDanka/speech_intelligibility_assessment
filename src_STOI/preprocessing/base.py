from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union

import torch


class WaveformPreprocessor(ABC):
    """Transforms raw audio into model input (fixed-length mono tensor)."""

    @abstractmethod
    def process_degraded_file(self, path: Union[str, Path], chunk_index: int) -> torch.Tensor:
        """Return shape ``(num_samples,)`` float tensor."""

    @abstractmethod
    def process_reference_file(self, path: Union[str, Path], chunk_index: int) -> torch.Tensor:
        """Return shape ``(num_samples,)`` for STOI target (aligned chunk)."""


class StoiTargetComputer(ABC):
    """Computes scalar STOI(reference_chunk, degraded_chunk) for supervision."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Sample rate of waveforms passed to ``compute`` (Hz)."""

    @abstractmethod
    def compute(self, reference: torch.Tensor, degraded: torch.Tensor) -> float:
        """1D CPU or float tensors of equal length."""

    def compute_batch_metadata(self) -> Dict[str, Any]:
        return {}
