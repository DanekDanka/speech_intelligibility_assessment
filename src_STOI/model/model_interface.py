from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class StoiPredictorModel(nn.Module, ABC):
    """Regressor: degraded waveform -> scalar STOI in ``(0, 1)``."""

    @abstractmethod
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: ``(batch, time)`` mono samples.
        Returns:
            Tensor of shape ``(batch, 1)``.
        """
