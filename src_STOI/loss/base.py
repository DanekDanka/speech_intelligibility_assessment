from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class StoiTrainingCriterion(nn.Module, ABC):
    """Loss between model output and supervision target."""

    @abstractmethod
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prediction: ``(batch, 1)`` model output.
            target: ``(batch,)`` or ``(batch, 1)`` STOI targets.
        """
