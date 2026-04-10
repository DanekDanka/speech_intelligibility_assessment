from __future__ import annotations

import torch
import torch.nn as nn

from .base import StoiTrainingCriterion


class MseStoiCriterion(StoiTrainingCriterion):
    """
    MSE(pred, target_stoi). Targets match ``metrics.stoi_backend`` (``torchaudio.functional.stoi`` when
    present in TorchAudio, else Taal STOI with ``torchaudio.functional.resample`` to 10 kHz).
    Config key: ``torchaudio_stoi``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._mse = nn.MSELoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = prediction.view(-1)
        tgt = target.view(-1)
        return self._mse(pred, tgt)


def build_criterion(name: str) -> StoiTrainingCriterion:
    name = name.lower().strip()
    if name in ("torchaudio_stoi", "stoi_torchaudio", "mse", "l2", "mse_stoi"):
        return MseStoiCriterion()
    raise KeyError(f"Unknown loss {name!r}")
