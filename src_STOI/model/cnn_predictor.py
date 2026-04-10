from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from .model_interface import StoiPredictorModel


class CnnStoiPredictor(StoiPredictorModel):
    """1D CNN + global pooling + MLP head (no external encoders)."""

    def __init__(
        self,
        in_channels: int = 1,
        num_filters: Sequence[int] = (64, 128, 256, 512),
        kernel_sizes: Sequence[int] = (11, 7, 5, 3),
        stride: int = 2,
        dropout: float = 0.2,
        fc_hidden_dim: int = 512,
        num_fc_layers: int = 3,
    ) -> None:
        super().__init__()
        if len(num_filters) != len(kernel_sizes):
            raise ValueError("num_filters and kernel_sizes must match")
        layers: List[nn.Module] = []
        c = in_channels
        for out_ch, k in zip(num_filters, kernel_sizes):
            layers.append(nn.Conv1d(c, out_ch, k, stride=stride, padding=k // 2))
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            c = out_ch
        self.conv = nn.Sequential(*layers)
        d = num_filters[-1] * 2
        fc: List[nn.Module] = []
        fc.append(nn.Linear(d, fc_hidden_dim))
        fc.append(nn.LayerNorm(fc_hidden_dim))
        fc.append(nn.ReLU(inplace=True))
        fc.append(nn.Dropout(dropout))
        for _ in range(num_fc_layers - 1):
            fc.append(nn.Linear(fc_hidden_dim, fc_hidden_dim))
            fc.append(nn.LayerNorm(fc_hidden_dim))
            fc.append(nn.ReLU(inplace=True))
            fc.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*fc)
        self.head = nn.Sequential(
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = waveform.unsqueeze(1)
        x = self.conv(x)
        avg = torch.nn.functional.adaptive_avg_pool1d(x, 1).squeeze(-1)
        mx = torch.nn.functional.adaptive_max_pool1d(x, 1).squeeze(-1)
        h = torch.cat([avg, mx], dim=1)
        h = self.fc(h)
        return self.head(h)
