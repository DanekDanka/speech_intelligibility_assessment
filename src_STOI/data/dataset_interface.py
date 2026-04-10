from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset


class StoiPredictionDataset(Dataset, ABC):
    """
    Each item: degraded waveform chunk for the model and scalar STOI target
    (reference is used only when building the dataset / labels).
    """

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Returns dict with at least:
          - ``waveform``: FloatTensor ``(num_samples,)``
          - ``stoi_target``: FloatTensor scalar
          - ``sample_id``: str
        """

    @abstractmethod
    def split_indices(self, train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
        """Return disjoint index lists for train / val / test."""
