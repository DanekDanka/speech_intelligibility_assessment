from __future__ import annotations

from typing import Any, Dict, Type

import torch.nn as nn

from .cnn_predictor import CnnStoiPredictor
from .model_interface import StoiPredictorModel
from .stoi_net_predictor import StoiNetPredictor

MODEL_REGISTRY: Dict[str, Type[StoiPredictorModel]] = {
    "cnn_stoi_predictor": CnnStoiPredictor,
    "stoi_net_predictor": StoiNetPredictor,
}


def build_model(name: str, kwargs: Dict[str, Any] | None = None) -> StoiPredictorModel:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}")
    kw = dict(kwargs or {})
    return MODEL_REGISTRY[name](**kw)
