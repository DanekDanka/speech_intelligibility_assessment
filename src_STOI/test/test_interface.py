"""
Contract for prediction dumps produced by ``train.py`` (no model imports here).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union


@dataclass(frozen=True)
class PredictionPair:
    sample_id: str
    predicted: float
    target: float


def load_prediction_file(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pairs_from_payload(payload: Dict[str, Any]) -> List[PredictionPair]:
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Payload must contain list 'entries'")
    out: List[PredictionPair] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        out.append(
            PredictionPair(
                sample_id=str(e["sample_id"]),
                predicted=float(e["predicted"]),
                target=float(e["target"]),
            )
        )
    return out


def arrays_from_pairs(pairs: Sequence[PredictionPair]):
    import numpy as np

    pred = np.array([p.predicted for p in pairs], dtype=np.float64)
    tgt = np.array([p.target for p in pairs], dtype=np.float64)
    return pred, tgt
