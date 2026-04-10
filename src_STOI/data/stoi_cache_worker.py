"""
Топ-уровневые функции для multiprocessing (spawn) при заполнении STOI-кэша.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def compute_stoi_group(payload: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Один processed↔original: WAV загружены один раз, STOI по списку чанков (только CPU)."""
    root = payload.get("_repo_root")
    if root:
        root = str(root)
        if root not in sys.path:
            sys.path.insert(0, root)

    from src_STOI.metrics.stoi_backend import compute_stoi_scalar_from_tensors
    from src_STOI.preprocessing.default_preprocessors import TorchaudioResampleMonoChunkPreprocessor

    sr = int(payload["sample_rate"])
    chunk_sec = float(payload["chunk_duration_sec"])
    pre = TorchaudioResampleMonoChunkPreprocessor(sample_rate=sr, chunk_duration_sec=chunk_sec)

    proc = Path(payload["proc"])
    orig = Path(payload["orig"])
    ref_w = pre.load_mono_resampled(orig)
    deg_w = pre.load_mono_resampled(proc)

    extended = bool(payload["extended"])
    resample_mode = str(payload["resample_mode"])
    apply_silence_removal = bool(payload.get("apply_silence_removal", True))
    out: List[Tuple[str, float]] = []

    import torch

    for item in payload["items"]:
        chunk = int(item["chunk"])
        key = str(item["key"])
        ref = pre.chunk_waveform(ref_w, chunk)
        deg = pre.chunk_waveform(deg_w, chunk)
        v = compute_stoi_scalar_from_tensors(
            ref,
            deg,
            sr,
            extended=extended,
            resample_mode=resample_mode,
            device=torch.device("cpu"),
            apply_silence_removal=apply_silence_removal,
        )
        out.append((key, float(max(0.0, min(1.0, v)))))
    return out


def _pool_init(repo_root: str) -> None:
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
