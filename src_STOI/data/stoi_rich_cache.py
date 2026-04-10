"""
Расширенный STOI-кэш: пути degraded/reference, значение STOI, флаги шума и реверба.
Используется train_partly и опционально PairWavStoiDataset (rich_cache_path).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

RICH_CACHE_FORMAT_VERSION = 1


def infer_noise_reverb_from_processed_path(processed_path: Path) -> Tuple[bool, bool]:
    """
    По компонентам пути (папки CMU / Vox): noise, reverb, noise_reverb.
    ``extreme_stoi`` и прочие папки без этих имён дают (False, False).
    """
    pset = frozenset(processed_path.resolve().parts)
    if "noise_reverb" in pset:
        return True, True
    return ("noise" in pset), ("reverb" in pset)


def load_rich_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.is_file():
        return {}
    with open(path, "rb") as f:
        blob = pickle.load(f)
    if not isinstance(blob, dict):
        return {}
    ent = blob.get("entries")
    if isinstance(ent, dict):
        return dict(ent)
    return {}


def save_rich_cache(path: Path, entries: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": RICH_CACHE_FORMAT_VERSION, "entries": entries}
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def merge_float_cache_for_keys(
    legacy_path: Path,
    needed_keys: set[str],
    target: Dict[str, float],
) -> int:
    """Подмешивает в target отсутствующие ключи из legacy pickle (только из needed_keys). Число добавленных."""
    if not legacy_path.is_file():
        return 0
    with open(legacy_path, "rb") as f:
        blob = pickle.load(f)
    if not isinstance(blob, dict):
        return 0
    added = 0
    for k in needed_keys:
        if k not in target and k in blob and isinstance(blob[k], (int, float)):
            target[k] = float(blob[k])
            added += 1
    return added
