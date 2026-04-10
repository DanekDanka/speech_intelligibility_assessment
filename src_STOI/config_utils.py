from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Dict, MutableMapping, Union


def deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def load_json_config(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_merged_config(paths: list[Union[str, Path]]) -> Dict[str, Any]:
    if not paths:
        raise ValueError("At least one config path required")
    cfg = load_json_config(paths[0])
    for p in paths[1:]:
        cfg = deep_merge(cfg, load_json_config(p))
    return cfg
