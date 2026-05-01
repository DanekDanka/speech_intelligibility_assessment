#!/usr/bin/env python3
"""
Строит сравнительное распределение STOI (0…1) из float-кэша PairWavStoiDataset:
зелёным — только папка ``noise/``, фиолетовым — только ``reverb/``.
Записи из ``noise_reverb/`` и прочих путей (например ``extreme_stoi``) по умолчанию пропускаются.

Пример:
  python scripts/plot_stoi_cache_noise_vs_reverb.py --cache stoi_label_cache.pkl
  python scripts/plot_stoi_cache_noise_vs_reverb.py --cache ./stoi_label_cache.pkl -o out/stoi_cmp.png
  # рядом с .png создаётся тот же stem с расширением .eps
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src_STOI.data.stoi_rich_cache import infer_noise_reverb_from_processed_path


def degraded_path_from_cache_key(key: str) -> Optional[Path]:
    """Ключ вида ``<proc>|...|c=...`` — возвращает путь к деградированному wav."""
    parts = key.split("|")
    for i, p in enumerate(parts):
        if p.startswith("c="):
            if i >= 1:
                try:
                    return Path(parts[0])
                except Exception:
                    return None
            return None
    if parts:
        try:
            cand = Path(parts[0])
            if cand.suffix.lower() == ".wav" and cand.name:
                return cand
        except Exception:
            pass
    return None


def load_stoi_float_cache(path: Path) -> dict:
    with open(path, "rb") as f:
        blob = pickle.load(f)
    if not isinstance(blob, dict):
        raise ValueError(f"Ожидался dict в {path}, получен {type(blob)}")
    return blob


def split_noise_reverb_stoi(
    cache: dict,
    *,
    assign_noise_reverb: str = "skip",
) -> Tuple[List[float], List[float], int, int]:
    """
    assign_noise_reverb: ``skip`` | ``noise`` | ``reverb`` | ``both`` —
    куда отнести чанки из папки ``noise_reverb`` (по умолчанию не в график).
    ``both`` — дублирует значение в оба списка (для плотности не идеально, но опция).
    """
    noise_vals: List[float] = []
    reverb_vals: List[float] = []
    skipped = 0
    bad = 0

    for key, val in cache.items():
        try:
            v = float(val)
        except (TypeError, ValueError):
            bad += 1
            continue
        if not (0.0 <= v <= 1.0 + 1e-6):
            v = float(np.clip(v, 0.0, 1.0))

        proc = degraded_path_from_cache_key(str(key))
        if proc is None:
            bad += 1
            continue

        n_flag, r_flag = infer_noise_reverb_from_processed_path(proc)

        if n_flag and r_flag:
            if assign_noise_reverb == "noise":
                noise_vals.append(v)
            elif assign_noise_reverb == "reverb":
                reverb_vals.append(v)
            elif assign_noise_reverb == "both":
                noise_vals.append(v)
                reverb_vals.append(v)
            else:
                skipped += 1
            continue

        if n_flag and not r_flag:
            noise_vals.append(v)
        elif r_flag and not n_flag:
            reverb_vals.append(v)
        else:
            skipped += 1

    return noise_vals, reverb_vals, skipped, bad


def _configure_matplotlib_style() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "font.style": "normal",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "mathtext.default": "regular",
            "savefig.dpi": 600,
        }
    )


def plot_histograms(
    noise_vals: List[float],
    reverb_vals: List[float],
    out_path: Path,
    *,
    bins: int = 50,
) -> None:
    _configure_matplotlib_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    edges = np.linspace(0.0, 1.0, bins + 1)

    kw = dict(bins=edges, density=True, alpha=0.55, edgecolor="black", linewidth=0.35)

    if noise_vals:
        ax.hist(noise_vals, **kw, color="#2ecc71", label="С шумом")
    if reverb_vals:
        ax.hist(reverb_vals, **kw, color="#9b59b6", label="С реверберацией")

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Индекс STOI")
    ax.set_ylabel("Плотность вероятности")
    ax.set_title("Сравнительное распределение STOI")
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, color="0.75", alpha=0.85)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.35, color="0.82", alpha=0.65)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")

    hmax = 0.0
    for vals in (noise_vals, reverb_vals):
        if vals:
            h, _ = np.histogram(vals, bins=edges, density=True)
            hmax = max(hmax, float(np.max(h)))
    ax.set_ylim(0.0, max(hmax * 1.08, 0.05))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    dpi = 600
    fig.savefig(out_path, dpi=dpi)
    eps_path = out_path.with_suffix(".eps")
    fig.savefig(eps_path, dpi=dpi, format="eps")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Гистограмма STOI из кэша: шум vs реверберация")
    p.add_argument(
        "--cache",
        type=str,
        default=str(_REPO_ROOT / "stoi_label_cache.pkl"),
        help="Путь к pickle float-кэшу (как data.cache_stoi_path)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=str(_REPO_ROOT / "test_model_plots" / "stoi_noise_vs_reverb_distribution.png"),
        help="Базовый путь для рисунка (.png и рядом .eps с тем же именем)",
    )
    p.add_argument("--bins", type=int, default=50, help="Число интервалов по оси STOI [0,1]")
    p.add_argument(
        "--noise-reverb-mode",
        type=str,
        choices=("skip", "noise", "reverb", "both"),
        default="skip",
        help="Как учитывать папку noise_reverb",
    )
    args = p.parse_args()

    cache_path = Path(args.cache).expanduser()
    if not cache_path.is_absolute():
        cache_path = (_REPO_ROOT / cache_path).resolve()

    if not cache_path.is_file():
        raise SystemExit(f"Файл кэша не найден: {cache_path}")

    cache = load_stoi_float_cache(cache_path)
    noise_vals, reverb_vals, skipped, bad = split_noise_reverb_stoi(
        cache,
        assign_noise_reverb=args.noise_reverb_mode,
    )

    print(f"Кэш: {cache_path} (записей {len(cache)})")
    print(f"  «С шумом» (только …/noise/…):     {len(noise_vals)} значений")
    print(f"  «С реверберацией» (только …/reverb/…): {len(reverb_vals)} значений")
    print(f"  пропущено (прочие пути / noise_reverb при skip): {skipped}, битых ключей: {bad}")

    if not noise_vals and not reverb_vals:
        raise SystemExit("Нет данных для графика — проверьте пути в ключах кэша.")

    out_path = Path(args.output).expanduser()
    if not out_path.is_absolute():
        out_path = (_REPO_ROOT / out_path).resolve()

    plot_histograms(noise_vals, reverb_vals, out_path, bins=max(10, args.bins))
    print(f"Сохранено: {out_path}")
    print(f"Сохранено: {out_path.with_suffix('.eps')}")


if __name__ == "__main__":
    main()
