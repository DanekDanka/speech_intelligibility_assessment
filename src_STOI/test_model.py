"""
Оценка обученной модели на train/val/test сплите (как при обучении из того же config).

Метки STOI: ``PairWavStoiDataset`` подгружает ``data.cache_stoi_path`` (путь относительно корня
репозитория, если не абсолютный). Если файла нет — считает недостающие ключи и сохраняет кэш.
Если ``cache_stoi_path`` в конфиге пуст — по умолчанию ``<repo>/stoi_label_cache.pkl``.

По умолчанию: STOI-Net (``checkpoints_src_stoi_net/best.pt`` + ``configs/train_stoi_net.json``).

Пример:
  python src_STOI/test_model.py
  python src_STOI/test_model.py --split val
  python src_STOI/test_model.py --checkpoint checkpoints_src_stoi/best.pt --config src_STOI/configs/example_train.json
  python src_STOI/test_model.py --plots-dir ./my_plots
  python src_STOI/test_model.py --no-plots
"""

from __future__ import annotations

import argparse
import json
import sys
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src_STOI.config_utils import load_merged_config
from src_STOI.data import PairWavStoiDataset, subset_by_indices
from src_STOI.model import build_model
from src_STOI.preprocessing import (
    TorchaudioBackedStoiTargetComputer,
    TorchaudioResampleMonoChunkPreprocessor,
)


def _resolve_under_repo(path_or_str: Optional[Union[str, Path]], repo_root: Path) -> Optional[Path]:
    """Относительные пути — от корня репозитория (как при train из корня проекта)."""
    if path_or_str is None:
        return None
    if isinstance(path_or_str, str) and not path_or_str.strip():
        return None
    p = Path(path_or_str).expanduser()
    if not p.is_absolute():
        p = (repo_root / p).resolve()
    return p


def _resolve_path_list(paths: Optional[Sequence[Union[str, Path]]], repo_root: Path) -> List[Path]:
    if not paths:
        return []
    out: List[Path] = []
    for x in paths:
        r = _resolve_under_repo(x, repo_root)
        if r is not None:
            out.append(r)
    return out


def collate_stoi_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    waveforms = torch.stack([b["waveform"] for b in batch], dim=0)
    targets = torch.stack([b["stoi_target"] for b in batch], dim=0)
    ids = [b["sample_id"] for b in batch]
    return {"waveform": waveforms, "stoi_target": targets, "sample_id": ids}


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: List[float] = []
    tgts: List[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="inference", leave=False):
            x = batch["waveform"].to(device, non_blocking=True)
            y = batch["stoi_target"]
            p = model(x).view(-1).cpu().numpy()
            t = y.view(-1).numpy()
            preds.extend(p.tolist())
            tgts.extend(t.tolist())
    return np.asarray(preds, dtype=np.float64), np.asarray(tgts, dtype=np.float64)


def compute_metrics(pred: np.ndarray, tgt: np.ndarray) -> Dict[str, float]:
    err = pred - tgt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    ss_res = float(np.sum((tgt - pred) ** 2))
    ss_tot = float(np.sum((tgt - np.mean(tgt)) ** 2))
    if ss_tot > 1e-14:
        r2 = 1.0 - ss_res / ss_tot
    else:
        r2 = float("nan")

    try:
        from scipy.stats import spearmanr

        sp = spearmanr(pred, tgt)
        spearman_rho = float(sp.correlation) if sp.correlation is not None and not np.isnan(sp.correlation) else float("nan")
    except Exception:
        spearman_rho = float("nan")

    abs_err = np.abs(err)
    out: Dict[str, float] = {
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "spearman_rho": spearman_rho,
    }
    for tol in (0.05, 0.10, 0.15, 0.20):
        key = f"pct_abs_err_le_{tol:.2f}".replace(".", "_")
        out[key] = float(np.mean(abs_err <= tol) * 100.0)
    return out


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


def save_test_plots(
    pred: np.ndarray,
    tgt: np.ndarray,
    out_dir: Path,
    stem: str,
) -> Tuple[Path, Path, Path, Path]:
    """Графики: распределение ошибки (pred − target) и предсказание vs истина (PNG и EPS)."""
    _configure_matplotlib_style()
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    err = pred - tgt
    path_err_png = out_dir / f"{stem}_error_distribution.png"
    path_err_eps = out_dir / f"{stem}_error_distribution.eps"
    path_scatter_png = out_dir / f"{stem}_pred_vs_target.png"
    path_scatter_eps = out_dir / f"{stem}_pred_vs_target.eps"

    dpi = 600

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    err_lo, err_hi = -0.2, 0.2
    ax1.hist(
        err,
        bins=min(100, max(20, int(np.sqrt(len(err))))),
        range=(err_lo, err_hi),
        color="steelblue",
        edgecolor="white",
        alpha=0.88,
    )
    ax1.axvline(0.0, color="black", linewidth=1.2, label="Нулевая ошибка")
    ax1.set_xlim(err_lo, err_hi)
    ax1.set_xlabel("Ошибка предсказания STOI (предсказанное − истинное)")
    ax1.set_ylabel("Количество")
    ax1.set_title("Распределение ошибок предсказания STOI")
    ax1.grid(True, which="major", linestyle="-", linewidth=0.5, color="0.75", alpha=0.85)
    ax1.minorticks_on()
    ax1.grid(True, which="minor", linestyle=":", linewidth=0.35, color="0.82", alpha=0.65)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right")
    fig1.tight_layout()
    fig1.savefig(path_err_png, dpi=dpi)
    fig1.savefig(path_err_eps, dpi=dpi, format="eps")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6.5, 6.5))
    n = int(len(tgt))
    # Классический scatter: субсэмплинг вместо hexbin, чтобы не «пятнить» картинку.
    max_pts = 12_000
    if n > max_pts:
        rng = np.random.default_rng(42)
        sub = rng.choice(n, size=max_pts, replace=False)
        t, p = tgt[sub], pred[sub]
    else:
        t, p = tgt, pred

    ax2.scatter(
        t,
        p,
        s=14 if n < 800 else 10,
        alpha=0.45 if n < 800 else 0.28,
        c="#1f77b4",
        edgecolors="white",
        linewidths=0.25,
        rasterized=True,
        zorder=2,
    )

    lo = float(min(t.min(), p.min(), 0.0))
    hi = float(max(t.max(), p.max(), 1.0))
    pad = 0.02 * (hi - lo + 1e-6)
    ax2.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        color="black",
        linestyle="--",
        linewidth=1.25,
        dashes=(5, 3),
        label="Идеал: y = x",
        zorder=3,
    )
    ax2.set_xlim(lo - pad, hi + pad)
    ax2.set_ylim(lo - pad, hi + pad)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("Истинное STOI")
    ax2.set_ylabel("Предсказанное STOI")
    ax2.set_title("Предсказанное vs истинное STOI")
    ax2.grid(True, which="major", linestyle="-", linewidth=0.6, color="0.75", alpha=0.85, zorder=1)
    ax2.set_axisbelow(True)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.9)
    ax2.legend(loc="lower right", frameon=True, fancybox=False, edgecolor="0.4")
    fig2.tight_layout()
    fig2.savefig(path_scatter_png, dpi=dpi)
    fig2.savefig(path_scatter_eps, dpi=dpi, format="eps")
    plt.close(fig2)

    return path_err_png, path_err_eps, path_scatter_png, path_scatter_eps


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Тест обученной модели STOI (метрики на сплите)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(_REPO_ROOT / "checkpoints_src_stoi_net" / "best.pt"),
        help="Путь к best.pt (или last.pt); по умолчанию STOI-Net",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(here / "configs" / "train_stoi_net.json"),
        help="Тот же JSON, что и при train (данные и seed); по умолчанию как train.py (STOI-Net)",
    )
    parser.add_argument("--split", type=str, choices=("train", "val", "test"), default="test")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="По умолчанию из config.training.batch_size")
    parser.add_argument("--metrics-json", type=str, default=None, help="Сохранить метрики в JSON")
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Каталог для рисунков PNG и EPS (по умолчанию: <repo>/test_model_plots)",
    )
    parser.add_argument("--no-plots", action="store_true", help="Не строить графики")
    args = parser.parse_args()

    cfg = load_merged_config([args.config])
    seed = int(cfg.get("seed", 42))
    device_s = args.device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    data_cfg = cfg["data"]
    stoi_cfg = cfg.get("stoi_target", {})
    sample_rate = int(data_cfg["sample_rate"])
    chunk_sec = float(data_cfg["chunk_duration_sec"])

    pre = TorchaudioResampleMonoChunkPreprocessor(sample_rate=sample_rate, chunk_duration_sec=chunk_sec)
    apply_silence_removal = bool(stoi_cfg.get("apply_silence_removal", False))
    stoi_computer = TorchaudioBackedStoiTargetComputer(
        sample_rate,
        extended=bool(stoi_cfg.get("extended", False)),
        resample_mode=str(stoi_cfg.get("resample_mode", "torchaudio")),
        compute_device=stoi_cfg.get("compute_device"),
        apply_silence_removal=apply_silence_removal,
    )

    cache_raw = data_cfg.get("cache_stoi_path")
    if cache_raw is None or (isinstance(cache_raw, str) and not str(cache_raw).strip()):
        cache_stoi_path: Optional[Path] = _REPO_ROOT / "stoi_label_cache.pkl"
        print(
            f"data.cache_stoi_path не задан — по умолчанию {cache_stoi_path} "
            f"(загрузка при наличии файла, иначе расчёт и сохранение)"
        )
    else:
        cache_stoi_path = _resolve_under_repo(cache_raw, _REPO_ROOT)

    rich_raw = data_cfg.get("stoi_rich_cache_path")
    rich_cache_path = _resolve_under_repo(rich_raw, _REPO_ROOT) if rich_raw else None
    legacy_paths = _resolve_path_list(data_cfg.get("stoi_import_legacy_cache_paths"), _REPO_ROOT)

    if cache_stoi_path is not None:
        if cache_stoi_path.exists():
            try:
                with open(cache_stoi_path, "rb") as f:
                    blob = pickle.load(f)
                n_ent = len(blob) if isinstance(blob, dict) else 0
            except Exception:
                n_ent = -1
            extra = f"{n_ent} записей" if n_ent >= 0 else "не удалось прочитать"
            print(f"STOI кэш на диске: найден {cache_stoi_path} ({extra}) — подгрузка + добор недостающих ключей при необходимости")
        else:
            print(f"STOI кэш на диске: файла нет ({cache_stoi_path}) — метки будут посчитаны и файл сохранён")
    else:
        print("STOI кэш: путь не задан — только в памяти, на диск не пишется")

    base = Path(data_cfg["audio_base_dir"])
    subdirs = data_cfg.get("subdirs")
    audio_dirs = [base / s for s in subdirs] if subdirs else [base]

    idx_frac = float(data_cfg.get("index_fraction", 1.0))
    shuffle_chunks = bool(data_cfg.get("shuffle_chunks", True))
    ds = PairWavStoiDataset(
        audio_dirs=audio_dirs,
        original_dir=data_cfg["original_dir"],
        waveform_preprocessor=pre,
        stoi_computer=stoi_computer,
        single_chunk_per_file=bool(data_cfg.get("single_chunk_per_file", False)),
        cache_stoi_path=cache_stoi_path,
        max_items=data_cfg.get("max_items"),
        stoi_cache_num_workers=data_cfg.get("stoi_cache_num_workers"),
        index_fraction=idx_frac,
        index_sample_seed=seed + 91231,
        shuffle_chunks=shuffle_chunks,
        rich_cache_path=rich_cache_path,
        import_legacy_cache_paths=legacy_paths or None,
    )

    tr_r = float(data_cfg.get("train_ratio", 0.8))
    va_r = float(data_cfg.get("val_ratio", 0.1))
    te_r = float(data_cfg.get("test_ratio", 0.1))
    train_idx, val_idx, test_idx = ds.split_indices(tr_r, va_r, te_r, seed)

    if args.split == "train":
        idx = train_idx
    elif args.split == "val":
        idx = val_idx
    else:
        idx = test_idx

    t_cfg = cfg["training"]
    bs = int(args.batch_size or t_cfg.get("batch_size", 16))
    nw = int(t_cfg.get("num_workers", 0))
    loader = DataLoader(
        subset_by_indices(ds, idx),
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=device.type == "cuda",
        collate_fn=collate_stoi_batch,
    )

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = (_REPO_ROOT / ckpt_path).resolve()

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    m_cfg = ckpt["model"]
    model = build_model(m_cfg["name"], m_cfg.get("kwargs") or {})
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)

    print("=" * 60)
    print(f"Чекпоинт: {ckpt_path}")
    print(f"Конфиг:   {args.config}")
    print(f"Сплит:    {args.split} (n={len(idx)}), seed={seed}")
    print(f"Устройство: {device}")
    print(f"Модель:   {m_cfg['name']}, эпоха в ckpt: {ckpt.get('epoch', '?')}")
    print("=" * 60)

    pred, tgt = collect_predictions(model, loader, device)
    metrics = compute_metrics(pred, tgt)

    print("\nМетрики (pred vs target_stoi из датасета):\n")
    print(f"  MAE:              {metrics['mae']:.6f}")
    print(f"  RMSE:             {metrics['rmse']:.6f}")
    print(f"  R²:               {metrics['r2']:.6f}")
    print(f"  Spearman ρ:       {metrics['spearman_rho']:.6f}")
    print("\n  Доля примеров с |pred − target| ≤ порога (%):\n")
    for tol in (0.05, 0.10, 0.15, 0.20):
        key = f"pct_abs_err_le_{tol:.2f}".replace(".", "_")
        print(f"    ≤ {tol:.2f}:  {metrics[key]:.2f}%")

    if args.metrics_json:
        out_path = Path(args.metrics_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(ckpt_path.resolve()),
            "config": str(Path(args.config).resolve()),
            "split": args.split,
            "n_samples": int(len(idx)),
            "metrics": metrics,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nМетрики записаны в {out_path}")

    if not args.no_plots:
        plots_dir = Path(args.plots_dir).expanduser() if args.plots_dir else (_REPO_ROOT / "test_model_plots")
        if not plots_dir.is_absolute():
            plots_dir = (_REPO_ROOT / plots_dir).resolve()
        plot_stem = f"{args.split}_{ckpt_path.stem}"
        try:
            pe_png, pe_eps, ps_png, ps_eps = save_test_plots(pred, tgt, plots_dir, plot_stem)
            print(f"\nГрафики: {pe_png}")
            print(f"          {pe_eps}")
            print(f"          {ps_png}")
            print(f"          {ps_eps}")
        except ImportError as e:
            print(f"\nГрафики пропущены (matplotlib): {e}")


if __name__ == "__main__":
    main()
