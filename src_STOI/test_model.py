"""
Оценка обученной модели на train/val/test сплите (как при обучении из того же config).

По умолчанию: STOI-Net (``checkpoints_src_stoi_net/best.pt`` + ``configs/train_stoi_net.json``).

Пример:
  python src_STOI/test_model.py
  python src_STOI/test_model.py --split val
  python src_STOI/test_model.py --checkpoint checkpoints_src_stoi/best.pt --config src_STOI/configs/example_train.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    stoi_computer = TorchaudioBackedStoiTargetComputer(
        sample_rate,
        extended=bool(stoi_cfg.get("extended", False)),
        resample_mode=str(stoi_cfg.get("resample_mode", "torchaudio")),
        compute_device=stoi_cfg.get("compute_device"),
    )

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
        cache_stoi_path=data_cfg.get("cache_stoi_path"),
        max_items=data_cfg.get("max_items"),
        stoi_cache_num_workers=data_cfg.get("stoi_cache_num_workers"),
        index_fraction=idx_frac,
        index_sample_seed=seed + 91231,
        shuffle_chunks=shuffle_chunks,
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


if __name__ == "__main__":
    main()
