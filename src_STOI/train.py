from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src_STOI.config_utils import load_merged_config
from src_STOI.data import PairWavStoiDataset, subset_by_indices
from src_STOI.loss import build_criterion
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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices_two_way(n: int, train_ratio: float, seed: int, seed_offset: int = 7919) -> Tuple[List[int], List[int]]:
    """Делит индексы 0..n-1 на train и val (без пересечения). train_ratio ∈ [0, 1]."""
    if n <= 0:
        return [], []
    if train_ratio < 0 or train_ratio > 1:
        raise ValueError("train_ratio must be in [0, 1]")
    g = torch.Generator().manual_seed(int(seed) + int(seed_offset))
    perm = torch.randperm(n, generator=g).tolist()
    if train_ratio >= 1.0 - 1e-12:
        return perm, []
    if train_ratio <= 1e-12:
        return [], perm
    n_tr = int(train_ratio * n)
    n_tr = max(0, min(n, n_tr))
    n_va = n - n_tr
    if n_va == 0 and n > 1:
        n_tr = n - 1
    elif n_tr == 0 and n > 1:
        n_tr = 1
    return perm[:n_tr], perm[n_tr:]


def subsample_train_dataset(
    train_ds: Any,
    fraction: float,
    seed: int,
    *,
    seed_offset: int = 17041,
) -> Tuple[Any, int, int]:
    """
    Случайная подвыборка обучающего датасета (детерминированно по seed).
    Возвращает (новый датасет, было чанков, стало чанков).
    """
    n = len(train_ds)
    if n == 0:
        return train_ds, 0, 0
    if fraction >= 1.0 - 1e-12:
        return train_ds, n, n
    if fraction <= 0 or fraction > 1:
        raise ValueError("train_data_fraction must be in (0, 1]")
    k = max(1, int(n * fraction))
    k = min(k, n)
    g = torch.Generator().manual_seed(int(seed) + int(seed_offset))
    perm = torch.randperm(n, generator=g).tolist()
    return subset_by_indices(train_ds, perm[:k]), n, k


def make_optimizer(name: str, model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name_l = name.lower()
    lr = float(cfg.get("lr", 1e-4))
    wd = float(cfg.get("weight_decay", 0.0))
    if name_l == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name_l == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name_l == "sgd":
        mom = float(cfg.get("momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=mom, weight_decay=wd)
    raise KeyError(f"Unknown optimizer {name!r}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    grad_clip: float,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n = 0
    use_amp = scaler is not None and device.type == "cuda"
    for batch in tqdm(loader, desc="train", total=len(loader), leave=False, unit="batch"):
        x = batch["waveform"].to(device, non_blocking=True)
        y = batch["stoi_target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        with torch.no_grad():
            mae = (pred.view(-1) - y.view(-1)).abs().mean()
        total_loss += float(loss.item())
        total_mae += float(mae.item())
        n += 1
    return total_loss / max(n, 1), total_mae / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    preds: List[float] = []
    tgts: List[float] = []
    n = 0
    for batch in tqdm(loader, desc="val", total=len(loader), leave=False, unit="batch"):
        x = batch["waveform"].to(device, non_blocking=True)
        y = batch["stoi_target"].to(device, non_blocking=True)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += float(loss.item())
        total_mae += float((pred.view(-1) - y.view(-1)).abs().mean().item())
        preds.extend(pred.view(-1).cpu().numpy().tolist())
        tgts.extend(y.view(-1).cpu().numpy().tolist())
        n += 1
    if len(preds) > 1:
        c = np.corrcoef(preds, tgts)[0, 1]
        r = float(c) if not np.isnan(c) else 0.0
    else:
        r = 0.0
    return total_loss / max(n, 1), total_mae / max(n, 1), r


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> List[Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    for batch in tqdm(loader, desc="test predict", total=len(loader), leave=False, unit="batch"):
        x = batch["waveform"].to(device, non_blocking=True)
        y = batch["stoi_target"]
        pred = model(x).view(-1).cpu().numpy().tolist()
        tgt = y.view(-1).numpy().tolist()
        for sid, p, t in zip(batch["sample_id"], pred, tgt):
            rows.append({"sample_id": sid, "predicted": float(p), "target": float(t)})
    return rows


def main() -> None:
    _default_cfg = (Path(__file__).resolve().parent / "configs" / "train_stoi_net.json").as_posix()
    parser = argparse.ArgumentParser(description="Train STOI regressor (src_STOI)")
    parser.add_argument(
        "--config",
        type=str,
        default=_default_cfg,
        help=f"Base JSON config (default: {_default_cfg})",
    )
    parser.add_argument(
        "--config-override",
        type=str,
        action="append",
        default=[],
        help="Additional JSON merged into base (for hyperparameter search)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--export-test-predictions", action="store_true")
    parser.add_argument("--test-predictions-path", type=str, default=None)
    args = parser.parse_args()

    paths = [args.config] + list(args.config_override)
    print("=" * 72)
    print("src_STOI / обучение предсказателя STOI")
    print("=" * 72)
    print(f"Базовый конфиг: {paths[0]}")
    if paths[1:]:
        print(f"Переопределения (--config-override): {paths[1:]}")

    cfg = load_merged_config(paths)
    print("Конфиг загружен (после merge).")

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    print(f"seed={seed}")

    device_s = args.device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)
    cuda_name = ""
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_name = f" ({torch.cuda.get_device_name(0)})"
    print(f"Устройство: {device_s}{cuda_name}")

    data_cfg = cfg["data"]
    stoi_cfg = cfg.get("stoi_target", {})
    sample_rate = int(data_cfg["sample_rate"])
    chunk_sec = float(data_cfg["chunk_duration_sec"])

    pre = TorchaudioResampleMonoChunkPreprocessor(sample_rate=sample_rate, chunk_duration_sec=chunk_sec)
    # Метки STOI по всему чанку: без отбрасывания «тихих» STFT-окон (Taal remove_silent_frames).
    # Опционально вернуть старое поведение: "apply_silence_removal": true в stoi_target конфига.
    apply_silence_removal = bool(stoi_cfg.get("apply_silence_removal", False))
    stoi_computer = TorchaudioBackedStoiTargetComputer(
        sample_rate,
        extended=bool(stoi_cfg.get("extended", False)),
        resample_mode=str(stoi_cfg.get("resample_mode", "torchaudio")),
        compute_device=stoi_cfg.get("compute_device"),
        apply_silence_removal=apply_silence_removal,
    )

    base = Path(data_cfg["audio_base_dir"])
    subdirs = data_cfg.get("subdirs")
    if subdirs:
        audio_dirs = [base / s for s in subdirs]
    else:
        audio_dirs = [base]

    print("-" * 72)
    print("Данные")
    print(f"  processed: {base}  (подпапки: {subdirs if subdirs else '— вся база'})")
    for d in audio_dirs:
        exists = "OK" if d.exists() else "НЕТ"
        print(f"  каталог [{exists}]: {d}")
    print(f"  оригиналы (clean): {data_cfg['original_dir']}")
    print(f"  sample_rate={sample_rate} Hz, длина чанка={chunk_sec} с ({pre.chunk_samples} сэмплов)")
    print(
        f"  train_data_fraction={float(data_cfg.get('train_data_fraction', 1.0))} "
        f"(доля чанков train после сплита; 1.0 = весь train)"
    )
    print(
        f"  index_fraction={float(data_cfg.get('index_fraction', 1.0))} "
        f"(доля чанков до STOI-кэша и в датасете; 1.0 = все чанки)"
    )
    _vhf = float(data_cfg.get("val_holdout_fraction", 0.01))
    print(
        f"  train_on_all_data={bool(data_cfg.get('train_on_all_data', True))} "
        f"(true: почти все чанки в train; val = val_holdout_fraction от CMU и от Vox train)"
    )
    print(
        f"  val_holdout_fraction={_vhf} "
        f"(доля чанков под val при train_on_all; индексы из готового кэша, STOI не пересчитывается; 0 = без val)"
    )
    print(f"  shuffle_chunks={bool(data_cfg.get('shuffle_chunks', True))} (перемешивание порядка чанков в датасете)")
    print(
        f"  STOI-метки: extended={stoi_cfg.get('extended', False)}, "
        f"resample_mode={stoi_cfg.get('resample_mode', 'torchaudio')}, "
        f"compute_device={stoi_computer.stoi_compute_device}, "
        f"apply_silence_removal={apply_silence_removal} "
        f"(False = весь чанк, без VAD-подобного отбора тишины в Taal)"
    )
    scw = data_cfg.get("stoi_cache_num_workers")
    scw_note = "авто (CPU−1, max 16)" if scw is None or scw == 0 else str(scw)
    print(f"  кэш STOI: группировка по файлам + процессов={scw_note} (1 = только последовательно, можно GPU)")

    cache_path = data_cfg.get("cache_stoi_path")
    print(f"  кэш меток STOI: {cache_path or 'нет (только в памяти)'}")
    print("Сбор датасета и расчёт/загрузка меток STOI (может занять время при первом запуске)...")

    vox_cfg = cfg.get("voxceleb") or {}
    use_vox = bool(vox_cfg.get("enabled", False))
    index_fraction = float(data_cfg.get("index_fraction", 1.0))
    shuffle_chunks = bool(data_cfg.get("shuffle_chunks", True))

    ds_main = PairWavStoiDataset(
        audio_dirs=audio_dirs,
        original_dir=data_cfg["original_dir"],
        waveform_preprocessor=pre,
        stoi_computer=stoi_computer,
        single_chunk_per_file=bool(data_cfg.get("single_chunk_per_file", False)),
        cache_stoi_path=cache_path,
        max_items=data_cfg.get("max_items"),
        stoi_cache_num_workers=data_cfg.get("stoi_cache_num_workers"),
        index_fraction=index_fraction,
        index_sample_seed=seed + 91231,
        shuffle_chunks=shuffle_chunks,
    )

    n_pairs = len(ds_main.pairs)
    n_chunks = len(ds_main)
    print(
        f"Готово (основной датасет): пар processed↔original={n_pairs}, всего чанков={n_chunks}, "
        f"эффективных воркеров кэша STOI={ds_main._effective_stoi_cache_workers()}"
    )
    if n_chunks == 0:
        print("ОШИБКА: основной датасет пуст — проверь пути и имена файлов (__name=...).")
        raise SystemExit(1)

    tr_r = float(data_cfg.get("train_ratio", 0.8))
    va_r = float(data_cfg.get("val_ratio", 0.1))
    te_r = float(data_cfg.get("test_ratio", 0.1))
    train_on_all = bool(data_cfg.get("train_on_all_data", True))
    val_holdout = float(data_cfg.get("val_holdout_fraction", 0.01))
    if val_holdout < 0 or val_holdout >= 1.0:
        print("ОШИБКА: val_holdout_fraction должен быть в [0, 1)")
        raise SystemExit(1)

    if train_on_all:
        n_all = len(ds_main)
        _, _, test_idx = ds_main.split_indices(tr_r, va_r, te_r, seed)
        test_idx = list(test_idx)
        if val_holdout <= 1e-12:
            train_idx = list(range(n_all))
            val_idx = []
        else:
            tr_part = 1.0 - val_holdout
            train_idx, val_idx = split_indices_two_way(n_all, tr_part, seed, seed_offset=48271)
    else:
        train_idx, val_idx, test_idx = ds_main.split_indices(tr_r, va_r, te_r, seed)

    t_cfg = cfg["training"]
    bs = int(t_cfg["batch_size"])
    nw = int(t_cfg.get("num_workers", 0))

    if use_vox:
        aug_root = Path(vox_cfg["augmented_root"])
        mirror_root = Path(vox_cfg["mirror_root"])
        train_sub = vox_cfg.get(
            "train_subdirs",
            ["train/noise", "train/reverb", "train/noise_reverb"],
        )
        test_sub = vox_cfg.get(
            "test_subdirs",
            ["test/noise", "test/reverb", "test/noise_reverb"],
        )
        vox_train_dirs = [aug_root / s for s in train_sub]
        vox_test_dirs = [aug_root / s for s in test_sub]
        print("-" * 72)
        print(
            "VoxCeleb (augmented): train/val = CMU-MOSEI + доли voxceleb_train; "
            "test только voxceleb_test/"
        )
        print(f"  augmented_root={aug_root}")
        print(f"  mirror_root (чистые)={mirror_root}")
        for d in vox_train_dirs + vox_test_dirs:
            tag = "OK" if d.exists() else "НЕТ"
            print(f"  [{tag}] {d}")

        ds_vox_tr = PairWavStoiDataset(
            audio_dirs=vox_train_dirs,
            original_dir=mirror_root,
            mirror_original_root=mirror_root,
            waveform_preprocessor=pre,
            stoi_computer=stoi_computer,
            single_chunk_per_file=bool(data_cfg.get("single_chunk_per_file", False)),
            cache_stoi_path=cache_path,
            max_items=vox_cfg.get("max_items"),
            stoi_cache_num_workers=data_cfg.get("stoi_cache_num_workers"),
            index_fraction=index_fraction,
            index_sample_seed=seed + 91232,
            shuffle_chunks=shuffle_chunks,
        )
        ds_vox_te = PairWavStoiDataset(
            audio_dirs=vox_test_dirs,
            original_dir=mirror_root,
            mirror_original_root=mirror_root,
            waveform_preprocessor=pre,
            stoi_computer=stoi_computer,
            single_chunk_per_file=bool(data_cfg.get("single_chunk_per_file", False)),
            cache_stoi_path=cache_path,
            max_items=vox_cfg.get("max_items_test"),
            stoi_cache_num_workers=data_cfg.get("stoi_cache_num_workers"),
            index_fraction=index_fraction,
            index_sample_seed=seed + 91233,
            shuffle_chunks=shuffle_chunks,
        )
        n_vox_tr, n_vox_te = len(ds_vox_tr), len(ds_vox_te)
        print(f"  Vox train чанков={n_vox_tr}, Vox test чанков={n_vox_te}")
        if n_vox_tr == 0:
            print("ОШИБКА: VoxCeleb train пуст — проверь augmented_root и train_subdirs.")
            raise SystemExit(1)
        if n_vox_te == 0:
            print("ОШИБКА: VoxCeleb test пуст — проверь test_subdirs.")
            raise SystemExit(1)

        vox_tr_r = float(vox_cfg.get("train_ratio", 0.9))
        vox_va_r = float(vox_cfg.get("val_ratio", 0.1))
        if not train_on_all and abs(vox_tr_r + vox_va_r - 1.0) > 1e-4:
            print("ОШИБКА: voxceleb.train_ratio + voxceleb.val_ratio должны суммироваться в 1.0")
            raise SystemExit(1)

        if train_on_all:
            if val_holdout <= 1e-12:
                vox_train_idx = list(range(n_vox_tr))
                vox_val_idx = []
            else:
                tr_part = 1.0 - val_holdout
                vox_train_idx, vox_val_idx = split_indices_two_way(
                    n_vox_tr, tr_part, seed, seed_offset=58477
                )
        else:
            vox_train_idx, vox_val_idx = split_indices_two_way(n_vox_tr, vox_tr_r, seed)

        train_ds: Any = ConcatDataset(
            [
                subset_by_indices(ds_main, train_idx),
                subset_by_indices(ds_vox_tr, vox_train_idx),
            ]
        )
        val_ds = ConcatDataset(
            [
                subset_by_indices(ds_main, val_idx),
                subset_by_indices(ds_vox_tr, vox_val_idx),
            ]
        )
        test_ds = ds_vox_te
        n_test_examples = n_vox_te

        print("-" * 72)
        print("Разбиение (режим VoxCeleb)")
        if train_on_all:
            print(
                f"  train_on_all_data=true: train ≈ (1−val_holdout) чанков CMU+Vox train; "
                f"val_holdout_fraction={val_holdout}"
            )
            print(
                f"  train = CMU ({len(train_idx)}) + Vox train ({len(vox_train_idx)}) "
                f"→ всего {len(train_ds)} чанков (метки из существующего STOI-кэша)"
            )
            print(
                f"  val   = CMU ({len(val_idx)}) + Vox train val ({len(vox_val_idx)}) "
                f"→ всего {len(val_ds)} чанков"
            )
            print(
                f"  test (экспорт): только VoxCeleb test ({n_test_examples} чанков); "
                f"доля MOSEI test={te_r} от shuffle для test_predictions"
            )
        else:
            print(f"  CMU-MOSEI: доли train/val/test={tr_r}/{va_r}/{te_r} (hold-out MOSEI test не используется)")
            print(f"  Vox train (augmented): доля в train/val={vox_tr_r}/{vox_va_r} от {n_vox_tr} чанков")
            print(
                f"  train = CMU train ({len(train_idx)}) + Vox train ({len(vox_train_idx)}) "
                f"→ всего {len(train_ds)} чанков"
            )
            print(
                f"  val   = CMU val ({len(val_idx)}) + Vox val ({len(vox_val_idx)}) "
                f"→ всего {len(val_ds)} чанков"
            )
            print(f"  test  = только VoxCeleb test ({n_test_examples} чанков)")
    else:
        train_ds = subset_by_indices(ds_main, train_idx)
        val_ds = subset_by_indices(ds_main, val_idx)
        test_ds = subset_by_indices(ds_main, test_idx)
        n_test_examples = len(test_idx)

        print("-" * 72)
        print("Разбиение")
        if train_on_all:
            print(
                f"  train_on_all_data=true: val_holdout_fraction={val_holdout} "
                f"(val без пересчёта STOI)"
            )
            print(f"  train={len(train_idx)}, val={len(val_idx)}, test (экспорт)={len(test_idx)}")
        else:
            print(f"  доли train/val/test={tr_r}/{va_r}/{te_r}")
            print(f"  train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_frac = float(data_cfg.get("train_data_fraction", 1.0))
    try:
        train_ds, n_train_before, n_train_total = subsample_train_dataset(train_ds, train_frac, seed)
    except ValueError as e:
        print(f"ОШИБКА: {e}")
        raise SystemExit(1) from e
    if train_frac < 1.0 - 1e-12:
        print("-" * 72)
        print(
            f"Доля данных для обучения: train_data_fraction={train_frac} "
            f"→ чанков train {n_train_before} → {n_train_total} (val/test без изменений)"
        )

    train_loader_gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        generator=train_loader_gen,
        num_workers=nw,
        pin_memory=device.type == "cuda",
        collate_fn=collate_stoi_batch,
    )
    val_loader: Optional[DataLoader] = None
    if len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=device.type == "cuda",
            collate_fn=collate_stoi_batch,
        )
    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=device.type == "cuda",
        collate_fn=collate_stoi_batch,
    )

    n_val_total = len(val_ds)
    n_train_batches = max(1, (n_train_total + bs - 1) // bs)
    n_val_batches = (n_val_total + bs - 1) // bs if n_val_total > 0 else 0
    val_b_note = str(n_val_batches) if n_val_batches > 0 else "0 (нет val)"
    print(f"  DataLoader: batch_size={bs}, num_workers={nw}, батчей train≈{n_train_batches}, val≈{val_b_note}")

    m_cfg = cfg["model"]
    model = build_model(m_cfg["name"], m_cfg.get("kwargs") or {})
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("-" * 72)
    print("Модель и обучение")
    print(f"  архитектура: {m_cfg['name']}")
    print(f"  параметров: {n_params:,} (обучаемых: {n_trainable:,})")

    opt_cfg = t_cfg.get("optimizer", {})
    optimizer = make_optimizer(str(opt_cfg.get("type", "adam")), model, opt_cfg)
    criterion = build_criterion(str(t_cfg.get("loss", "torchaudio_stoi")))

    epochs = int(t_cfg["epochs"])
    grad_clip = float(t_cfg.get("grad_clip", 1.0))
    use_amp = bool(t_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    print(f"  оптимизатор: {opt_cfg.get('type', 'adam')}, lr={opt_cfg.get('lr')}, weight_decay={opt_cfg.get('weight_decay', 0)}")
    print(f"  loss: {t_cfg.get('loss', 'torchaudio_stoi')}")
    print(f"  эпох: {epochs}, AMP (mixed precision): {use_amp}, grad_clip={grad_clip}")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    log_dir = t_cfg.get("log_dir")
    writer = SummaryWriter(log_dir) if log_dir else None
    best_note = "лучший val" if val_loader is not None else "лучший train loss (val выключен)"
    print(f"  чекпоинты: {best_path} ({best_note}), {last_path} (последний)")
    if log_dir:
        print(f"  TensorBoard: {log_dir}")
    else:
        print("  TensorBoard: выключен (log_dir=null)")

    print("=" * 72)
    print("Старт обучения")
    print("=" * 72)

    best_val = float("inf")
    best_epoch = 0
    epoch_pbar = tqdm(
        range(1, epochs + 1),
        desc="Эпохи",
        unit="epoch",
        leave=True,
    )
    for epoch in epoch_pbar:
        tr_loss, tr_mae = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, grad_clip)
        if val_loader is not None:
            va_loss, va_mae, va_r = evaluate(model, val_loader, criterion, device)
            score_for_best = va_loss
            tqdm.write(
                f"Эпоха {epoch}/{epochs} | train loss={tr_loss:.4f} MAE={tr_mae:.4f} | "
                f"val loss={va_loss:.4f} MAE={va_mae:.4f} r={va_r:.4f}"
            )
        else:
            va_loss = va_mae = va_r = float("nan")
            score_for_best = tr_loss
            tqdm.write(
                f"Эпоха {epoch}/{epochs} | train loss={tr_loss:.4f} MAE={tr_mae:.4f} | val пропущен (все данные в train)"
            )
        if writer:
            writer.add_scalar("loss/train", tr_loss, epoch)
            writer.add_scalar("mae/train", tr_mae, epoch)
            if val_loader is not None:
                writer.add_scalar("loss/val", va_loss, epoch)
                writer.add_scalar("mae/val", va_mae, epoch)
                writer.add_scalar("corr/val", va_r, epoch)
        if score_for_best < best_val:
            best_val = score_for_best
            best_epoch = epoch
            torch.save({"model": m_cfg, "epoch": epoch, "state_dict": model.state_dict()}, best_path)
            tag = "val_loss" if val_loader is not None else "train_loss"
            tqdm.write(f"  → новый лучший {tag}={best_val:.4f}, сохранён {best_path.name}")
        torch.save({"model": m_cfg, "epoch": epoch, "state_dict": model.state_dict()}, last_path)
        if val_loader is not None:
            epoch_pbar.set_postfix(
                tr_loss=f"{tr_loss:.4f}",
                val_loss=f"{va_loss:.4f}",
                val_mae=f"{va_mae:.3f}",
                best_val=f"{best_val:.4f}" if best_val < float("inf") else "—",
            )
        else:
            epoch_pbar.set_postfix(
                tr_loss=f"{tr_loss:.4f}",
                best_train=f"{best_val:.4f}" if best_val < float("inf") else "—",
            )

    if writer:
        writer.close()

    print("=" * 72)
    print("Обучение завершено")
    best_label = "val_loss" if val_loader is not None else "train_loss (без val)"
    print(f"  лучший {best_label}={best_val:.4f} (эпоха {best_epoch})")
    print(f"  веса: {best_path.name}, {last_path.name}")
    print("=" * 72)

    export_cfg = cfg.get("export_test_predictions") or {}
    do_export = args.export_test_predictions or bool(export_cfg.get("enabled", False))
    out_pred = args.test_predictions_path or export_cfg.get("path")
    if do_export:
        if not out_pred:
            out_pred = str(out_dir / "test_predictions.json")
        print("-" * 72)
        print("Экспорт предсказаний на тестовой выборке")
        print(f"  загрузка лучшего чекпоинта: {best_path}")
        try:
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"  тестовых примеров: {n_test_examples}, инференс...")
        rows = predict_loader(model, test_loader, device)
        payload = {
            "format_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "stoi_target_config": stoi_cfg,
            "entries": rows,
        }
        Path(out_pred).parent.mkdir(parents=True, exist_ok=True)
        with open(out_pred, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  записано строк: {len(rows)} → {out_pred}")
    else:
        print("Экспорт предсказаний на тест не запрошен (--export-test-predictions или export_test_predictions.enabled).")

    print("Готово.")


if __name__ == "__main__":
    main()
