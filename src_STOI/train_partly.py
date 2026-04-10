"""
Обучение с той же схемой данных, что и ``train.py``: CMU-MOSEI + опционально VoxCeleb,
``index_fraction``, ``train_data_fraction``, ``apply_silence_removal`` из ``stoi_target``.

Число полных проходов по train — ``training.epochs`` (по умолчанию 10). Каждая эпоха: train
делится на ~N участков; после каждого участка — валидация на Рине; best.pt — по MAE на Рине.

Порядок сигналов в батчах — через ``DataLoader(shuffle=True)``. В ``train_partly`` по умолчанию
``shuffle_chunks=false`` в data (без перемешивания списка чанков в датасете); задать
``"shuffle_chunks": true`` в JSON, если нужно как в ``train.py``.

Расширенный STOI-кэш (``data.stoi_rich_cache_path``): degraded/reference пути, STOI, флаги
``noise`` / ``reverb`` (по имени папки в пути). Подмешивание числовых меток из большого pickle:
``data.stoi_import_legacy_cache_paths`` (список путей), без пересчёта имеющихся ключей.

Запуск (из корня репозитория):
  python src_STOI/train_partly.py --config src_STOI/configs/train_stoi_net.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset
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
from src_STOI.train import (
    collate_stoi_batch,
    make_optimizer,
    set_seed,
    split_indices_two_way,
    subsample_train_dataset,
    train_one_epoch,
)


def _load_stereo_module():
    path = _REPO_ROOT / "vew_some_wav" / "stereo_stoi_channels.py"
    spec = importlib.util.spec_from_file_location("stereo_stoi_channels_rina", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Не удалось загрузить модуль: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def split_train_indices_into_segments(
    perm: Sequence[int], n_segments: int
) -> List[List[int]]:
    """Ровно n_segments блоков; длины отличаются не более чем на 1."""
    n = len(perm)
    if n == 0:
        return [[] for _ in range(n_segments)]
    base = n // n_segments
    rem = n % n_segments
    out: List[List[int]] = []
    off = 0
    for s in range(n_segments):
        ln = base + (1 if s < rem else 0)
        out.append(list(perm[off : off + ln]))
        off += ln
    return out


def evaluate_rina_vs_pystoi(
    model: nn.Module,
    device: torch.device,
    stereo: Any,
    wav_dir: Path,
    reference_name: str,
    stereo_files: Sequence[str],
    tone_windows: Dict[str, Tuple[float, float]],
    reference_is_left: bool,
    duration_ref_sync_sec: float,
    model_pre_sr: int,
    model_chunk_samples: int,
    step_sec: float,
) -> Tuple[float, float, List[Dict[str, float]]]:
    """
    Для каждого стерео: pystoi после L–R выравнивания (как в stereo_stoi_channels) и
    среднее предсказание модели по окнам на тестовом канале.
    Возвращает (MAE, Pearson r, список по файлам).
    """
    from pystoi import stoi

    ref_path = wav_dir / reference_name
    if not ref_path.is_file():
        raise FileNotFoundError(f"Эталон Рины: {ref_path}")

    ref_audio, ref_sr = stereo.load_mono_float_full(str(ref_path))
    ref_tone_end = stereo.detect_800hz_tone(ref_audio, ref_sr)
    ref_segment, _, _ = stereo.extract_audio_after_tone(
        ref_audio, ref_sr, ref_tone_end, duration_sec=duration_ref_sync_sec
    )
    ref_segment = ref_segment.astype(np.float32)

    model.eval()
    rows: List[Dict[str, float]] = []
    pystoi_vals: List[float] = []
    pred_vals: List[float] = []

    for filename in stereo_files:
        path = wav_dir / filename
        if not path.is_file():
            continue
        L, R, sr = stereo.load_stereo_float_full(str(path))
        if len(R) == 0:
            continue
        tone_win = tone_windows.get(filename)
        tone_end = stereo.detect_800hz_tone(L, sr, time_window_sec=tone_win)
        sync = stereo.sync_stereo_find_delay(ref_segment, ref_sr, L, R, sr, tone_end)
        if sync[0] is None:
            continue
        best_delay, L_seg, R_seg, sr_eff = sync
        L_speech = L_seg[best_delay:]
        R_speech = R_seg[best_delay:]
        n = min(len(L_speech), len(R_speech))
        if n < 64:
            continue
        L_lr, R_lr, lag_lr = stereo.align_lr_by_crosscorrelation(
            L_speech[:n], R_speech[:n], sr_eff, max_lag_ms=stereo.LR_CORR_MAX_LAG_MS
        )
        if len(L_lr) < 64:
            continue
        if reference_is_left:
            ref_lr, deg_lr = L_lr, R_lr
        else:
            ref_lr, deg_lr = R_lr, L_lr
        try:
            py = float(stoi(ref_lr.astype(np.float32), deg_lr.astype(np.float32), sr_eff, extended=False))
        except Exception:
            continue

        test_wav = stereo.waveform_test_channel_match_src_stoi_training(
            str(path),
            tone_end,
            best_delay,
            n,
            lag_lr,
            len(L_lr),
            ref_sr,
            reference_is_left,
        )
        if test_wav is None or len(test_wav) < 64:

            def _fallback_seg():
                lr_ok = len(L_lr) >= 64
                if reference_is_left:
                    seg = R_lr if lr_ok else R_speech[:n]
                else:
                    seg = L_lr if lr_ok else L_speech[:n]
                return np.ascontiguousarray(seg, dtype=np.float32)

            test_wav = _fallback_seg()

        pr = stereo.model_mean_stoi_src_stoi(
            test_wav,
            sr_eff,
            model,
            device,
            model_pre_sr,
            model_chunk_samples,
            step_sec=step_sec,
        )
        if pr is None:
            continue
        pr = float(max(0.0, min(1.0, pr)))
        pystoi_vals.append(py)
        pred_vals.append(pr)
        rows.append({"file": filename, "pystoi_lr": py, "model_mean": pr, "abs_err": abs(pr - py)})

    if not pystoi_vals:
        return float("inf"), float("nan"), []

    pv = np.asarray(pystoi_vals, dtype=np.float64)
    mv = np.asarray(pred_vals, dtype=np.float64)
    mae = float(np.mean(np.abs(mv - pv)))
    if len(pv) > 1:
        c = np.corrcoef(mv, pv)[0, 1]
        r = float(c) if not np.isnan(c) else float("nan")
    else:
        r = float("nan")
    return mae, r, rows


def _normalize_legacy_paths(data_cfg: Dict[str, Any]) -> List[Path]:
    raw = data_cfg.get("stoi_import_legacy_cache_paths")
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    return [Path(p) for p in raw]


def build_partly_train_dataset(cfg: Dict[str, Any], seed: int) -> Tuple[Any, Dict[str, Any]]:
    """Как ``train.py``: CMU + Vox (если enabled), сплиты, ``train_data_fraction``; rich/legacy кэш из data."""
    data_cfg = cfg["data"]
    stoi_cfg = cfg.get("stoi_target", {})
    sample_rate = int(data_cfg["sample_rate"])
    chunk_sec = float(data_cfg["chunk_duration_sec"])
    cache_path = data_cfg.get("cache_stoi_path")
    rich_path = data_cfg.get("stoi_rich_cache_path")
    legacy_paths = _normalize_legacy_paths(data_cfg)
    index_fraction = float(data_cfg.get("index_fraction", 1.0))
    # По умолчанию false: shuffle только в DataLoader (порядок сэмплов/батчей), не reorder chunk_index.
    shuffle_chunks = bool(data_cfg.get("shuffle_chunks", False))

    pre = TorchaudioResampleMonoChunkPreprocessor(sample_rate=sample_rate, chunk_duration_sec=chunk_sec)
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
    audio_dirs = [base / s for s in subdirs] if subdirs else [base]

    print("-" * 72)
    print("train_partly / данные (схема как train.py)")
    print(f"  processed: {base}  (подпапки: {subdirs if subdirs else '— вся база'})")
    print(f"  index_fraction={index_fraction}, train_data_fraction={float(data_cfg.get('train_data_fraction', 1.0))}")
    print(f"  shuffle_chunks={shuffle_chunks}")
    print(f"  apply_silence_removal={apply_silence_removal}")
    print(f"  float STOI-кэш: {cache_path or '—'}")
    print(f"  rich STOI-кэш: {rich_path or '—'}")
    print(f"  импорт legacy pickle: {legacy_paths or '—'}")
    print("Сбор датасетов и STOI-кэш (CMU, затем при необходимости Vox)...")

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
        rich_cache_path=rich_path,
        import_legacy_cache_paths=legacy_paths,
    )
    if len(ds_main) == 0:
        raise SystemExit("CMU-MOSEI датасет пуст — проверь data.audio_base_dir и subdirs")

    tr_r = float(data_cfg.get("train_ratio", 0.8))
    va_r = float(data_cfg.get("val_ratio", 0.1))
    te_r = float(data_cfg.get("test_ratio", 0.1))
    train_idx, val_idx, test_idx = ds_main.split_indices(tr_r, va_r, te_r, seed)

    vox_cfg = cfg.get("voxceleb") or {}
    use_vox = bool(vox_cfg.get("enabled", False))
    train_ds: Any

    if use_vox:
        aug_root = Path(vox_cfg["augmented_root"])
        mirror_root = Path(vox_cfg["mirror_root"])
        train_sub = vox_cfg.get(
            "train_subdirs",
            ["train/noise", "train/reverb", "train/noise_reverb"],
        )
        vox_train_dirs = [aug_root / s for s in train_sub]
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
            rich_cache_path=rich_path,
            import_legacy_cache_paths=legacy_paths,
        )
        if len(ds_vox_tr) == 0:
            raise SystemExit("VoxCeleb train пуст — проверь augmented_root и train_subdirs")
        vox_tr_r = float(vox_cfg.get("train_ratio", 0.9))
        vox_va_r = float(vox_cfg.get("val_ratio", 0.1))
        if abs(vox_tr_r + vox_va_r - 1.0) > 1e-4:
            raise SystemExit("voxceleb.train_ratio + voxceleb.val_ratio должны давать 1.0")
        vox_train_idx, _ = split_indices_two_way(len(ds_vox_tr), vox_tr_r, seed)
        train_ds = ConcatDataset(
            [
                subset_by_indices(ds_main, train_idx),
                subset_by_indices(ds_vox_tr, vox_train_idx),
            ]
        )
        print(
            f"  train чанков: CMU={len(train_idx)} + Vox={len(vox_train_idx)} → всего {len(train_ds)} "
            f"(val CMU={len(val_idx)}, test CMU={len(test_idx)} не в train_partly)"
        )
    else:
        train_ds = subset_by_indices(ds_main, train_idx)
        print(f"  train чанков: {len(train_ds)} (только CMU)")

    train_frac = float(data_cfg.get("train_data_fraction", 1.0))
    train_ds, n_before, n_after = subsample_train_dataset(train_ds, train_frac, seed)
    if train_frac < 1.0 - 1e-12:
        print(f"  после train_data_fraction={train_frac}: {n_before} → {n_after} чанков")

    meta = {
        "sample_rate": sample_rate,
        "chunk_samples": pre.chunk_samples,
        "chunk_sec": chunk_sec,
    }
    return train_ds, meta


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Обучение по участкам (training.epochs эпох) + выбор best по Рине"
    )
    parser.add_argument("--config", type=str, default=str(here / "configs" / "train_stoi_net.json"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Число эпох (по умолчанию training.epochs из конфига, обычно 10)",
    )
    parser.add_argument("--segments", type=int, default=30, help="Число участков за одну эпоху")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_REPO_ROOT / "checkpoints_src_stoi_partly"),
        help="Куда сохранять best.pt / last.pt",
    )
    parser.add_argument("--rina-wav-dir", type=str, default="/home/danya/datasets/speech_thesisis/")
    parser.add_argument("--rina-reference", type=str, default="Рина_Эталон.wav")
    parser.add_argument(
        "--rina-stereo",
        type=str,
        nargs="*",
        default=[
            "Рина_1м_СТЕРЕО.wav",
            "Рина_2м_СТЕРЕО.wav",
            "Рина_4м_СТЕРЕО.wav",
            "Рина_8м_СТЕРЕО.wav",
        ],
    )
    parser.add_argument(
        "--rina-reference-is-left",
        action="store_true",
        help="По умолчанию как stereo_stoi_channels: эталон R, тест L (REFERENCE_IS_LEFT=False)",
    )
    parser.add_argument("--rina-sync-ref-sec", type=float, default=10.0)
    parser.add_argument("--rina-model-step-sec", type=float, default=1.0)
    args = parser.parse_args()

    reference_is_left = bool(args.rina_reference_is_left)
    cfg = load_merged_config([args.config])
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device_s = args.device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_s)

    t_cfg = cfg["training"]
    bs = int(t_cfg.get("batch_size", 16))
    nw = int(t_cfg.get("num_workers", 0))
    grad_clip = float(t_cfg.get("grad_clip", 1.0))
    use_amp = bool(t_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    num_epochs = int(args.epochs) if args.epochs is not None else int(t_cfg.get("epochs", 10))
    num_epochs = max(1, num_epochs)

    train_ds, meta = build_partly_train_dataset(cfg, seed)
    n_train = len(train_ds)
    print(f"  train чанков: {n_train}")

    m_cfg = cfg["model"]
    model = build_model(m_cfg["name"], m_cfg.get("kwargs") or {})
    model.to(device)
    opt_cfg = t_cfg.get("optimizer", {})
    optimizer = make_optimizer(str(opt_cfg.get("type", "adam")), model, opt_cfg)
    criterion = build_criterion(str(t_cfg.get("loss", "torchaudio_stoi")))

    stereo = _load_stereo_module()
    wav_dir = Path(args.rina_wav_dir)
    tone_windows = dict(getattr(stereo, "STEREO_TONE_SEARCH_WINDOW_SEC", {}))

    n_seg = max(1, int(args.segments))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"

    best_mae = float("inf")
    best_r = float("-inf")
    best_seg = -1
    best_epoch = -1

    print("=" * 72)
    print(
        f"Устройство: {device}, эпох: {num_epochs}, участков/эпоха: {n_seg}, "
        f"shuffle батчей в DataLoader; валидация — Рина (MAE vs pystoi L–R)"
    )
    print(f"  выход: {out_dir}")
    print("=" * 72)

    epoch_pbar = tqdm(range(1, num_epochs + 1), desc="Эпохи", unit="epoch", leave=True)
    for epoch in epoch_pbar:
        g = torch.Generator().manual_seed(seed + epoch * 1_000_001)
        perm = torch.randperm(n_train, generator=g).tolist()
        segment_indices = split_train_indices_into_segments(perm, n_seg)
        epoch_pbar.set_postfix(epoch=f"{epoch}/{num_epochs}")

        for seg_i, idx_list in enumerate(
            tqdm(segment_indices, desc=f"  эп.{epoch} сегменты", unit="seg", leave=False)
        ):
            if not idx_list:
                tqdm.write(f"  эп.{epoch} сегмент {seg_i + 1}/{n_seg}: пусто — пропуск")
                continue
            subset = Subset(train_ds, idx_list)
            seg_gen = torch.Generator().manual_seed(seed + epoch * 1_000_003 + seg_i * 100_003)
            loader = DataLoader(
                subset,
                batch_size=bs,
                shuffle=True,
                generator=seg_gen,
                num_workers=nw,
                pin_memory=device.type == "cuda",
                collate_fn=collate_stoi_batch,
            )
            tr_loss, tr_mae = train_one_epoch(model, loader, criterion, optimizer, device, scaler, grad_clip)
            mae_r, r_r, rows = evaluate_rina_vs_pystoi(
                model,
                device,
                stereo,
                wav_dir,
                args.rina_reference,
                args.rina_stereo,
                tone_windows,
                reference_is_left,
                args.rina_sync_ref_sec,
                meta["sample_rate"],
                meta["chunk_samples"],
                args.rina_model_step_sec,
            )
            tqdm.write(
                f"  эп.{epoch}/{num_epochs} сегм.{seg_i + 1}/{n_seg}: train loss={tr_loss:.4f} train_MAE={tr_mae:.4f} | "
                f"Рина MAE(pred,pystoi)={mae_r:.4f} r={r_r:.4f} файлов={len(rows)}"
            )
            for row in rows:
                tqdm.write(f"    {row['file']}: pystoi={row['pystoi_lr']:.4f} model={row['model_mean']:.4f}")

            better = mae_r < best_mae - 1e-9 or (
                abs(mae_r - best_mae) <= 1e-9 and not np.isnan(r_r) and r_r > best_r
            )
            if better and np.isfinite(mae_r):
                best_mae = mae_r
                best_r = r_r if not np.isnan(r_r) else best_r
                best_seg = seg_i
                best_epoch = epoch
                torch.save(
                    {
                        "model": m_cfg,
                        "epoch": epoch,
                        "segment": seg_i,
                        "state_dict": model.state_dict(),
                        "rina_mae": mae_r,
                        "rina_r": r_r,
                        "rina_rows": rows,
                    },
                    best_path,
                )
                tqdm.write(
                    f"  → новый best по Рине: MAE={best_mae:.4f} (эп.{epoch}, сегм.{seg_i + 1}) → {best_path.name}"
                )

            torch.save(
                {
                    "model": m_cfg,
                    "epoch": epoch,
                    "segment": seg_i,
                    "state_dict": model.state_dict(),
                    "rina_mae": mae_r,
                    "rina_r": r_r,
                },
                last_path,
            )

    print("=" * 72)
    if best_seg >= 0 and best_epoch >= 0:
        print(
            f"Готово. Лучший чекпоинт по Рине: MAE={best_mae:.4f}, "
            f"эп.{best_epoch}/{num_epochs}, сегм.{best_seg + 1}/{n_seg} → {best_path}"
        )
    else:
        print("Готово. Рина-валидация не дала ни одного файла — best не обновлялся.")
    print(f"Последнее состояние: {last_path}")
    summary = {
        "config": str(Path(args.config).resolve()),
        "epochs": num_epochs,
        "segments_per_epoch": n_seg,
        "best_epoch": best_epoch if best_epoch >= 0 else None,
        "best_segment": best_seg if best_seg >= 0 else None,
        "best_rina_mae": best_mae if best_seg >= 0 else None,
        "output_dir": str(out_dir.resolve()),
    }
    with open(out_dir / "train_partly_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
