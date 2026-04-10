#!/usr/bin/env python3
"""
Оценка STOI с помощью модели по 5-секундным отрезкам.
Режимы: --use_vad (по умолчанию) — VAD по эталону, в модель подаются только речевые участки
        (склеиваются в одну дорожку, затем нарезка 5 с с шагом). Без флага — нарезка всего аудио.

Запуск (из корня проекта или из vew_some_wav):
  python vew_some_wav/model_stoi_sliding_window.py --checkpoint checkpoints_cnn_final/best_cnn_final.pt
  python vew_some_wav/model_stoi_sliding_window.py --no-use_vad   # без VAD, как раньше
"""
import os
import sys
import json
import argparse

# Корень проекта для src.model
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal as scipy_signal
from scipy.signal import resample
from tqdm import tqdm

from src.model import CNNSTOIPredictor
from exercises_blank import energy_gmm_vad, gauss_pdf

# Пути
PATH_TO_WAV_FOLDER = "/home/danya/datasets/speech_thesisis/"
REFERENCE_NAME = "Рина_Эталон.wav"
SYNCED_DIR = os.path.join(PATH_TO_WAV_FOLDER, "synced_1sec_chunks")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "continuous_speech_output")
# Длина чанка модели: 5 сек @ 16 kHz
MODEL_SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 5.0
MODEL_CHUNK_SAMPLES = int(CHUNK_DURATION_SEC * MODEL_SAMPLE_RATE)


def vad_to_segment_indices(vad_markup, thr=0.5):
    """VAD-разметка -> список (start, end) в отсчётах."""
    segments = []
    in_speech = False
    for i in range(len(vad_markup)):
        if vad_markup[i] > thr:
            if not in_speech:
                start = i
                in_speech = True
        else:
            if in_speech:
                segments.append((start, i))
                in_speech = False
    if in_speech:
        segments.append((start, len(vad_markup)))
    return segments


def extract_speech_only(signal_np, segment_indices):
    """Склеивает только речевые участки по segment_indices в один массив."""
    if not segment_indices:
        return np.array([], dtype=np.float32)
    parts = [signal_np[s:e] for (s, e) in segment_indices]
    return np.concatenate(parts).astype(np.float32)


def load_wav_normalized(path):
    """Загрузка WAV в float32 [-1, 1], пик-нормализация как torchaudio.load(normalize=True)."""
    sr, data = wavfile.read(path)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32) / np.iinfo(data.dtype).max
    # Привести к пик-нормализации: max abs = 1 (как при обучении в датасете)
    peak = np.max(np.abs(data)) + 1e-8
    data = data / peak
    return data, sr


def load_ref_and_test_from_synced():
    if not os.path.isdir(SYNCED_DIR):
        raise FileNotFoundError(
            f"Папка синхронизированных файлов не найдена: {SYNCED_DIR}. "
            "Сначала выполните ячейки ноутбука make_dataset.ipynb."
        )
    synced_files = [f for f in os.listdir(SYNCED_DIR) if f.endswith("_synced.wav")]
    ref_name = None
    for f in synced_files:
        if "Рина_Эталон" in f or REFERENCE_NAME.replace(".wav", "") in f:
            ref_name = f
            break
    if not ref_name:
        raise FileNotFoundError(f"В {SYNCED_DIR} не найден эталон (Рина_Эталон*_synced.wav).")
    ref_path = os.path.join(SYNCED_DIR, ref_name)
    ref_signal, ref_sr = load_wav_normalized(ref_path)
    tests = []
    for f in synced_files:
        if f == ref_name or "Эталон" in f:
            continue
        path = os.path.join(SYNCED_DIR, f)
        sig, sr = load_wav_normalized(path)
        if sr != ref_sr:
            sig = resample(sig, int(len(sig) * ref_sr / sr)).astype(np.float32)
        tests.append((sig, f))
    return (ref_signal, ref_sr), tests, ref_name


def load_model(checkpoint_path, device, model_name="cnn_hyperopt"):
    """Загрузка модели из чекпоинта (логика как в test_models.py)."""
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_kwargs = checkpoint.get("model_kwargs", {})
    if not model_kwargs and "cnn" in model_name.lower():
        hyperparams_file = os.path.join(checkpoint_dir, "best_hyperparameters.json")
        if os.path.exists(hyperparams_file):
            with open(hyperparams_file, "r") as f:
                hyperparams_data = json.load(f)
                model_kwargs = hyperparams_data["hyperparameters"]["model_kwargs"]
    if not model_kwargs:
        model_kwargs = {}
    model = CNNSTOIPredictor(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def slice_audio_5s_chunks(signal_np, sr, chunk_sec=CHUNK_DURATION_SEC, step_sec=1.0):
    """
    Нарезает аудио на чанки по chunk_sec секунд с шагом step_sec.
    Возвращает список чанков (каждый float32, длина = chunk_sec * sr в отсчётах после ресемплинга).
    Последний чанк дополняется нулями, если короче.
    После ресемплинга выполняется пик-нормализация (scipy.resample меняет масштаб амплитуды).
    """
    target_sr = MODEL_SAMPLE_RATE
    target_len = MODEL_CHUNK_SAMPLES
    if sr != target_sr:
        signal_np = scipy_signal.resample(
            signal_np, int(len(signal_np) * target_sr / sr)
        ).astype(np.float32)
        sr = target_sr
    # Ресемплинг в scipy меняет масштаб амплитуды; пик-нормализация как при обучении (датасет)
    peak = np.max(np.abs(signal_np)) + 1e-8
    signal_np = signal_np / peak
    duration_sec = len(signal_np) / sr
    if duration_sec < chunk_sec:
        chunk = np.zeros(target_len, dtype=np.float32)
        chunk[: len(signal_np)] = signal_np
        return [chunk]
    step_samples = int(step_sec * sr)
    chunk_samples = int(chunk_sec * sr)
    chunks = []
    start = 0
    while start + chunk_samples <= len(signal_np):
        piece = signal_np[start : start + chunk_samples]
        chunks.append(piece.astype(np.float32))
        start += step_samples
    # последний неполный чанк — с паддингом
    if start < len(signal_np):
        piece = signal_np[start:]
        if len(piece) < target_len:
            padded = np.zeros(target_len, dtype=np.float32)
            padded[: len(piece)] = piece
            chunks.append(padded)
        else:
            chunks.append(piece[:target_len].astype(np.float32))
    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="STOI модель по 5с отрезкам (нарезка без VAD)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints_cnn_final", "best_cnn_final.pt"),
        help="Путь к чекпоинту модели (CNN)",
    )
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--output_name", type=str, default="model_stoi_sliding_window.png")
    parser.add_argument(
        "--chunk_step_sec",
        type=float,
        default=1.0,
        help="Шаг нарезки в секундах (длина окна 5 с)",
    )
    parser.add_argument(
        "--use_vad",
        action="store_true",
        default=True,
        help="Использовать VAD: в модель подавать только речевые участки (по умолчанию)",
    )
    parser.add_argument(
        "--no-use_vad",
        action="store_false",
        dest="use_vad",
        help="Не использовать VAD: нарезать всё аудио на 5 с",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Чекпоинт не найден: {args.checkpoint}")
        print("Укажите --checkpoint путь к best_cnn_final.pt (или аналог).")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Данные: эталон + тестовые файлы (полные синхронизированные треки)
    (ref_signal, ref_sr), tests, ref_name = load_ref_and_test_from_synced()
    ref_signal = ref_signal.astype(np.float32)
    test_signals = {}
    for sig, nm in tests:
        test_signals[nm] = sig.astype(np.float32)

    # VAD по эталону (если use_vad): сегменты используем для всех файлов (синхронны)
    segment_indices = None
    if args.use_vad:
        vad_markup = energy_gmm_vad(
            signal=ref_signal,
            window=320,
            shift=160,
            gauss_pdf=gauss_pdf,
            n_realignment=10,
            vad_thr=0.5,
            mask_size_morph_filt=5,
        )
        segment_indices = vad_to_segment_indices(vad_markup, thr=0.5)
        speech_duration_sec = sum(e - s for s, e in segment_indices) / ref_sr
        print(f"VAD: речевых сегментов {len(segment_indices)}, суммарная длина речи {speech_duration_sec:.2f} с")

    # Загрузка модели
    model = load_model(args.checkpoint, args.device)
    print(f"Длительность чанка в модель: {CHUNK_DURATION_SEC} с ({MODEL_CHUNK_SAMPLES} отсчётов @ {MODEL_SAMPLE_RATE} Hz)")

    # По каждому файлу: при use_vad — только речь -> 5с чанки; иначе — всё аудио -> 5с чанки
    stoi_model_windows = {}
    synced_files = sorted([f for f in os.listdir(SYNCED_DIR) if f.endswith("_synced.wav")])

    for synced_name in tqdm(synced_files, desc="Модель STOI по 5с отрезкам"):
        if ref_name and synced_name == ref_name:
            sig = ref_signal
            if args.use_vad and segment_indices:
                sig = extract_speech_only(sig, segment_indices)
            duration_sec = len(sig) / ref_sr
            print(f"  {synced_name}: длина {duration_sec:.2f} с ({len(sig)} отсч.) перед нарезкой в чанки")
            chunks = slice_audio_5s_chunks(sig, ref_sr, step_sec=args.chunk_step_sec)
            stoi_model_windows[synced_name] = [1.0] * len(chunks)
            continue

        test_signal = test_signals.get(synced_name)
        if test_signal is None:
            continue

        if args.use_vad and segment_indices:
            sig = extract_speech_only(test_signal, segment_indices)
            if len(sig) == 0:
                stoi_model_windows[synced_name] = []
                continue
        else:
            sig = test_signal

        duration_sec = len(sig) / ref_sr
        print(f"  {synced_name}: длина {duration_sec:.2f} с ({len(sig)} отсч.) перед нарезкой в чанки")
        chunks = slice_audio_5s_chunks(
            sig, ref_sr,
            chunk_sec=CHUNK_DURATION_SEC,
            step_sec=args.chunk_step_sec,
        )
        window_preds = []
        with torch.no_grad():
            for chunk_np in chunks:
                if len(chunk_np) != MODEL_CHUNK_SAMPLES:
                    chunk_np = np.pad(
                        chunk_np,
                        (0, max(0, MODEL_CHUNK_SAMPLES - len(chunk_np))),
                        mode="constant",
                        constant_values=0,
                    )[:MODEL_CHUNK_SAMPLES]
                wav_tensor = torch.from_numpy(chunk_np).float().unsqueeze(0).to(args.device)
                pred = model(wav_tensor)
                pred_val = max(0.0, min(1.0, pred.item()))
                window_preds.append(pred_val)
        stoi_model_windows[synced_name] = window_preds

    n_chunks = max(len(v) for v in stoi_model_windows.values()) if stoi_model_windows else 0
    mode = "только речь (VAD)" if args.use_vad else "всё аудио"
    print(f"Отрезков по 5 с (шаг {args.chunk_step_sec} с), режим: {mode}: {n_chunks}")

    # График: ось X — номер 5с отрезка, ось Y — STOI (модель)
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, vals in stoi_model_windows.items():
        if not vals:
            continue
        if ref_name and name == ref_name:
            continue
        x = np.arange(len(vals))
        label = name.replace("_synced.wav", "")
        ax.plot(x, vals, "o-", label=label, alpha=0.8, markersize=4)
    ax.set_xlabel(f"Номер отрезка (5 с, шаг {args.chunk_step_sec} с)")
    ax.set_ylabel("STOI (модель)")
    title_suffix = "только речевые участки (VAD)" if args.use_vad else "без VAD"
    ax.set_title(f"STOI по 5с отрезкам — предсказание модели (CNN), {title_suffix}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    base_name = args.output_name
    if args.use_vad and not base_name.endswith("_vad.png"):
        base_name = base_name.replace(".png", "_vad.png")
    out_path = os.path.join(args.output_dir, base_name)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Сохранён график: {out_path}")

    # Сводка по файлам
    print(f"\n--- STOI (модель) по 5с отрезкам ({title_suffix}): среднее по файлам ---")
    all_vals = []
    for name in sorted(
        stoi_model_windows.keys(),
        key=lambda n: np.mean(stoi_model_windows[n] or [0]),
        reverse=True,
    ):
        vals = stoi_model_windows[name]
        if not vals:
            print(f"  {name}: N/A")
            continue
        mean_val = np.mean(vals)
        all_vals.extend(vals)
        short = name.replace("_synced.wav", "")
        print(f"  {short}: ср. {mean_val:.4f} (мин {min(vals):.4f}, макс {max(vals):.4f})")
    if all_vals:
        print(f"\nОбщее среднее STOI (модель) по всем отрезкам и файлам: {np.mean(all_vals):.4f}")


if __name__ == "__main__":
    main()
