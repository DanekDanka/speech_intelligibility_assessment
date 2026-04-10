#!/usr/bin/env python3
"""
Эксперимент: влияние сдвига эталона на STOI для уже синхронизированных сигналов.
VAD как в ноутбуке, 7 сегментов, сдвиг с шагом 1. График — линия STOI(shift) для каждого файла.
В таблицу — среднее STOI по файлу.
Графики сохраняются в continuous_speech_output.
"""

import os
import sys

# Чтобы при запуске из корня проекта находился exercises_blank
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal as scipy_signal
from scipy.signal import resample
from pystoi import stoi

from exercises_blank import energy_gmm_vad, gauss_pdf

# Пути из ноутбука
PATH_TO_WAV_FOLDER = "/home/danya/datasets/speech_thesisis/"
REFERENCE_NAME = "Рина_Эталон.wav"
SYNCED_DIR = os.path.join(PATH_TO_WAV_FOLDER, "synced_1sec_chunks")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "continuous_speech_output")
# Сдвиги с шагом 1 для линейных графиков (как в ноутбуке)
SHIFT_STEP = 1
SHIFT_MAX = 512  # максимальный сдвиг (0, 1, 2, ..., 512)
N_SEGMENTS = 7  # число речевых сегментов по VAD
# Для старых графиков по фиксированным сдвигам
SHIFTS_SAMPLES = [0, 64, 128, 256, 512]
# Скользящее окно по речевым отрезкам (как в ячейке «STOI только по речевому»)
WINDOW_SIZE_SLIDING = 7
STEP_SLIDING = 1


def vad_to_segment_indices(vad_markup, thr=0.5):
    """VAD-разметка -> список (start, end) в отсчётах. Как в ноутбуке."""
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


def load_wav_normalized(path):
    """Загрузка WAV как float32, нормализация по максимуму."""
    sr, data = wavfile.read(path)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return data, sr


def load_ref_and_test_from_synced():
    """
    Загружает эталон и тестовые файлы из папки synced_1sec_chunks.
    Возвращает (ref_signal, ref_sr), список (test_signal, name).
    """
    if not os.path.isdir(SYNCED_DIR):
        raise FileNotFoundError(
            f"Папка синхронизированных файлов не найдена: {SYNCED_DIR}. "
            "Сначала выполните ячейки ноутбука make_dataset.ipynb для создания synced_1sec_chunks."
        )
    synced_files = [f for f in os.listdir(SYNCED_DIR) if f.endswith("_synced.wav")]
    ref_name = None
    for f in synced_files:
        if "Рина_Эталон" in f or REFERENCE_NAME.replace(".wav", "") in f:
            ref_name = f
            break
    if not ref_name:
        raise FileNotFoundError(
            f"В {SYNCED_DIR} не найден эталон (Рина_Эталон*_synced.wav)."
        )
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
    return (ref_signal, ref_sr), tests


def extract_speech_from_vad_segments(signal, segment_indices, n_segments=N_SEGMENTS):
    """Склеить первые n_segments речевых отрезков по VAD. Как в ноутбуке."""
    first_n = segment_indices[:n_segments]
    if not first_n:
        return np.array([], dtype=np.float32)
    parts = [signal[s:e] for (s, e) in first_n]
    return np.concatenate(parts).astype(np.float32)


def stoi_with_shift(ref, test, sr, shift_samples):
    """
    STOI при сдвиге эталона вперёд на shift_samples отсчётов.
    ref_shifted = ref[shift:], test_trimmed = test[:-shift], чтобы длины совпадали.
    """
    if shift_samples <= 0:
        n = min(len(ref), len(test))
        return stoi(ref[:n].astype(np.float32), test[:n].astype(np.float32), sr, extended=False)
    if shift_samples >= len(ref) or shift_samples >= len(test):
        return np.nan
    ref_shifted = ref[shift_samples:].astype(np.float32)
    test_trimmed = test[:-shift_samples].astype(np.float32)
    n = min(len(ref_shifted), len(test_trimmed))
    ref_shifted = ref_shifted[:n]
    test_trimmed = test_trimmed[:n]
    return stoi(ref_shifted, test_trimmed, sr, extended=False)


def run_experiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    (ref, sr), tests = load_ref_and_test_from_synced()

    # VAD по эталону (как в ноутбуке), первые N_SEGMENTS сегментов
    vad_markup = energy_gmm_vad(
        signal=ref,
        window=320,
        shift=160,
        gauss_pdf=gauss_pdf,
        n_realignment=10,
        vad_thr=0.5,
        mask_size_morph_filt=5,
    )
    segment_indices = vad_to_segment_indices(vad_markup, thr=0.5)
    if len(segment_indices) < N_SEGMENTS:
        raise ValueError(
            f"По VAD найдено только {len(segment_indices)} сегментов, нужно минимум {N_SEGMENTS}."
        )

    # ---------- STOI по скользящему окну для каждого сдвига эталона (0, 64, 128, 256, 512) ----------
    ref_signal = ref
    ref_sr = sr
    n_windows = max(0, len(segment_indices) - WINDOW_SIZE_SLIDING + 1)
    print(f"Речевых отрезков по VAD: {len(segment_indices)}, окон по {WINDOW_SIZE_SLIDING} с шагом {STEP_SLIDING}: {n_windows}")

    synced_files = sorted([f for f in os.listdir(SYNCED_DIR) if f.endswith("_synced.wav")])
    ref_name = next((f for f in synced_files if "Рина_Эталон" in f or REFERENCE_NAME.replace(".wav", "") in f), None)

    # Подготовка тестовых сигналов (длины как у ref)
    test_signals = {}
    for ts, nm in tests:
        sig = ts
        if len(sig) < len(ref_signal):
            sig = np.pad(sig, (0, len(ref_signal) - len(sig)), mode="constant")
        else:
            sig = sig[: len(ref_signal)]
        test_signals[nm] = sig

    for shift in SHIFTS_SAMPLES:
        stoi_all_windows = {}
        for synced_name in synced_files:
            if ref_name and synced_name == ref_name:
                if shift == 0:
                    stoi_all_windows[synced_name] = [1.0] * n_windows
                else:
                    window_stois = []
                    for i in range(n_windows):
                        window_segments = segment_indices[i : i + WINDOW_SIZE_SLIDING]
                        ref_parts = [ref_signal[s:e] for (s, e) in window_segments]
                        ref_speech_win = np.concatenate(ref_parts).astype(np.float32)
                        if len(ref_speech_win) <= shift:
                            break
                        stoi_val = stoi_with_shift(ref_speech_win, ref_speech_win.copy(), ref_sr, shift)
                        window_stois.append(stoi_val)
                    stoi_all_windows[synced_name] = window_stois if len(window_stois) == n_windows else None
                continue

            test_signal = test_signals.get(synced_name)
            if test_signal is None:
                continue

            window_stois = []
            for i in range(n_windows):
                window_segments = segment_indices[i : i + WINDOW_SIZE_SLIDING]
                ref_parts = [ref_signal[s:e] for (s, e) in window_segments]
                ref_speech_win = np.concatenate(ref_parts).astype(np.float32)

                test_parts = []
                for (s, e) in window_segments:
                    if e > len(test_signal):
                        break
                    chunk = test_signal[s:e]
                    n_ref = e - s
                    if len(chunk) != n_ref:
                        chunk = scipy_signal.resample(chunk, n_ref)
                    test_parts.append(chunk.astype(np.float32))
                if len(test_parts) < WINDOW_SIZE_SLIDING:
                    break
                test_speech_win = np.concatenate(test_parts)
                if len(test_speech_win) != len(ref_speech_win):
                    test_speech_win = scipy_signal.resample(test_speech_win, len(ref_speech_win)).astype(np.float32)
                stoi_val = stoi_with_shift(ref_speech_win, test_speech_win, ref_sr, shift)
                window_stois.append(stoi_val)

            stoi_all_windows[synced_name] = window_stois if len(window_stois) == n_windows else None

        # График для этого сдвига: все STOI по окнам (одна линия на файл), без эталона
        fig_win, ax_win = plt.subplots(figsize=(12, 5))
        for name, vals in stoi_all_windows.items():
            if vals is None or (ref_name and name == ref_name) or "Рина_Эталон" in name or "Эталон" in name:
                continue
            x = np.arange(len(vals))
            label = name.replace("_synced.wav", "")
            ax_win.plot(x, vals, "o-", label=label, alpha=0.8, markersize=4)
        ax_win.set_xlabel(f"Номер окна ({WINDOW_SIZE_SLIDING} сегментов, шаг {STEP_SLIDING})")
        ax_win.set_ylabel("STOI")
        ax_win.set_title(f"STOI по скользящему окну (сдвиг эталона {shift} отсч., {WINDOW_SIZE_SLIDING} речевых отрезков, шаг {STEP_SLIDING})")
        ax_win.legend(loc="best", fontsize=8)
        ax_win.grid(True, alpha=0.3)
        ax_win.set_ylim(0, 1.05)
        fig_win.tight_layout()
        path_win = os.path.join(OUTPUT_DIR, f"stoi_sliding_window_shift_{shift}.png")
        fig_win.savefig(path_win, dpi=150)
        if shift == 0:
            path_win_0 = os.path.join(OUTPUT_DIR, "stoi_sliding_window.png")
            fig_win.savefig(path_win_0, dpi=150)
            print(f"Сохранён график: {path_win_0}")
        plt.close(fig_win)
        print(f"Сохранён график: {path_win}")

        # Среднее по файлам для этого сдвига
        if shift == 0:
            print("\n--- STOI по речевым отрезкам (окно {}, шаг {}): среднее по файлам ---".format(WINDOW_SIZE_SLIDING, STEP_SLIDING))
            all_vals = []
            for name in sorted(stoi_all_windows.keys(), key=lambda n: np.mean(stoi_all_windows[n] or [0]), reverse=True):
                vals = stoi_all_windows[name]
                if vals is None:
                    print(f"  {name}: N/A")
                    continue
                mean_val = np.mean(vals)
                all_vals.extend(vals)
                short = name.replace("_synced.wav", "")
                print(f"  {short}: ср. {mean_val:.4f} (мин {min(vals):.4f}, макс {max(vals):.4f})")
            if all_vals:
                print(f"\nОбщее среднее STOI по всем окнам и файлам: {np.mean(all_vals):.4f}")

    # ---------- Далее: эксперимент со сдвигом эталона (7 сегментов, шаг 1) ----------
    ref_speech = extract_speech_from_vad_segments(ref, segment_indices, N_SEGMENTS)
    print(f"\nДлина ref_speech для сдвигов: {len(ref_speech)} отсч.")

    # Сдвиги с шагом 1: 0, 1, 2, ..., SHIFT_MAX
    shifts_line = list(range(0, SHIFT_MAX + 1, SHIFT_STEP))

    # Для каждого файла: кривая STOI(shift), среднее — в таблицу
    results_line = {}  # name -> [stoi при shift 0, 1, ..., 512]
    for test_sig, name in tests:
        test_speech = extract_speech_from_vad_segments(test_sig, segment_indices, N_SEGMENTS)
        if len(test_speech) != len(ref_speech):
            test_speech = resample(test_speech, len(ref_speech)).astype(np.float32)
        curve = []
        for sh in shifts_line:
            s = stoi_with_shift(ref_speech, test_speech, sr, sh)
            curve.append(s)
        results_line[name] = curve

    # Таблица: среднее STOI по файлу
    print("\nSTOI при сдвиге эталона с шагом 1 (VAD, 7 сегментов). Среднее по файлу:")
    file_means = {}
    for name in sorted(results_line.keys()):
        curve = results_line[name]
        m = np.nanmean(curve)
        file_means[name] = m
        short = name.replace("_synced.wav", "")
        print(f"  {short}: среднее STOI = {m:.4f}")

    # График 1: линия — как меняется STOI со сдвигом (среднее по файлам ± std)
    fig, ax = plt.subplots(figsize=(10, 5))
    mean_curve = np.nanmean([results_line[n] for n in results_line], axis=0)
    std_curve = np.nanstd([results_line[n] for n in results_line], axis=0)
    ax.plot(shifts_line, mean_curve, "b-", linewidth=2, label="Среднее по файлам")
    ax.fill_between(shifts_line, mean_curve - std_curve, mean_curve + std_curve, alpha=0.3)
    for name, curve in list(results_line.items())[:6]:
        ax.plot(shifts_line, curve, "-", alpha=0.6, label=name.replace("_synced.wav", ""))
    ax.set_xlabel("Сдвиг эталона (отсчёты)")
    ax.set_ylabel("STOI")
    ax.set_title("STOI от сдвига эталона (VAD, 7 сегментов, шаг сдвига 1)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "stoi_vs_reference_shift.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"Сохранён график: {path1}")

    # График 2: столбцы — среднее STOI по файлам при сдвигах 0, 64, 128, 256, 512
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    idx_at = [shifts_line.index(s) for s in SHIFTS_SAMPLES]
    mean_at = [mean_curve[i] for i in idx_at]
    x = np.arange(len(SHIFTS_SAMPLES))
    ax2.bar(x, mean_at, width=0.4, align="center", label="STOI (среднее)", color="steelblue")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in SHIFTS_SAMPLES])
    ax2.set_xlabel("Сдвиг эталона (отсчёты)")
    ax2.set_ylabel("STOI")
    ax2.set_title("Средний STOI при сдвиге эталона")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    fig2.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "stoi_reference_shift_bars.png")
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"Сохранён график: {path2}")

    # Удаляем старые графики по сдвигам (заменяем их графиками по файлам)
    for shift in SHIFTS_SAMPLES:
        old_path = os.path.join(OUTPUT_DIR, f"stoi_per_file_shift_{shift}.png")
        if os.path.isfile(old_path):
            os.remove(old_path)

    # График для каждого файла: линия STOI(shift)
    for name, curve in results_line.items():
        short_name = name.replace("_synced.wav", "")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(shifts_line, curve, "o-", markersize=2, alpha=0.8)
        ax.set_xlabel("Сдвиг эталона (отсчёты)")
        ax.set_ylabel("STOI")
        ax.set_title(f"STOI от сдвига эталона — {short_name} (VAD, 7 сегментов, шаг 1)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        m = file_means[name]
        ax.axhline(m, color="gray", linestyle="--", alpha=0.7, label=f"Среднее = {m:.4f}")
        ax.legend(loc="upper right")
        fig.tight_layout()
        safe_name = short_name.replace(" ", "_")
        path_file = os.path.join(OUTPUT_DIR, f"stoi_per_file_{safe_name}.png")
        fig.savefig(path_file, dpi=150)
        plt.close(fig)
        print(f"Сохранён график: {path_file}")

    # Таблица в файл: файл -> среднее STOI
    table_path = os.path.join(OUTPUT_DIR, "stoi_reference_shift_results.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("STOI при сдвиге эталона (VAD, 7 сегментов, шаг сдвига 1)\n")
        f.write("Среднее значение STOI для каждого файла:\n\n")
        for name in sorted(file_means.keys()):
            short = name.replace("_synced.wav", "")
            f.write(f"  {short}: {file_means[name]:.4f}\n")
    print(f"Сохранены результаты: {table_path}")

    return results_line


if __name__ == "__main__":
    run_experiment()
