#!/usr/bin/env python3
"""
Эксперимент со стерео-файлами Рина:
1. Синхронизация с эталоном по максимуму кросс-корреляции (вместо поиска по STOI).
2. Для синхронизированных стерео — графики корреляции между каналами L и R.
3. В консоль: задержка (в отсчётах и секундах), на которой достигнут максимум корреляции между каналами.

Интервалы, на которых считается корреляция:
- Синхронизация с эталоном: эталон — сегмент DURATION_AFTER_TONE_SEC (10 с) после тона 800 Гц;
  кросс-корреляция этого эталона с левым каналом стерео (от конца тона до конца файла).
- Корреляция между каналами L–R: считается по синхронизированному фрагменту той же длительности
  (10 с), т.е. по целому выровненному отрезку после сдвига по эталону.
"""
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt
from tqdm import tqdm

PATH_TO_WAV = "/home/danya/datasets/speech_thesisis/"
REFERENCE_NAME = "Рина_Эталон.wav"
STEREO_FILES = [
    "Рина_1м_СТЕРЕО.wav",
    "Рина_2м_СТЕРЕО.wav",
    "Рина_4м_СТЕРЕО.wav",
    "Рина_8м_СТЕРЕО.wav",
]
OUTPUT_DIR = os.path.join(_script_dir, "continuous_speech_output")
# Длительность сегмента после тона 800 Гц (в секундах). На этом интервале берётся эталон и
# синхронизированные стерео-фрагменты; на нём же считается корреляция между каналами L и R.
DURATION_AFTER_TONE_SEC = 10.0


def bandpass_filter(signal, sample_rate, lowcut, highcut, order=4):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def detect_800hz_tone(signal, sample_rate, tone_freq=800, threshold=0.3, min_duration=0.5):
    filtered_signal = bandpass_filter(signal, sample_rate, tone_freq - 50, tone_freq + 50)
    envelope = np.abs(filtered_signal)
    smooth_window = int(0.02 * sample_rate)
    if smooth_window > 1:
        envelope_smooth = np.convolve(envelope, np.ones(smooth_window) / smooth_window, mode="same")
    else:
        envelope_smooth = envelope
    envelope_norm = envelope_smooth / (np.max(envelope_smooth) + 1e-12)
    tone_mask = envelope_norm > threshold
    tone_segments = []
    in_tone = False
    start_idx = 0
    for i in range(len(tone_mask)):
        if tone_mask[i] and not in_tone:
            in_tone = True
            start_idx = i
        elif not tone_mask[i] and in_tone:
            in_tone = False
            end_idx = i
            duration = (end_idx - start_idx) / sample_rate
            if duration >= min_duration:
                tone_segments.append((start_idx, end_idx, duration))
    if in_tone:
        duration = (len(tone_mask) - start_idx) / sample_rate
        if duration >= min_duration:
            tone_segments.append((start_idx, len(tone_mask), duration))
    if tone_segments:
        tone_segments.sort(key=lambda x: np.mean(envelope_norm[x[0] : x[1]]), reverse=True)
        best_tone_start, best_tone_end, _ = tone_segments[0]
        tone_end_sample = min(best_tone_end + int(0.1 * sample_rate), len(signal))
        return tone_end_sample
    return 0


def extract_audio_after_tone(signal, sample_rate, tone_end_sample, duration_sec=10.0):
    start_sample = tone_end_sample
    end_sample = min(len(signal), start_sample + int(duration_sec * sample_rate))
    if end_sample > start_sample:
        return signal[start_sample:end_sample], start_sample, end_sample
    return np.array([]), start_sample, start_sample


def load_mono_float(path, max_duration_sec=25):
    sr, data = wavfile.read(path)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32) / (np.iinfo(data.dtype).max + 1e-12)
    peak = np.max(np.abs(data)) + 1e-12
    data = data / peak
    n = min(len(data), int(max_duration_sec * sr))
    return data[:n], sr


def load_stereo_float(path, max_duration_sec=25):
    sr, data = wavfile.read(path)
    if len(data.shape) == 1:
        return data.astype(np.float32), np.array([], dtype=np.float32), sr
    L = data[:, 0].astype(np.float32) / (np.iinfo(data.dtype).max + 1e-12)
    R = data[:, 1].astype(np.float32) / (np.iinfo(data.dtype).max + 1e-12)
    n = min(len(L), int(max_duration_sec * sr))
    L, R = L[:n], R[:n]
    peak = max(np.max(np.abs(L)), np.max(np.abs(R))) + 1e-12
    L, R = L / peak, R / peak
    return L, R, sr


def sync_stereo_by_correlation(ref_segment, ref_sr, L, R, sr, tone_end_sample):
    """Синхронизация стерео с эталоном по максимуму кросс-корреляции (по каналу L).
    Эталон ref_segment имеет длительность DURATION_AFTER_TONE_SEC (10 с). Корреляция
    считается при сдвиге эталона по левому каналу теста (от конца тона до конца файла)."""
    n_ref = len(ref_segment)
    # Сегмент теста от тона до конца (достаточно длинный для корреляции)
    start = tone_end_sample
    L_seg = L[start:]
    R_seg = R[start:]
    if sr != ref_sr:
        n_resample = int(len(L_seg) * ref_sr / sr)
        L_seg = scipy_signal.resample(L_seg, n_resample).astype(np.float32)
        R_seg = scipy_signal.resample(R_seg, n_resample).astype(np.float32)
        sr = ref_sr
    if len(L_seg) < n_ref:
        return None, None, None
    # Нормированная кросс-корреляция: ref и L_seg
    ref_n = ref_segment - np.mean(ref_segment)
    ref_n = ref_n / (np.sqrt(np.sum(ref_n ** 2)) + 1e-12)
    L_n = (L_seg - np.mean(L_seg))
    corr = np.correlate(L_n, ref_n, mode="valid")
    best_delay = int(np.argmax(corr))
    # Синхронизированные каналы той же длины, что и эталон
    L_synced = L_seg[best_delay : best_delay + n_ref].astype(np.float32)
    R_synced = R_seg[best_delay : best_delay + n_ref].astype(np.float32)
    return L_synced, R_synced, best_delay


def correlation_between_channels(L, R, sr, max_lag_ms=50):
    """Кросс-корреляция между каналами L и R по всему переданному отрезку.
    Обычно L и R — синхронизированный фрагмент длительностью DURATION_AFTER_TONE_SEC (10 с).
    Возвращает массив корреляций и лагов в мс (диапазон лагов ±max_lag_ms)."""
    max_lag_samples = int(max_lag_ms * sr / 1000)
    n = len(L)
    if n < 2 * max_lag_samples:
        max_lag_samples = max(1, (n - 1) // 2)
    L_c = L - np.mean(L)
    R_c = R - np.mean(R)
    norm = np.sqrt(np.sum(L_c ** 2) * np.sum(R_c ** 2)) + 1e-12
    corr = np.correlate(L_c, R_c, mode="full") / norm
    # full: длина 2*n-1, центр (n-1) — лаг 0
    center = n - 1
    lag_start = center - max_lag_samples
    lag_end = center + max_lag_samples + 1
    lag_start = max(0, lag_start)
    lag_end = min(len(corr), lag_end)
    corr_win = corr[lag_start:lag_end]
    lags_samples = np.arange(lag_start - center, lag_end - center)
    lags_ms = lags_samples * 1000.0 / sr
    return corr_win, lags_ms, lags_samples


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Эталон: сегмент после тона 800 Гц
    ref_path = os.path.join(PATH_TO_WAV, REFERENCE_NAME)
    if not os.path.isfile(ref_path):
        print(f"Не найден эталон: {ref_path}")
        return
    ref_audio, ref_sr = load_mono_float(ref_path)
    ref_tone_end = detect_800hz_tone(ref_audio, ref_sr)
    ref_segment, _, _ = extract_audio_after_tone(
        ref_audio, ref_sr, ref_tone_end, duration_sec=DURATION_AFTER_TONE_SEC
    )
    ref_segment = ref_segment.astype(np.float32)
    n_ref = len(ref_segment)
    print(f"Эталон: {REFERENCE_NAME}, сегмент после тона: {n_ref} отсч. ({n_ref/ref_sr:.2f} с), ref_sr={ref_sr}")
    print(f"Корреляция с эталоном и корреляция L–R считаются на интервале длительностью {DURATION_AFTER_TONE_SEC} с.")

    results = []

    for filename in tqdm(STEREO_FILES, desc="Стерео файлы"):
        path = os.path.join(PATH_TO_WAV, filename)
        if not os.path.isfile(path):
            print(f"Пропуск (нет файла): {path}")
            continue

        L, R, sr = load_stereo_float(path)
        if len(R) == 0:
            print(f"Пропуск (не стерео): {filename}")
            continue

        # Тон по левому каналу
        tone_end = detect_800hz_tone(L, sr)
        L_synced, R_synced, delay_samples = sync_stereo_by_correlation(
            ref_segment, ref_sr, L, R, sr, tone_end
        )
        if L_synced is None:
            print(f"{filename}: недостаточно данных после тона для синхронизации")
            continue

        delay_sec = delay_samples / ref_sr
        print(f"\n--- {filename} ---")
        print(f"  Синхронизация с эталоном: задержка = {delay_samples} отсч. ({delay_sec:.4f} с)")

        # Корреляция между каналами L и R по синхронизированному фрагменту (длительность = DURATION_AFTER_TONE_SEC)
        max_lag_ms = 50
        corr_lr, lags_ms, lags_samples = correlation_between_channels(
            L_synced, R_synced, ref_sr, max_lag_ms=max_lag_ms
        )
        best_idx = np.argmax(corr_lr)
        best_lag_ms = lags_ms[best_idx]
        best_lag_samples = lags_samples[best_idx]
        best_lag_sec = best_lag_samples / ref_sr
        print(f"  Корреляция L–R: максимум при задержке правого канала = {best_lag_samples} отсч. ({best_lag_sec:.4f} с, {best_lag_ms:.2f} мс)")

        results.append({
            "filename": filename,
            "sync_delay_samples": delay_samples,
            "sync_delay_sec": delay_sec,
            "L": L_synced,
            "R": R_synced,
            "sr": ref_sr,
            "corr_lr": corr_lr,
            "lags_ms": lags_ms,
            "lags_samples": lags_samples,
            "best_lag_samples": best_lag_samples,
            "best_lag_sec": best_lag_sec,
            "best_lag_ms": best_lag_ms,
        })

        # График: корреляция между каналами от лага (мс)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(lags_ms, corr_lr, "b-", linewidth=2)
        ax.axvline(x=best_lag_ms, color="red", linestyle="--", label=f"Макс.: {best_lag_ms:.2f} мс")
        ax.axhline(y=corr_lr[best_idx], color="gray", linestyle=":", alpha=0.7)
        ax.set_xlabel("Задержка правого канала относительно левого (мс)")
        ax.set_ylabel("Корреляция L–R")
        ax.set_title(f"Корреляция между каналами (после синхронизации с эталоном)\n{filename}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_name = filename.replace(".wav", "_stereo_correlation.png")
        out_path = os.path.join(OUTPUT_DIR, out_name)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  График сохранён: {out_path}")

    print("\n--- Итог: задержка (макс. корреляция между каналами) ---")
    for r in results:
        print(f"  {r['filename']}: {r['best_lag_samples']} отсч. ({r['best_lag_sec']:.4f} с, {r['best_lag_ms']:.2f} мс)")


if __name__ == "__main__":
    main()
