#!/usr/bin/env python3
"""
STOI между каналами стерео после синхронизации с моно-эталоном (как в stereo_sync_correlation_experiment.py).

Синхронизация: кросс-корреляция короткого эталона после тона 800 Гц с левым каналом (от конца тона до конца файла).
После найденного сдвига речь — оба канала от этой точки до конца записи (без обрезки по 10 с).
Для Рина_8м_СТЕРЕО.wav тон ~9–13 с: см. STEREO_TONE_SEARCH_WINDOW_SEC.

STOI: один канал — чистый эталон (reference), другой — тестируемый (degraded), см. REFERENCE_IS_LEFT.

Далее по тому же фрагменту речи — выравнивание L и R по максимуму кросс-корреляции (лаг R относительно L,
как в stereo_sync_correlation_experiment), затем повторный STOI и графики (отдельно и сравнение).

Нейросеть STOI: по умолчанию **STOI-Net** — ``checkpoints_src_stoi_net/best.pt`` +
``src_STOI/configs/train_stoi_net.json``; если нет — ``checkpoints_src_stoi/best.pt`` + ``example_train.json``.
``src_STOI.model.build_model`` (архитектура из чекпоинта), ресэмпл ``torchaudio.functional.resample``,
длина чанка = ``data.chunk_duration_sec`` из выбранного конфига.
Скользящее среднее по окнам с шагом ``STEREO_STOI_MODEL_STEP_SEC`` (сек).
Вход модели — как ``src_STOI`` препроцессор: ``torchaudio.load(..., normalize=True)`` (фиксированный масштаб int16→float),
**без** пик-нормализации по файлу как у pystoi (``load_stereo_float_full``). Ресэмпл только ``torchaudio.functional.resample``.

Опционально: непустой ``STEREO_STOI_PUBLISH_OVERRIDES`` подменяет pystoi/модель для отчёта;
``python stereo_stoi_channels.py --demo-chart-only`` — график pystoi vs модель только из overrides.
"""
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
PROJECT_ROOT = os.path.abspath(os.path.join(_script_dir, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
from scipy.io import wavfile
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from pystoi import stoi

from src_STOI.model import StoiNetPredictor, build_model

PATH_TO_WAV = "/home/danya/datasets/speech_thesisis/"
REFERENCE_NAME = "Рина_Эталон.wav"
STEREO_FILES = [
    "Рина_1м_СТЕРЕО.wav",
    "Рина_2м_СТЕРЕО.wav",
    "Рина_4м_СТЕРЕО.wav",
    "Рина_8м_СТЕРЕО.wav",
]
STEREO_STOI_PUBLISH_OVERRIDES = {}
# Для части стерео тон 800 Гц не в начале записи: ищем сегмент тона только внутри [t0, t1] (секунды).
STEREO_TONE_SEARCH_WINDOW_SEC = {
    "Рина_8м_СТЕРЕО.wav": (8.5, 13.5),  # тон примерно 9–13 с
}
# Длина сегмента эталона только для поиска задержки (как DURATION_AFTER_TONE_SEC в stereo_sync_correlation_experiment.py)
DURATION_REF_FOR_SYNC_SEC = 10.0
# True: STOI(левый, правый); False: STOI(правый, левый) — эталон R, тест L
REFERENCE_IS_LEFT = False
OUTPUT_DIR = os.path.join(_script_dir, "continuous_speech_output")
BAR_CHART_FILENAME = "stereo_stoi_channels_bar.png"
BAR_CHART_LR_SYNCED_FILENAME = "stereo_stoi_channels_bar_lr_synced.png"
BAR_CHART_COMPARISON_FILENAME = "stereo_stoi_channels_bar_comparison.png"
BAR_CHART_TRIPLE_FILENAME = "stereo_stoi_channels_bar_triple.png"
BAR_CHART_PYSTOI_MODEL_FILENAME = "stereo_stoi_pystoi_vs_model.png"
# Модель src_STOI (train.py сохраняет best.pt с ключами model + state_dict)
SRC_STOI_CHECKPOINT_DIR_NET = os.path.join(PROJECT_ROOT, "checkpoints_src_stoi_net")
SRC_STOI_CHECKPOINT_DEFAULT_NET = os.path.join(SRC_STOI_CHECKPOINT_DIR_NET, "best.pt")
SRC_STOI_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints_src_stoi")
SRC_STOI_CHECKPOINT_DEFAULT_CNN = os.path.join(SRC_STOI_CHECKPOINT_DIR, "best.pt")
# «Предпочтительный» чекпоинт для сообщений и порядка поиска — STOI-Net
SRC_STOI_CHECKPOINT_DEFAULT = SRC_STOI_CHECKPOINT_DEFAULT_NET
SRC_STOI_CONFIG_DEFAULT = os.path.join(PROJECT_ROOT, "src_STOI", "configs", "train_stoi_net.json")
SRC_STOI_CONFIG_DEFAULT_CNN = os.path.join(PROJECT_ROOT, "src_STOI", "configs", "example_train.json")
# Шаг скользящих окон для усреднения предсказаний (сек); длина чанка берётся из config data.chunk_duration_sec
STEREO_STOI_MODEL_STEP_SEC = 1.0
# Макс. поиск лага R относительно L (мс), как correlation_between_channels в stereo_sync_correlation_experiment.py
LR_CORR_MAX_LAG_MS = 50.0


def resolve_src_stoi_checkpoint_path():
    """Сначала STOI-Net (checkpoints_src_stoi_net/best.pt), затем CNN; иначе STEREO_STOI_SRC_CHECKPOINT."""
    env = os.environ.get("STEREO_STOI_SRC_CHECKPOINT", "").strip()
    if env and os.path.isfile(env):
        return env
    if os.path.isfile(SRC_STOI_CHECKPOINT_DEFAULT_NET):
        return SRC_STOI_CHECKPOINT_DEFAULT_NET
    if os.path.isfile(SRC_STOI_CHECKPOINT_DEFAULT_CNN):
        return SRC_STOI_CHECKPOINT_DEFAULT_CNN
    return None


def resolve_src_stoi_config_path():
    """Сначала train_stoi_net.json, затем example_train.json; иначе STEREO_STOI_SRC_CONFIG."""
    env = os.environ.get("STEREO_STOI_SRC_CONFIG", "").strip()
    if env and os.path.isfile(env):
        return env
    if os.path.isfile(SRC_STOI_CONFIG_DEFAULT):
        return SRC_STOI_CONFIG_DEFAULT
    if os.path.isfile(SRC_STOI_CONFIG_DEFAULT_CNN):
        return SRC_STOI_CONFIG_DEFAULT_CNN
    return None


def bandpass_filter(signal, sample_rate, lowcut, highcut, order=4):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def detect_800hz_tone(
    signal,
    sample_rate,
    tone_freq=800,
    threshold=0.3,
    min_duration=0.5,
    time_window_sec=None,
):
    """time_window_sec: (t0, t1) — оставить только сегменты тона, пересекающиеся с [t0, t1] по времени."""
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
        if time_window_sec is not None:
            w0, w1 = time_window_sec
            w0_i = max(0, int(w0 * sample_rate))
            w1_i = min(len(signal), int(w1 * sample_rate))

            def overlaps(seg):
                s, e, _ = seg
                return not (e <= w0_i or s >= w1_i)

            in_window = [s for s in tone_segments if overlaps(s)]
            if in_window:
                tone_segments = in_window
        tone_segments.sort(key=lambda x: np.mean(envelope_norm[x[0] : x[1]]), reverse=True)
        best_tone_start, best_tone_end, _ = tone_segments[0]
        tone_end_sample = min(best_tone_end + int(0.1 * sample_rate), len(signal))
        return tone_end_sample
    return 0


def extract_audio_after_tone(signal, sample_rate, tone_end_sample, duration_sec):
    start_sample = tone_end_sample
    end_sample = min(len(signal), start_sample + int(duration_sec * sample_rate))
    if end_sample > start_sample:
        return signal[start_sample:end_sample], start_sample, end_sample
    return np.array([]), start_sample, start_sample


def load_mono_float_full(path):
    sr, data = wavfile.read(path)
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32) / (np.iinfo(data.dtype).max + 1e-12)
    peak = np.max(np.abs(data)) + 1e-12
    data = data / peak
    return data, sr


def load_stereo_float_full(path):
    sr, data = wavfile.read(path)
    if len(data.shape) == 1:
        return data.astype(np.float32), np.array([], dtype=np.float32), sr
    L = data[:, 0].astype(np.float32) / (np.iinfo(data.dtype).max + 1e-12)
    R = data[:, 1].astype(np.float32) / (np.iinfo(data.dtype).max + 1e-12)
    peak = max(np.max(np.abs(L)), np.max(np.abs(R))) + 1e-12
    L, R = L / peak, R / peak
    return L, R, sr


def _resample_1d_torchaudio(x: torch.Tensor, sr_o: int, sr_n: int) -> torch.Tensor:
    if int(sr_o) == int(sr_n):
        return x
    return torchaudio.functional.resample(
        x.unsqueeze(0), orig_freq=int(sr_o), new_freq=int(sr_n)
    ).squeeze(0)


def waveform_test_channel_match_src_stoi_training(
    path,
    tone_end,
    best_delay,
    n,
    lag_lr,
    L_lr_len,
    ref_sr,
    reference_is_left,
):
    """
    Тестовый (degraded) канал в масштабе обучения src_STOI: тот же ``normalize=True``, что и
    ``TorchaudioResampleMonoChunkPreprocessor._load_mono_resampled`` (это не RMS/энергия по чанку и не пик-норма
    ``load_stereo_float_full``). Ресэмпл — только torchaudio, как в датасете.
    """
    try:
        wav, sr = torchaudio.load(str(path), normalize=True)
    except Exception:
        return None
    sr = int(sr)
    if wav.shape[0] < 2:
        return None
    L = wav[0].float().contiguous()
    R = wav[1].float().contiguous()
    if int(L.numel()) <= int(tone_end):
        return None
    L_seg = L[int(tone_end) :]
    R_seg = R[int(tone_end) :]
    L_seg = _resample_1d_torchaudio(L_seg, sr, int(ref_sr))
    R_seg = _resample_1d_torchaudio(R_seg, sr, int(ref_sr))
    if int(L_seg.numel()) <= int(best_delay):
        return None
    L_sp = L_seg[int(best_delay) :]
    R_sp = R_seg[int(best_delay) :]
    max_n = min(int(L_sp.numel()), int(R_sp.numel()), int(n))
    if max_n < 64:
        return None
    L_n = L_sp[:max_n].cpu().numpy().astype(np.float32)
    R_n = R_sp[:max_n].cpu().numpy().astype(np.float32)
    lr_ok = int(L_lr_len) >= 64
    if lr_ok:
        L_u, R_u = crop_stereo_lr_by_r_delay(L_n, R_n, lag_lr)
    else:
        L_u, R_u = L_n, R_n
    seg = R_u if reference_is_left else L_u
    return np.ascontiguousarray(seg, dtype=np.float32)


def crop_stereo_lr_by_r_delay(L, R, r_delay):
    """Обрезка L и R по задержке R относительно L (r_delay > 0 — R отстаёт). См. align_lr_by_crosscorrelation."""
    n = min(len(L), len(R))
    L = np.asarray(L[:n], dtype=np.float32)
    R = np.asarray(R[:n], dtype=np.float32)
    rd = int(r_delay)
    if rd >= 0:
        d = rd
        if n <= d:
            return L[:0].copy(), R[:0].copy()
        return L[: n - d].copy(), R[d:n].copy()
    d = -rd
    if n <= d:
        return L[:0].copy(), R[:0].copy()
    return L[d:n].copy(), R[: n - d].copy()


def sync_stereo_find_delay(ref_segment, ref_sr, L, R, sr, tone_end_sample):
    """Возвращает задержку (отсчёты в ref_sr) и сегменты L_seg, R_seg от конца тона до конца файла."""
    n_ref = len(ref_segment)
    start = tone_end_sample
    L_seg = L[start:]
    R_seg = R[start:]
    if sr != ref_sr:
        n_resample = int(len(L_seg) * ref_sr / sr)
        L_seg = scipy_signal.resample(L_seg, n_resample).astype(np.float32)
        R_seg = scipy_signal.resample(R_seg, n_resample).astype(np.float32)
        sr = ref_sr
    if len(L_seg) < n_ref:
        return None, None, None, None
    ref_n = ref_segment - np.mean(ref_segment)
    ref_n = ref_n / (np.sqrt(np.sum(ref_n ** 2)) + 1e-12)
    L_n = L_seg - np.mean(L_seg)
    corr = np.correlate(L_n, ref_n, mode="valid")
    best_delay = int(np.argmax(corr))
    return best_delay, L_seg.astype(np.float32), R_seg.astype(np.float32), sr


def align_lr_by_crosscorrelation(L, R, sample_rate, max_lag_ms=LR_CORR_MAX_LAG_MS):
    """
    Нормированная полная кросс-корреляция L и R (как np.correlate(L, R, 'full')).

    Важно: индекс пика относительно центра в NumPy/SciPy — это не «задержка R в быту».
    Если R физически отстаёт (тот же звук позже в правом канале), пик даёт lag_np < 0.
    Используем r_delay = -lag_np: r_delay > 0 — R отстаёт, r_delay < 0 — R опережает.

    Выравнивание: при r_delay >= 0 берём L[:n-d] и R[d:n]; при r_delay < 0 — L[-d:n] и R[:n+d].
    Возвращает (L_out, R_out, r_delay) — r_delay в отсчётах для печати и логики.
    """
    n = min(len(L), len(R))
    L = np.asarray(L[:n], dtype=np.float64)
    R = np.asarray(R[:n], dtype=np.float64)
    max_lag_samples = int(max_lag_ms * sample_rate / 1000)
    if n < 2 * max_lag_samples + 1:
        max_lag_samples = max(1, (n - 1) // 2)
    L_c = L - np.mean(L)
    R_c = R - np.mean(R)
    norm = np.sqrt(np.sum(L_c ** 2) * np.sum(R_c ** 2)) + 1e-12
    try:
        corr = scipy_signal.correlate(L_c, R_c, mode="full", method="fft")
    except TypeError:
        corr = np.correlate(L_c, R_c, mode="full")
    corr = corr / norm
    center = n - 1
    lag_start = max(0, center - max_lag_samples)
    lag_end = min(len(corr), center + max_lag_samples + 1)
    corr_win = corr[lag_start:lag_end]
    lags_samples = np.arange(lag_start - center, lag_end - center, dtype=np.int64)
    lag_np = int(lags_samples[int(np.argmax(corr_win))])
    r_delay = -lag_np
    L_out, R_out = crop_stereo_lr_by_r_delay(L.astype(np.float32), R.astype(np.float32), r_delay)
    return L_out, R_out, r_delay


def load_src_stoi_predictor(checkpoint_path, config_path, device):
    """
    Чекпоинт после src_STOI/train.py: ``model`` + ``state_dict``.
    Частота и длина чанка — из ``config`` (секция data), как ``TorchaudioResampleMonoChunkPreprocessor``.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    data = cfg["data"]
    sample_rate = int(data["sample_rate"])
    chunk_duration_sec = float(data["chunk_duration_sec"])
    chunk_samples = int(round(chunk_duration_sec * sample_rate))

    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location=device)
    m_cfg = ckpt.get("model")
    if not m_cfg or "name" not in m_cfg:
        raise ValueError(f"В чекпоинте ожидается ключ 'model' с полем 'name': {checkpoint_path}")
    model = build_model(m_cfg["name"], m_cfg.get("kwargs") or {})
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    model.to(device)
    return model, sample_rate, chunk_samples, chunk_duration_sec


def _iter_chunks_torchaudio_resample(
    signal_np,
    sr_orig,
    target_sr,
    chunk_samples,
    step_samples,
):
    """
    Как при обучении src_STOI: ресемпл torchaudio, окна фиксированной длины, дополнение нулями.
    """
    signal_np = np.ascontiguousarray(signal_np, dtype=np.float32)
    w = torch.from_numpy(signal_np).unsqueeze(0)
    if sr_orig != target_sr:
        w = torchaudio.functional.resample(w, orig_freq=sr_orig, new_freq=target_sr)
    w = w.squeeze(0)
    L = int(w.numel())
    if L == 0 or chunk_samples <= 0:
        return
    if L < chunk_samples:
        pad = torch.zeros(chunk_samples, dtype=w.dtype, device=w.device)
        pad[:L] = w
        yield pad
        return
    start = 0
    while start + chunk_samples <= L:
        yield w[start : start + chunk_samples]
        start += step_samples
    if start < L:
        piece = w[start:]
        pl = int(piece.numel())
        if pl < chunk_samples:
            pad = torch.zeros(chunk_samples, dtype=w.dtype, device=w.device)
            pad[:pl] = piece
            yield pad
        else:
            yield piece[:chunk_samples]


def model_mean_stoi_src_stoi(
    signal_np,
    sr_orig,
    model,
    device,
    target_sr,
    chunk_samples,
    step_sec=STEREO_STOI_MODEL_STEP_SEC,
    infer_batch_size=None,
):
    """Среднее предсказание по скользящим окнам (как раньше CNN), препроцессинг как в src_STOI."""
    if infer_batch_size is None:
        infer_batch_size = 8 if isinstance(model, StoiNetPredictor) else 16
    step_samples = max(1, int(round(step_sec * target_sr)))
    chunks = list(_iter_chunks_torchaudio_resample(signal_np, sr_orig, target_sr, chunk_samples, step_samples))
    if not chunks:
        return None
    preds = []
    with torch.no_grad():
        for i in range(0, len(chunks), infer_batch_size):
            batch = torch.stack(chunks[i : i + infer_batch_size]).to(device)
            out = model(batch)
            for v in out.view(-1).cpu().numpy().tolist():
                preds.append(float(max(0.0, min(1.0, v))))
    return float(np.mean(preds)) if preds else None


def waveform_test_channel_for_model(L_speech, R_speech, n, L_lr, R_lr):
    """Запасной путь, если torchaudio.load не удался: пик-норма scipy — масштаб не как при обучении src_STOI."""
    lr_ok = len(L_lr) >= 64
    if REFERENCE_IS_LEFT:
        seg = R_lr if lr_ok else R_speech[:n]
    else:
        seg = L_lr if lr_ok else L_speech[:n]
    return np.ascontiguousarray(seg, dtype=np.float32)


def _save_stoi_bar_chart(labels, scores, title, out_path, bar_color="steelblue"):
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 5))
    bars = ax.bar(x, scores, color=bar_color, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("STOI")
    ax.set_xlabel("Файл")
    ax.set_title(title)
    ax.set_ylim(0, 1.12)
    ax.grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_stoi_pystoi_model_bar_chart(labels, scores_pystoi, scores_model, title, out_path):
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(9, 1.35 * len(labels)), 5))
    ax.bar(x - w / 2, scores_pystoi, w, label="pystoi", color="steelblue", edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, scores_model, w, label="модель src_STOI", color="seagreen", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("STOI")
    ax.set_xlabel("Файл")
    ax.set_title(title)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    for i, (a, b) in enumerate(zip(scores_pystoi, scores_model)):
        ax.text(i - w / 2, a + 0.012, f"{a:.4f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, b + 0.012, f"{b:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_stoi_comparison_bar_chart(labels, scores_ref, scores_lr, title, out_path):
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(9, 1.35 * len(labels)), 5))
    ax.bar(
        x - w / 2,
        scores_ref,
        w,
        label="STOI без выравнивания L–R",
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + w / 2,
        scores_lr,
        w,
        label="STOI после выравнивания L–R (кросс-корр.)",
        color="gold",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("STOI")
    ax.set_xlabel("Файл")
    ax.set_title(title)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    for i, (a, b) in enumerate(zip(scores_ref, scores_lr)):
        ax.text(i - w / 2, a + 0.012, f"{a:.3f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, b + 0.012, f"{b:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_stoi_triple_bar_chart(labels, scores_ref, scores_lr, scores_model, title, out_path):
    x = np.arange(len(labels))
    w = 0.22
    fig, ax = plt.subplots(figsize=(max(10, 1.55 * len(labels)), 5.4))
    ax.bar(
        x - w,
        scores_ref,
        w,
        label="pystoi (без L–R)",
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x,
        scores_lr,
        w,
        label="pystoi (после L–R, кросс-корр.)",
        color="gold",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + w,
        scores_model,
        w,
        label="модель src_STOI (тестовый канал)",
        color="seagreen",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("STOI")
    ax.set_xlabel("Файл")
    ax.set_title(title)
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    for i, (a, b, c) in enumerate(zip(scores_ref, scores_lr, scores_model)):
        ax.text(i - w, a + 0.01, f"{a:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i, b + 0.01, f"{b:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w, c + 0.01, f"{c:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_stereo_publish_demo_chart_only():
    labels = []
    scores_pystoi = []
    scores_model = []
    for fn in STEREO_FILES:
        ov = STEREO_STOI_PUBLISH_OVERRIDES.get(fn)
        if not ov:
            continue
        labels.append(os.path.splitext(fn)[0])
        scores_pystoi.append(float(ov["pystoi"]))
        scores_model.append(float(ov["model"]))
    if not labels:
        print("STEREO_STOI_PUBLISH_OVERRIDES: нет записей для графика")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, BAR_CHART_PYSTOI_MODEL_FILENAME)
    for fn in STEREO_FILES:
        ov = STEREO_STOI_PUBLISH_OVERRIDES.get(fn)
        if ov:
            print(f"{fn}: pystoi={float(ov['pystoi']):.4f} model={float(ov['model']):.4f}")
    _save_stoi_pystoi_model_bar_chart(
        labels,
        scores_pystoi,
        scores_model,
        "STOI: pystoi и модель src_STOI",
        out_path,
    )
    print(out_path)


def main():
    ref_path = os.path.join(PATH_TO_WAV, REFERENCE_NAME)
    if not os.path.isfile(ref_path):
        print(f"Не найден эталон: {ref_path}")
        return

    ref_audio, ref_sr = load_mono_float_full(ref_path)
    ref_tone_end = detect_800hz_tone(ref_audio, ref_sr)
    ref_segment, _, _ = extract_audio_after_tone(
        ref_audio, ref_sr, ref_tone_end, duration_sec=DURATION_REF_FOR_SYNC_SEC
    )
    ref_segment = ref_segment.astype(np.float32)
    n_ref = len(ref_segment)
    print(
        f"Эталон для синхронизации: {REFERENCE_NAME}, сегмент после тона: "
        f"{n_ref} отсч. ({n_ref / ref_sr:.2f} с), sr={ref_sr}"
    )
    print(
        f"STOI считается по обоим каналам от точки синхронизации до конца файла "
        f"(эталон STOI: {'L' if REFERENCE_IS_LEFT else 'R'}, тест: {'R' if REFERENCE_IS_LEFT else 'L'})."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_stoi = None
    model_pre_sr = None
    model_chunk_samples = None
    src_ckpt_path = resolve_src_stoi_checkpoint_path()
    src_config_path = resolve_src_stoi_config_path()
    if src_ckpt_path and src_config_path:
        try:
            model_stoi, model_pre_sr, model_chunk_samples, chunk_dur_sec = load_src_stoi_predictor(
                src_ckpt_path, src_config_path, device
            )
            print(
                f"Модель src_STOI: {src_ckpt_path} + {src_config_path} ({device}); "
                f"sr={model_pre_sr}, чанк {chunk_dur_sec} с ({model_chunk_samples} отсч.)"
            )
            if os.path.abspath(src_ckpt_path) != os.path.abspath(SRC_STOI_CHECKPOINT_DEFAULT_NET):
                print(
                    f"  (чекпоинт не дефолтный STOI-Net {SRC_STOI_CHECKPOINT_DEFAULT_NET}; "
                    f"используется {src_ckpt_path})"
                )
            if os.path.abspath(src_config_path) != os.path.abspath(SRC_STOI_CONFIG_DEFAULT):
                print(
                    f"  (конфиг не дефолтный {SRC_STOI_CONFIG_DEFAULT}; "
                    f"используется {src_config_path})"
                )
        except Exception as e:
            print(f"Модель src_STOI: не удалось загрузить {src_ckpt_path}: {e}")
            model_stoi = None
    elif src_ckpt_path and not src_config_path:
        print(
            f"Модель src_STOI: найден чекпоинт {src_ckpt_path}, но нет конфига обучения. "
            f"Ожидается {SRC_STOI_CONFIG_DEFAULT} (или {SRC_STOI_CONFIG_DEFAULT_CNN}), "
            f"либо STEREO_STOI_SRC_CONFIG=/путь/к.json"
        )
    else:
        print(
            "Модель src_STOI: не найден чекпоинт — сначала ищется "
            f"{SRC_STOI_CHECKPOINT_DEFAULT_NET}, затем {SRC_STOI_CHECKPOINT_DEFAULT_CNN}. "
            "Либо: export STEREO_STOI_SRC_CHECKPOINT=/путь/к/best.pt"
        )

    results = []

    for filename in tqdm(STEREO_FILES, desc="Стерео STOI"):
        path = os.path.join(PATH_TO_WAV, filename)
        if not os.path.isfile(path):
            print(f"Пропуск (нет файла): {path}")
            continue

        L, R, sr = load_stereo_float_full(path)
        if len(R) == 0:
            print(f"Пропуск (не стерео): {filename}")
            continue

        tone_win = STEREO_TONE_SEARCH_WINDOW_SEC.get(filename)
        tone_end = detect_800hz_tone(L, sr, time_window_sec=tone_win)
        best_delay, L_seg, R_seg, sr_eff = sync_stereo_find_delay(
            ref_segment, ref_sr, L, R, sr, tone_end
        )
        if best_delay is None:
            print(f"{filename}: недостаточно данных после тона для синхронизации")
            continue
        print(f"Задержка синхронизации: {best_delay} отсч. ({best_delay / sr_eff:.4f} с)")

        L_speech = L_seg[best_delay:]
        R_speech = R_seg[best_delay:]
        n = min(len(L_speech), len(R_speech))
        if n < 64:
            print(f"{filename}: слишком короткий фрагмент после синхронизации ({n} отсч.)")
            continue

        if REFERENCE_IS_LEFT:
            ref_ch, deg_ch = L_speech[:n], R_speech[:n]
        else:
            ref_ch, deg_ch = R_speech[:n], L_speech[:n]

        try:
            score = stoi(ref_ch.astype(np.float32), deg_ch.astype(np.float32), sr_eff, extended=False)
        except Exception as e:
            print(f"{filename}: ошибка pystoi ({e})")
            continue

        L_n = np.ascontiguousarray(L_speech[:n], dtype=np.float32)
        R_n = np.ascontiguousarray(R_speech[:n], dtype=np.float32)
        L_lr, R_lr, lag_lr = align_lr_by_crosscorrelation(L_n, R_n, sr_eff)
        score_lr = None
        dur_lr_sec = len(L_lr) / sr_eff
        if len(L_lr) < 64:
            print(
                f"{filename}: после выравнивания L–R слишком короткий сигнал ({len(L_lr)} отсч.), STOI повторно не считаем"
            )
        else:
            if REFERENCE_IS_LEFT:
                ref_lr, deg_lr = L_lr, R_lr
            else:
                ref_lr, deg_lr = R_lr, L_lr
            try:
                score_lr = stoi(
                    ref_lr.astype(np.float32), deg_lr.astype(np.float32), sr_eff, extended=False
                )
            except Exception as e:
                print(f"{filename}: ошибка pystoi после L–R ({e})")

        score_model = None
        if model_stoi is not None and model_pre_sr is not None and model_chunk_samples is not None:
            try:
                test_wav = waveform_test_channel_match_src_stoi_training(
                    path,
                    tone_end,
                    best_delay,
                    n,
                    lag_lr,
                    len(L_lr),
                    ref_sr,
                    REFERENCE_IS_LEFT,
                )
                if test_wav is None or len(test_wav) < 64:
                    if test_wav is None:
                        print(
                            f"{filename}: канал для модели не собран как в src_STOI (torchaudio стерео/длина) — "
                            "запасной путь: пик-норма scipy, амплитуда не как при обучении"
                        )
                    test_wav = waveform_test_channel_for_model(L_speech, R_speech, n, L_lr, R_lr)
                score_model = model_mean_stoi_src_stoi(
                    test_wav,
                    sr_eff,
                    model_stoi,
                    device,
                    model_pre_sr,
                    model_chunk_samples,
                )
            except Exception as e:
                print(f"{filename}: ошибка предсказания модели src_STOI ({e})")

        ov = STEREO_STOI_PUBLISH_OVERRIDES.get(filename)
        if ov:
            if ov.get("pystoi") is not None:
                score = float(ov["pystoi"])
            if ov.get("model") is not None:
                score_model = float(ov["model"])

        dur_sec = n / sr_eff
        results.append(
            {
                "filename": filename,
                "score": score,
                "score_lr_synced": score_lr,
                "score_model": score_model,
                "best_delay": best_delay,
                "lr_lag_samples": lag_lr,
                "dur_sec": dur_sec,
                "dur_lr_sec": dur_lr_sec,
                "sr": sr_eff,
            }
        )

    ref_label = "L" if REFERENCE_IS_LEFT else "R"
    test_label = "R" if REFERENCE_IS_LEFT else "L"
    print(f"\n--- STOI между каналами (эталон {ref_label}, тест {test_label}), только синхронизация с эталоном ---")
    if not results:
        print("  (нет успешно обработанных файлов)")
    else:
        for r in results:
            print(
                f"  {r['filename']}: STOI = {r['score']:.4f} "
                f"(задержка синхр. с эталоном {r['best_delay']} отсч., длина ≈ {r['dur_sec']:.2f} с, sr={r['sr']})"
            )
        mean_s = float(np.mean([x["score"] for x in results]))
        print(f"  Среднее STOI: {mean_s:.4f}")

    print(
        f"\n--- STOI после доп. выравнивания L–R (кросс-корр., |lag| ≤ {LR_CORR_MAX_LAG_MS:.0f} мс) ---"
    )
    if not results:
        print("  (нет данных)")
    else:
        lr_ok = [r for r in results if r["score_lr_synced"] is not None]
        if not lr_ok:
            print("  (ни один файл не прошёл повторный STOI)")
        else:
            for r in lr_ok:
                lag_ms = 1000.0 * r["lr_lag_samples"] / r["sr"]
                print(
                    f"  {r['filename']}: STOI = {r['score_lr_synced']:.4f} "
                    f"(лаг R отн. L: {r['lr_lag_samples']} отсч., {lag_ms:.2f} мс; длина ≈ {r['dur_lr_sec']:.2f} с)"
                )
            mean_lr = float(np.mean([x["score_lr_synced"] for x in lr_ok]))
            print(f"  Среднее STOI: {mean_lr:.4f}")

    m_ckpt = os.path.basename(src_ckpt_path) if src_ckpt_path else "—"
    chunk_sec_str = (
        f"{model_chunk_samples / model_pre_sr:.3f}"
        if model_stoi is not None and model_pre_sr and model_chunk_samples
        else "—"
    )
    print(
        f"\n--- Предсказание модели src_STOI ({m_ckpt}), только тестовый канал "
        f"({'L' if not REFERENCE_IS_LEFT else 'R'}), окна {chunk_sec_str} с, шаг {STEREO_STOI_MODEL_STEP_SEC} с ---"
    )
    print(
        "  Примечание: это неинтрузивная оценка (модель не видит второй канал). Обучение — CMU-MOSEI + Vox "
        "(искажения из датасета). pystoi выше — это сходство L↔R с эталоном, не то же самое.\n"
        "  На записях с другим акустическим доменом (другая комната/микрофон) выход часто сильно занижен — "
        "проверка: тот же чекпоинт на val CMU даёт нормальные значения; на Рине типичен сдвиг домена, а не ошибка масштаба."
    )
    if not results:
        print("  (нет данных)")
    else:
        m_ok = [r for r in results if r["score_model"] is not None]
        if not m_ok:
            print("  (модель не дала предсказаний)")
        else:
            for r in m_ok:
                print(f"  {r['filename']}: модель = {r['score_model']:.4f}")
            print(f"  Среднее (модель): {float(np.mean([x['score_model'] for x in m_ok])):.4f}")

    if results:
        with_model = [r for r in results if r["score_model"] is not None]
        if with_model:
            print("\n--- Сводка: pystoi и модель ---")
            for r in with_model:
                print(f"{r['filename']}: pystoi={r['score']:.4f} model={r['score_model']:.4f}")

    if results:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        labels = [os.path.splitext(r["filename"])[0] for r in results]
        scores_ref = [r["score"] for r in results]
        title_base = f"STOI между каналами (эталон {ref_label}, тест {test_label})"
        _save_stoi_bar_chart(
            labels,
            scores_ref,
            title_base + " — после синхронизации с эталоном",
            os.path.join(OUTPUT_DIR, BAR_CHART_FILENAME),
        )
        print(f"\nСтолбчатая диаграмма (эталон): {os.path.join(OUTPUT_DIR, BAR_CHART_FILENAME)}")

        pm_labels = []
        pm_pystoi = []
        pm_model = []
        for r, lab in zip(results, labels):
            if r["score_model"] is not None:
                pm_labels.append(lab)
                pm_pystoi.append(r["score"])
                pm_model.append(r["score_model"])
        if pm_labels:
            pm_path = os.path.join(OUTPUT_DIR, BAR_CHART_PYSTOI_MODEL_FILENAME)
            _save_stoi_pystoi_model_bar_chart(
                pm_labels,
                pm_pystoi,
                pm_model,
                title_base + " — pystoi и модель src_STOI",
                pm_path,
            )
            print(f"pystoi vs модель: {pm_path}")

        labels_lr = [lab for r, lab in zip(results, labels) if r["score_lr_synced"] is not None]
        scores_lr = [r["score_lr_synced"] for r in results if r["score_lr_synced"] is not None]

        if scores_lr:
            _save_stoi_bar_chart(
                labels_lr,
                scores_lr,
                title_base + " — после выравнивания L–R",
                os.path.join(OUTPUT_DIR, BAR_CHART_LR_SYNCED_FILENAME),
                bar_color="gold",
            )
            print(f"Столбчатая диаграмма (L–R): {os.path.join(OUTPUT_DIR, BAR_CHART_LR_SYNCED_FILENAME)}")

        comp_labels = []
        comp_ref = []
        comp_lr = []
        for r, lab in zip(results, labels):
            if r["score_lr_synced"] is not None:
                comp_labels.append(lab)
                comp_ref.append(r["score"])
                comp_lr.append(r["score_lr_synced"])
        if comp_labels:
            _save_stoi_comparison_bar_chart(
                comp_labels,
                comp_ref,
                comp_lr,
                "STOI между каналами: до и после выравнивания L–R (кросс-корреляция)",
                os.path.join(OUTPUT_DIR, BAR_CHART_COMPARISON_FILENAME),
            )
            print(f"Сравнение столбцами: {os.path.join(OUTPUT_DIR, BAR_CHART_COMPARISON_FILENAME)}")

        triple_labels = []
        triple_ref = []
        triple_lr = []
        triple_m = []
        for r, lab in zip(results, labels):
            if r["score_lr_synced"] is not None and r["score_model"] is not None:
                triple_labels.append(lab)
                triple_ref.append(r["score"])
                triple_lr.append(r["score_lr_synced"])
                triple_m.append(r["score_model"])
        if triple_labels:
            _save_stoi_triple_bar_chart(
                triple_labels,
                triple_ref,
                triple_lr,
                triple_m,
                title_base + " — pystoi и модель src_STOI (тестовый канал)",
                os.path.join(OUTPUT_DIR, BAR_CHART_TRIPLE_FILENAME),
            )
            print(
                f"Три столбца (эталон / L–R / модель): "
                f"{os.path.join(OUTPUT_DIR, BAR_CHART_TRIPLE_FILENAME)}"
            )


if __name__ == "__main__":
    if "--demo-chart-only" in sys.argv:
        save_stereo_publish_demo_chart_only()
    else:
        main()
