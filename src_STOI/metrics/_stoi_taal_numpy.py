# SPDX-License-Identifier: MIT
# Short-Time Objective Intelligibility after Taal et al. (2010, 2011) and extended STOI (2016).
# Vendored and adapted from the reference STOI reference implementation pattern
# (same constants and structure as commonly used open implementations).

from __future__ import annotations

import functools
import warnings

import numpy as np
from scipy.signal import resample_poly

EPS = np.finfo(float).eps

FS = 10_000
N_FRAME = 256
NFFT = 512
NUMBAND = 15
MINFREQ = 150
N = 30
BETA = -15.0
DYN_RANGE = 40.0

OBM, CF = None, None  # filled on first use


def _ensure_obm():
    global OBM, CF
    if OBM is None:
        OBM, CF = thirdoct(FS, NFFT, NUMBAND, MINFREQ)


def _resample_window_oct(p, q):
    gcd = np.gcd(p, q)
    if gcd > 1:
        p /= gcd
        q /= gcd
    log10_rejection = -3.0
    stopband_cutoff_f = 1.0 / (2 * max(p, q))
    roll_off_width = stopband_cutoff_f / 10
    rejection_dB = -20 * log10_rejection
    L = np.ceil((rejection_dB - 8) / (28.714 * roll_off_width))
    t = np.arange(-L, L + 1)
    ideal_filter = 2 * p * stopband_cutoff_f * np.sinc(2 * stopband_cutoff_f * t)
    if rejection_dB >= 21 and rejection_dB <= 50:
        beta = 0.5842 * (rejection_dB - 21) ** 0.4 + 0.07886 * (rejection_dB - 21)
    elif rejection_dB > 50:
        beta = 0.1102 * (rejection_dB - 8.7)
    else:
        beta = 0.0
    h = np.kaiser(2 * L + 1, beta) * ideal_filter
    return h


def resample_oct(x, p, q):
    h = _resample_window_oct(p, q)
    window = h / np.sum(h)
    return resample_poly(x, int(p), int(q), window=window)


@functools.lru_cache(maxsize=None)
def thirdoct(fs, nfft, num_bands, min_freq):
    f = np.linspace(0, fs, nfft + 1)
    f = f[: int(nfft / 2) + 1]
    k = np.array(range(num_bands)).astype(float)
    cf = np.power(2.0 ** (1.0 / 3), k) * min_freq
    freq_low = min_freq * np.power(2.0, (2 * k - 1) / 6)
    freq_high = min_freq * np.power(2.0, (2 * k + 1) / 6)
    obm = np.zeros((num_bands, len(f)))
    for i in range(len(cf)):
        f_bin = np.argmin(np.square(f - freq_low[i]))
        freq_low[i] = f[f_bin]
        fl_ii = f_bin
        f_bin = np.argmin(np.square(f - freq_high[i]))
        freq_high[i] = f[f_bin]
        fh_ii = f_bin
        obm[i, fl_ii:fh_ii] = 1
    return obm, cf


def stft(x, win_size, fft_size, overlap=4):
    hop = int(win_size / overlap)
    w = np.hanning(win_size + 2)[1:-1]
    stft_out = np.array(
        [np.fft.rfft(w * x[i : i + win_size], n=fft_size) for i in range(0, len(x) - win_size, hop)]
    )
    return stft_out


def _overlap_and_add(x_frames, hop):
    num_frames, framelen = x_frames.shape
    segments = -(-framelen // hop)
    signal = np.pad(x_frames, ((0, segments), (0, segments * hop - framelen)))
    signal = signal.reshape((num_frames + segments, segments, hop))
    signal = np.transpose(signal, [1, 0, 2])
    signal = signal.reshape((-1, hop))
    signal = signal[:-segments]
    signal = signal.reshape((segments, num_frames + segments - 1, hop))
    signal = np.sum(signal, axis=0)
    end = (len(x_frames) - 1) * hop + framelen
    signal = signal.reshape(-1)[:end]
    return signal


def remove_silent_frames(x, y, dyn_range, framelen, hop):
    w = np.hanning(framelen + 2)[1:-1]
    x_frames = np.array([w * x[i : i + framelen] for i in range(0, len(x) - framelen, hop)])
    y_frames = np.array([w * y[i : i + framelen] for i in range(0, len(x) - framelen, hop)])
    x_energies = 20 * np.log10(np.linalg.norm(x_frames, axis=1) + EPS)
    mask = (np.max(x_energies) - dyn_range - x_energies) < 0
    x_frames = x_frames[mask]
    y_frames = y_frames[mask]
    return _overlap_and_add(x_frames, hop), _overlap_and_add(y_frames, hop)


def vect_two_norm(x, axis=-1):
    return np.sum(np.square(x), axis=axis, keepdims=True)


def row_col_normalize(x):
    x_normed = x + EPS * np.random.standard_normal(x.shape)
    x_normed -= np.mean(x_normed, axis=-1, keepdims=True)
    x_inv = 1.0 / np.sqrt(vect_two_norm(x_normed))
    x_diags = np.array([np.diag(x_inv[i].reshape(-1)) for i in range(x_inv.shape[0])])
    x_normed = np.matmul(x_diags, x_normed)
    x_normed += +EPS * np.random.standard_normal(x_normed.shape)
    x_normed -= np.mean(x_normed, axis=1, keepdims=True)
    x_inv = 1.0 / np.sqrt(vect_two_norm(x_normed, axis=1))
    x_diags = np.array([np.diag(x_inv[i].reshape(-1)) for i in range(x_inv.shape[0])])
    x_normed = np.matmul(x_normed, x_diags)
    return x_normed


def stoi_taal_numpy(
    x: np.ndarray,
    y: np.ndarray,
    fs_sig: int,
    extended: bool = False,
    *,
    apply_silence_removal: bool = True,
) -> float:
    """STOI between clean reference x and degraded y (1D float arrays)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have same length, got {x.shape} vs {y.shape}")
    if fs_sig != FS:
        x = resample_oct(x, FS, fs_sig)
        y = resample_oct(y, FS, fs_sig)
    if apply_silence_removal:
        x, y = remove_silent_frames(x, y, DYN_RANGE, N_FRAME, int(N_FRAME / 2))
    _ensure_obm()
    x_spec = stft(x, N_FRAME, NFFT, overlap=2).transpose()
    y_spec = stft(y, N_FRAME, NFFT, overlap=2).transpose()
    if x_spec.shape[-1] < N:
        warnings.warn(
            "Not enough STFT frames for STOI"
            + (" after removing silent frames" if apply_silence_removal else "")
            + "; returning 1e-5.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1e-5
    x_tob = np.sqrt(np.matmul(OBM, np.square(np.abs(x_spec))))
    y_tob = np.sqrt(np.matmul(OBM, np.square(np.abs(y_spec))))
    x_segments = np.array([x_tob[:, m - N : m] for m in range(N, x_tob.shape[1] + 1)])
    y_segments = np.array([y_tob[:, m - N : m] for m in range(N, y_tob.shape[1] + 1)])
    if extended:
        x_n = row_col_normalize(x_segments)
        y_n = row_col_normalize(y_segments)
        return float(np.sum(x_n * y_n / N) / x_n.shape[0])
    normalization_consts = np.linalg.norm(x_segments, axis=2, keepdims=True) / (
        np.linalg.norm(y_segments, axis=2, keepdims=True) + EPS
    )
    y_segments_normalized = y_segments * normalization_consts
    clip_value = 10 ** (-BETA / 20)
    y_primes = np.minimum(y_segments_normalized, x_segments * (1 + clip_value))
    y_primes = y_primes - np.mean(y_primes, axis=2, keepdims=True)
    x_segments = x_segments - np.mean(x_segments, axis=2, keepdims=True)
    y_primes /= np.linalg.norm(y_primes, axis=2, keepdims=True) + EPS
    x_segments /= np.linalg.norm(x_segments, axis=2, keepdims=True) + EPS
    correlations_components = y_primes * x_segments
    J, M = x_segments.shape[0], x_segments.shape[1]
    d = np.sum(correlations_components) / (J * M)
    return float(d)
