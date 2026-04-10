# Taal STOI (extended=False) on GPU: resample via torchaudio, STFT + bands on torch.

from __future__ import annotations

import numpy as np
import torch
import torchaudio

from ._stoi_taal_numpy import (
    BETA,
    DYN_RANGE,
    FS,
    N,
    NFFT,
    N_FRAME,
    NUMBAND,
    MINFREQ,
    remove_silent_frames,
    thirdoct,
)

EPS = 1e-12
_OBM_CACHE: dict[str, torch.Tensor] = {}


def _obm_tensor(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = f"{device}:{dtype}"
    if key not in _OBM_CACHE:
        obm_np, _ = thirdoct(FS, NFFT, NUMBAND, MINFREQ)
        _OBM_CACHE[key] = torch.as_tensor(obm_np, device=device, dtype=dtype)
    return _OBM_CACHE[key]


def _hann_256(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    w = np.hanning(N_FRAME + 2)[1:-1].astype(np.float32)
    return torch.as_tensor(w, device=device, dtype=dtype)


def _resample_1d(x: torch.Tensor, orig_sr: int, new_sr: int) -> torch.Tensor:
    if orig_sr == new_sr:
        return x
    w = x.unsqueeze(0)
    y = torchaudio.functional.resample(w, orig_freq=orig_sr, new_freq=new_sr)
    return y.squeeze(0)


def stoi_taal_torch(
    reference: torch.Tensor,
    degraded: torch.Tensor,
    sample_rate: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    apply_silence_removal: bool = True,
) -> torch.Tensor:
    ref = reference.reshape(-1).to(device=device, dtype=dtype)
    deg = degraded.reshape(-1).to(device=device, dtype=dtype)
    n = min(ref.numel(), deg.numel())
    ref, deg = ref[:n], deg[:n]

    ref = _resample_1d(ref, sample_rate, FS)
    deg = _resample_1d(deg, sample_rate, FS)

    if apply_silence_removal:
        ref_np = ref.detach().cpu().float().numpy().astype(np.float64)
        deg_np = deg.detach().cpu().float().numpy().astype(np.float64)
        ref_np, deg_np = remove_silent_frames(ref_np, deg_np, DYN_RANGE, N_FRAME, int(N_FRAME / 2))
        ref = torch.as_tensor(ref_np, device=device, dtype=dtype)
        deg = torch.as_tensor(deg_np, device=device, dtype=dtype)

    if ref.numel() <= N_FRAME:
        return torch.tensor(1e-5, device=device, dtype=dtype)

    win = _hann_256(device, dtype)
    hop = N_FRAME // 2

    def _stft_1d(x1: torch.Tensor) -> torch.Tensor:
        z = torch.stft(
            x1,
            n_fft=NFFT,
            hop_length=hop,
            win_length=N_FRAME,
            window=win,
            center=False,
            return_complex=True,
        )
        return z.abs().pow(2)

    x_pow = _stft_1d(ref)
    y_pow = _stft_1d(deg)
    if x_pow.shape[-1] < N:
        return torch.tensor(1e-5, device=device, dtype=dtype)

    obm = _obm_tensor(device, dtype)
    x_tob = (obm @ x_pow).sqrt().clamp_min(EPS)
    y_tob = (obm @ y_pow).sqrt().clamp_min(EPS)

    x_seg = x_tob.unfold(1, N, 1)
    y_seg = y_tob.unfold(1, N, 1)
    if x_seg.shape[1] == 0:
        return torch.tensor(1e-5, device=device, dtype=dtype)

    x_seg = x_seg.permute(1, 0, 2)
    y_seg = y_seg.permute(1, 0, 2)

    norm_x = torch.linalg.norm(x_seg, dim=2, keepdim=True)
    norm_y = torch.linalg.norm(y_seg, dim=2, keepdim=True)
    y_n = y_seg * (norm_x / (norm_y + EPS))

    clip_value = 10 ** (-BETA / 20)
    y_p = torch.minimum(y_n, x_seg * (1.0 + clip_value))
    y_p = y_p - y_p.mean(dim=2, keepdim=True)
    x_c = x_seg - x_seg.mean(dim=2, keepdim=True)
    y_p = y_p / (torch.linalg.norm(y_p, dim=2, keepdim=True) + EPS)
    x_c = x_c / (torch.linalg.norm(x_c, dim=2, keepdim=True) + EPS)
    j, m, _k = x_c.shape
    d = (y_p * x_c).sum() / (j * m)
    return d
