"""
STOI: ``torchaudio.functional.stoi`` (если есть) → **pystoi** → Taal (numpy; GPU torch при ``device=cuda``).
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional, Union

import numpy as np
import torch

try:
    import torchaudio
except ImportError:  # pragma: no cover
    torchaudio = None

from ._stoi_taal_numpy import stoi_taal_numpy
from ._stoi_taal_torch import stoi_taal_torch

_NO_TORCHAUDIO_STOI_WARNED = False
_PystoiFn = Optional[Callable[..., float]]


def _get_pystoi_stoi() -> _PystoiFn:
    try:
        from pystoi import stoi as pystoi_stoi

        return pystoi_stoi
    except ImportError:
        return None


def _warn_no_torchaudio_stoi() -> None:
    global _NO_TORCHAUDIO_STOI_WARNED
    if _NO_TORCHAUDIO_STOI_WARNED:
        return
    try:
        import multiprocessing as mp

        if mp.current_process().name != "MainProcess":
            return
    except Exception:
        pass
    _NO_TORCHAUDIO_STOI_WARNED = True
    print(
        "[src_STOI] torchaudio.functional.stoi в этой версии TorchAudio отсутствует — "
        "используется pystoi (рекомендуется: pip install pystoi), иначе Taal."
    )


def _to_device(d: Optional[Union[str, torch.device]]) -> torch.device:
    if d is None:
        return torch.device("cpu")
    if isinstance(d, torch.device):
        return d
    return torch.device(d)


def _resample_waveform_torch(wave_1d: np.ndarray, orig_sr: int, new_sr: int) -> np.ndarray:
    if torchaudio is None:
        raise RuntimeError("torchaudio is required for resampling.")
    w = torch.from_numpy(wave_1d.astype(np.float32, copy=False)).unsqueeze(0)
    out = torchaudio.functional.resample(w, orig_freq=orig_sr, new_freq=new_sr)
    return out.squeeze(0).cpu().numpy().astype(np.float64)


def compute_stoi_scalar(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int,
    *,
    extended: bool = False,
    resample_mode: str = "torchaudio",
    device: Optional[Union[str, torch.device]] = None,
    apply_silence_removal: bool = True,
) -> float:
    dev = _to_device(device)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    degraded = np.asarray(degraded, dtype=np.float64).reshape(-1)

    if not apply_silence_removal:
        fs_stoi = 10_000
        ref = reference
        deg = degraded
        if sample_rate != fs_stoi:
            ref = _resample_waveform_torch(ref, sample_rate, fs_stoi)
            deg = _resample_waveform_torch(deg, sample_rate, fs_stoi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
        return float(
            stoi_taal_numpy(ref, deg, fs_stoi, extended=extended, apply_silence_removal=False)
        )

    use_gpu = dev.type == "cuda" and torch.cuda.is_available()
    if extended and use_gpu:
        use_gpu = False

    ta = None
    if torchaudio is not None:
        ta = getattr(torchaudio.functional, "stoi", None)

    if ta is not None:
        ref_t = torch.from_numpy(reference.astype(np.float32)).unsqueeze(0)
        deg_t = torch.from_numpy(degraded.astype(np.float32)).unsqueeze(0)
        if use_gpu:
            ref_t = ref_t.to(dev)
            deg_t = deg_t.to(dev)
        try:
            out = ta(ref_t, deg_t, sample_rate, extended=extended)
        except TypeError:
            out = ta(ref_t, deg_t, sample_rate)
        try:
            return float(out.reshape(-1)[0].item())
        except Exception:
            if use_gpu:
                ref_t = ref_t.cpu()
                deg_t = deg_t.cpu()
                try:
                    out = ta(ref_t, deg_t, sample_rate, extended=extended)
                except TypeError:
                    out = ta(ref_t, deg_t, sample_rate)
                return float(out.reshape(-1)[0].item())
            raise

    _warn_no_torchaudio_stoi()
    pystoi_stoi = _get_pystoi_stoi()
    if pystoi_stoi is not None:
        ref_f = reference.astype(np.float32)
        deg_f = degraded.astype(np.float32)
        return float(pystoi_stoi(ref_f, deg_f, sample_rate, extended=extended))

    if use_gpu and not extended:
        ref_t = torch.as_tensor(reference, device=dev, dtype=torch.float32)
        deg_t = torch.as_tensor(degraded, device=dev, dtype=torch.float32)
        return float(
            stoi_taal_torch(
                ref_t, deg_t, sample_rate, device=dev, apply_silence_removal=True
            ).item()
        )

    if resample_mode == "torchaudio":
        fs_stoi = 10_000
        if sample_rate != fs_stoi:
            reference = _resample_waveform_torch(reference, sample_rate, fs_stoi)
            degraded = _resample_waveform_torch(degraded, sample_rate, fs_stoi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return stoi_taal_numpy(reference, degraded, fs_stoi, extended=extended, apply_silence_removal=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return stoi_taal_numpy(reference, degraded, sample_rate, extended=extended, apply_silence_removal=True)


def compute_stoi_scalar_from_tensors(
    reference: torch.Tensor,
    degraded: torch.Tensor,
    sample_rate: int,
    *,
    extended: bool = False,
    resample_mode: str = "torchaudio",
    device: Optional[Union[str, torch.device]] = None,
    apply_silence_removal: bool = True,
) -> float:
    dev = _to_device(device)
    r = reference.detach().reshape(-1).float()
    d = degraded.detach().reshape(-1).float()
    n = min(r.numel(), d.numel())
    r, d = r[:n], d[:n]
    return compute_stoi_scalar(
        r.cpu().numpy(),
        d.cpu().numpy(),
        sample_rate,
        extended=extended,
        resample_mode=resample_mode,
        device=dev,
        apply_silence_removal=apply_silence_removal,
    )
