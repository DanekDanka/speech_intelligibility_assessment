from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
import torchaudio

from .base import StoiTargetComputer, WaveformPreprocessor


class TorchaudioResampleMonoChunkPreprocessor(WaveformPreprocessor):
    """
    Loads WAV, mixes down to mono, resamples to ``sample_rate``,
    extracts ``chunk_index``-th segment of ``chunk_duration_sec`` (zero-padded).
    """

    def __init__(
        self,
        sample_rate: int,
        chunk_duration_sec: float,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.chunk_duration_sec = float(chunk_duration_sec)
        self.chunk_samples = int(round(chunk_duration_sec * self.sample_rate))
        if self.chunk_samples <= 0:
            raise ValueError("chunk_duration_sec * sample_rate must be positive")

    def _load_mono_resampled(self, path: Union[str, Path]) -> torch.Tensor:
        # normalize=True — явно как в torchaudio по умолчанию (int16 → float [-1,1]); см. stereo_stoi_channels / инференс
        wav, sr = torchaudio.load(str(path), normalize=True)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)
        return wav.squeeze(0)

    def load_mono_resampled(self, path: Union[str, Path]) -> torch.Tensor:
        """Полный сигнал после ресэмпла (для кэша STOI: один раз на файл)."""
        return self._load_mono_resampled(path)

    def _chunk(self, waveform: torch.Tensor, chunk_index: int) -> torch.Tensor:
        start = chunk_index * self.chunk_samples
        end = start + self.chunk_samples
        if start >= waveform.numel():
            return torch.zeros(self.chunk_samples, dtype=waveform.dtype, device=waveform.device)
        chunk = waveform[start:end]
        if chunk.numel() < self.chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, self.chunk_samples - chunk.numel()))
        return chunk

    def chunk_waveform(self, waveform: torch.Tensor, chunk_index: int) -> torch.Tensor:
        """Вырезать чанк из уже загруженного тензора."""
        return self._chunk(waveform, chunk_index)

    def process_degraded_file(self, path: Union[str, Path], chunk_index: int) -> torch.Tensor:
        w = self._load_mono_resampled(path)
        return self._chunk(w, chunk_index)

    def process_reference_file(self, path: Union[str, Path], chunk_index: int) -> torch.Tensor:
        return self.process_degraded_file(path, chunk_index)


class TorchaudioBackedStoiTargetComputer(StoiTargetComputer):
    """Делегирует в ``metrics.stoi_backend`` (pystoi / Taal / GPU Taal)."""

    def __init__(
        self,
        sample_rate: int,
        *,
        extended: bool = False,
        resample_mode: str = "torchaudio",
        compute_device: str | None = None,
        apply_silence_removal: bool = True,
    ) -> None:
        self._sample_rate = int(sample_rate)
        self.extended = extended
        self.resample_mode = resample_mode
        self._apply_silence_removal = bool(apply_silence_removal)
        if compute_device is None:
            self._compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._compute_device = torch.device(compute_device)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def stoi_compute_device(self) -> torch.device:
        return self._compute_device

    @property
    def apply_silence_removal(self) -> bool:
        """Если False — Taal-STOI по полной длине чанка без отбрасывания тихих окон (как в train.py)."""
        return self._apply_silence_removal

    def compute(self, reference: torch.Tensor, degraded: torch.Tensor) -> float:
        from ..metrics.stoi_backend import compute_stoi_scalar_from_tensors

        r = reference.detach().float()
        d = degraded.detach().float()
        n = min(r.numel(), d.numel())
        r = r[:n]
        d = d[:n]
        return compute_stoi_scalar_from_tensors(
            r,
            d,
            self.sample_rate,
            extended=self.extended,
            resample_mode=self.resample_mode,
            device=self._compute_device,
            apply_silence_removal=self._apply_silence_removal,
        )
