from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_interface import StoiPredictorModel


def _default_twelve_layer_channels() -> Tuple[int, ...]:
    """Как в Zezario et al.: 12 слоёв, набор каналов {16,32,64,128} повторяется три раза."""
    return (16, 32, 64, 128, 16, 32, 64, 128, 16, 32, 64, 128)


def _freq_stride_pattern(num_layers: int) -> List[int]:
    """Периодически уменьшаем ось частоты (stride 2 по freq), время не трогаем."""
    # Трижды stride по freq на слоях 2, 6, 10 (индексация с 1) — ~как типичный CNN по спектрограмме.
    strides = [1] * num_layers
    for i in (1, 5, 9):
        if i < num_layers:
            strides[i] = 2
    return strides


class StoiNetPredictor(StoiPredictorModel):
    """
    Небинтрузивный предсказатель STOI в духе STOI-Net (Zezario et al.):
    STFT-спектрограмма → CNN (12×Conv2d) → BLSTM → multiplicative attention →
    frame-wise FC → global average → scalar STOI.

    Обучение — как у остальных моделей в проекте: один скалярный таргет на чанк (без frame-level loss из статьи).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: int = 512,
        conv_channels: Sequence[int] | None = None,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        fc_dim: int = 128,
        dropout: float = 0.1,
        use_log_mag: bool = True,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.use_log_mag = bool(use_log_mag)

        chans = tuple(conv_channels) if conv_channels is not None else _default_twelve_layer_channels()
        if len(chans) != 12:
            raise ValueError("STOI-Net (paper): ожидается 12 conv-слоёв; задайте conv_channels из 12 элементов.")

        strides_h = _freq_stride_pattern(len(chans))
        layers: List[nn.Module] = []
        c_in = 1
        for i, c_out in enumerate(chans):
            sh = strides_h[i]
            layers.append(
                nn.Conv2d(
                    c_in,
                    c_out,
                    kernel_size=3,
                    padding=1,
                    stride=(sh, 1),
                )
            )
            layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU(inplace=True))
            c_in = c_out
        self.cnn = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=chans[-1],
            hidden_size=int(lstm_hidden),
            num_layers=int(lstm_layers),
            batch_first=True,
            bidirectional=True,
            dropout=float(dropout) if lstm_layers > 1 else 0.0,
        )

        blstm_dim = 2 * int(lstm_hidden)
        self.attention_linear = nn.Linear(blstm_dim, blstm_dim)
        self.frame_fc = nn.Sequential(
            nn.Linear(blstm_dim, int(fc_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(fc_dim), 1),
        )

    def _spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform [B, T] -> magnitude [B, n_fft//2+1, n_frames]"""
        # STFT стабильнее в float32
        x = waveform.float()
        window = torch.hamming_window(self.win_length, device=x.device, dtype=x.dtype)
        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        mag = stft.abs().clamp_min(1e-8)
        if self.use_log_mag:
            mag = torch.log1p(mag)
        return mag

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: ``(batch, time)`` mono, ~16 kHz (как в конфиге данных).
        Returns:
            ``(batch, 1)`` STOI в (0, 1).
        """
        if waveform.is_cuda:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                spec = self._spectrogram(waveform)
        else:
            spec = self._spectrogram(waveform)
        x = spec.unsqueeze(1)
        x = self.cnn(x)
        x = F.adaptive_avg_pool2d(x, (1, None))
        x = x.squeeze(2)
        x = x.transpose(1, 2)

        h, _ = self.lstm(x)
        attn = torch.sigmoid(self.attention_linear(h))
        h = h * attn

        per_frame = self.frame_fc(h)
        out = per_frame.mean(dim=1)
        return torch.sigmoid(out)
