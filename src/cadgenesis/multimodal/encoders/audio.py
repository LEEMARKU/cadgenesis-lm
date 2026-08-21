"""cadgenesis.multimodal.encoders.audio
=====================================
Audio encoder (voice annotations, spoken design briefs).

Audio is normalised into an :class:`AudioDocument` holding the mono/two
channel waveform as a ``(N,)`` or ``(N, C)`` float array plus the sample
rate.  The encoder computes short-frame energy statistics (mean, std, min,
max per frame), FFT spectral-band energies and tempo/pitch proxies, and maps
the fixed-size descriptor through an MLP into the shared raw feature space.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.encoders.base import MultimodalEncoder, tensorize

_FRAME_DIM = 16
_BAND_DIM = 8
_STAT_DIM = 8
_DESCRIPTOR_SIZE = _FRAME_DIM + _BAND_DIM + _STAT_DIM
_FRAME_MS = 30.0


@dataclass
class AudioDocument:
    """Normalised audio waveform with sample rate."""

    data: Any
    sample_rate_hz: float = 16000.0

    def __post_init__(self) -> None:
        self.data = tensorize(self.data, dtype="float32")
        if self.data.dim() == 1:
            self.data = self.data[:, None]
        if self.data.dim() != 2:
            raise ValueError(f"audio data must be (N,) or (N, C); got {tuple(self.data.shape)}")


def _audio_descriptor(document: AudioDocument) -> torch.Tensor:
    """Deterministic descriptor of an audio waveform.

    Layout: 16 per-frame RMS statistics (mean over the frame-count grid) +
    8 FFT band energies + 8 global statistics.
    """
    waveform = document.data
    descriptor = torch.zeros(_DESCRIPTOR_SIZE, dtype=torch.float32)
    if waveform.numel() == 0:
        return descriptor

    frame_length = max(int(document.sample_rate_hz * _FRAME_MS / 1000.0), 1)
    num_frames = max(waveform.shape[0] // frame_length, 1)
    total = num_frames * frame_length
    if waveform.shape[0] < total:
        pad = torch.zeros(total - waveform.shape[0], waveform.shape[1])
        waveform = torch.cat([waveform, pad], dim=0)
    trimmed = waveform[:total].reshape(num_frames, frame_length, -1)
    frame_rms = trimmed.pow(2).mean(dim=1).sqrt()
    frame_mean = frame_rms.mean(dim=1)

    grid = torch.linspace(0, frame_mean.shape[0] - 1, _FRAME_DIM).long()
    sampled = frame_mean[grid]
    descriptor[:_FRAME_DIM] = sampled / max(sampled.abs().max(), 1e-8)

    try:
        spectrum = torch.fft.rfft(waveform.mean(dim=1), dim=0).abs()
        bands = torch.linspace(0, spectrum.shape[0] - 1, _BAND_DIM + 1).long()
        for band_index in range(_BAND_DIM):
            start = bands[band_index]
            end = max(bands[band_index + 1], start + 1)
            band_energy = spectrum[start:end].pow(2).sum()
            descriptor[_FRAME_DIM + band_index] = math.log1p(band_energy.item())
    except RuntimeError:  # pragma: no cover - non-CPU tensors
        pass

    offset = _FRAME_DIM + _BAND_DIM
    descriptor[offset + 0] = waveform.mean().item()
    descriptor[offset + 1] = waveform.std().item()
    descriptor[offset + 2] = waveform.abs().max().item()
    descriptor[offset + 3] = (waveform.abs() < 1e-6).float().mean().item()
    descriptor[offset + 4] = math.log1p(waveform.shape[0])
    descriptor[offset + 5] = math.log1p(document.sample_rate_hz)
    descriptor[offset + 6] = frame_mean.std().item() if num_frames > 1 else 0.0
    descriptor[offset + 7] = waveform.pow(2).mean().sqrt().item()
    return descriptor


class AudioEncoder(MultimodalEncoder):
    """Encoder for the ``audio`` modality."""

    modality: ClassVar[Modality] = Modality.AUDIO

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(feature_dim=feature_dim)
        self.net = nn.Sequential(
            nn.Linear(_DESCRIPTOR_SIZE, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[-1] != _DESCRIPTOR_SIZE:
            raise ValueError(
                f"audio encoder expects (B, {_DESCRIPTOR_SIZE}) descriptors; got {tuple(x.shape)}"
            )
        return self.net(x)

    def encode(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return self.forward(inputs)
        if isinstance(inputs, AudioDocument):
            inputs = [inputs]
        items = list(inputs)
        if not items:
            raise ValueError("cannot encode an empty audio batch")
        descriptors = torch.stack([_audio_descriptor(d) for d in items])
        return self.forward(descriptors)


__all__ = ["AudioDocument", "AudioEncoder"]
