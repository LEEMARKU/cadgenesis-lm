"""cadgenesis.multimodal.encoders.sensor
=======================================
Sensor encoder (vibration, force, temperature, pressure, telemetry).

A sensor signal is normalised into a :class:`SensorDocument` holding a
``(N, C)`` float array plus the channel names.  The encoder computes
time-domain statistics per channel (mean, std, min, max, energy, zero
crossings, FFT band energy) and maps the fixed-size descriptor through an
MLP into the shared raw feature space.  These embeddings underpin the
Sensor-to-Simulation cross-modal pairs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch
import torch.nn as nn

from cadgenesis.multimodal.common import Modality
from cadgenesis.multimodal.encoders.base import MultimodalEncoder, tensorize

_STAT_DIM = 10
_BAND_DIM = 8
_DESCRIPTOR_SIZE = _STAT_DIM + _BAND_DIM + 4


@dataclass
class SensorDocument:
    """Normalised sensor signal: (N, C) float array with channel names."""

    data: Any
    channels: list[str] = field(default_factory=list)
    source: str = "vibration"
    sample_rate_hz: float = 1000.0

    def __post_init__(self) -> None:
        self.data = tensorize(self.data, dtype="float32")
        if self.data.dim() != 2:
            raise ValueError(f"sensor data must be (N, C); got {tuple(self.data.shape)}")
        if not self.channels:
            self.channels = [f"ch{i}" for i in range(self.data.shape[1])]


def _sensor_descriptor(document: SensorDocument) -> torch.Tensor:
    """Deterministic descriptor of a sensor signal.

    Layout: 10 time-domain stats (mean, std, min, max, energy, zero-crossing,
    mean abs, variance, peak-to-peak, skewness-ish) + 8 FFT band energies
    (normalised log bands over the Nyquist range) + 4 source/rate fields.
    """
    signal = document.data
    descriptor = torch.zeros(_DESCRIPTOR_SIZE, dtype=torch.float32)
    if signal.numel() == 0:
        return descriptor

    stats = signal.mean(dim=0)
    std = signal.std(dim=0).clamp_min(1e-8)
    minimum = signal.min(dim=0).values
    maximum = signal.max(dim=0).values
    energy = (signal**2).sum(dim=0)
    zero_crossings = (signal[:-1] * signal[1:] < 0).sum(dim=0).float()
    mean_abs = signal.abs().mean(dim=0)
    skew = (((signal - stats) / std) ** 3).mean(dim=0).abs()

    descriptor[0] = stats.mean().item()
    descriptor[1] = std.mean().item()
    descriptor[2] = minimum.min().item()
    descriptor[3] = maximum.max().item()
    descriptor[4] = math.log1p(energy.sum().item())
    descriptor[5] = zero_crossings.sum().item()
    descriptor[6] = mean_abs.mean().item()
    descriptor[7] = (maximum - minimum).max().item()
    descriptor[8] = skew.mean().item()
    descriptor[9] = energy.mean().item()

    try:
        spectrum = torch.fft.rfft(signal, dim=0).abs()
        bands = torch.linspace(0, spectrum.shape[0] - 1, _BAND_DIM + 1).long()
        for band_index in range(_BAND_DIM):
            start = bands[band_index]
            end = max(bands[band_index + 1], start + 1)
            band_energy = spectrum[start:end].pow(2).sum()
            descriptor[_STAT_DIM + band_index] = math.log1p(band_energy.item())
    except RuntimeError:  # pragma: no cover - non-CPU tensors
        pass

    descriptor[_STAT_DIM + _BAND_DIM + 0] = math.log1p(document.sample_rate_hz)
    descriptor[_STAT_DIM + _BAND_DIM + 1] = math.log1p(signal.shape[0])
    descriptor[_STAT_DIM + _BAND_DIM + 2] = 1.0 if document.source == "vibration" else 0.0
    descriptor[_STAT_DIM + _BAND_DIM + 3] = 1.0 if len(document.channels) > 1 else 0.0
    return descriptor


class SensorEncoder(MultimodalEncoder):
    """Encoder for the ``sensor`` modality."""

    modality: ClassVar[Modality] = Modality.SENSOR

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
                f"sensor encoder expects (B, {_DESCRIPTOR_SIZE}) descriptors; got {tuple(x.shape)}"
            )
        return self.net(x)

    def encode(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return self.forward(inputs)
        if isinstance(inputs, SensorDocument):
            inputs = [inputs]
        items = list(inputs)
        if not items:
            raise ValueError("cannot encode an empty sensor batch")
        descriptors = torch.stack([_sensor_descriptor(d) for d in items])
        return self.forward(descriptors)


__all__ = ["SensorDocument", "SensorEncoder"]
