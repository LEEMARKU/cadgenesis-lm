"""
cadgenesis.runtime.benchmarks
=============================
Live micro-benchmarks for the HardwareAwareRuntime (v6.2).

Measures what the machine actually does *right now*: forward latency and
peak-memory deltas for a model config, plus decode-step latency.  Results are
returned as plain dataclasses (the caller persists/reports them); nothing is
invented — every number comes from a timed run.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ForwardBenchmark:
    """Timing + memory for one batched forward."""

    device: str
    batch_size: int
    seq_len: int
    dtype: str
    elapsed_s: float
    peak_mem_mib: float
    backend: str

    @property
    def tokens_per_s(self) -> float:
        total = self.batch_size * self.seq_len
        return total / self.elapsed_s if self.elapsed_s > 0 else float("inf")


@dataclass(frozen=True)
class DecodeBenchmark:
    """Per-step latency for KV-cached autoregressive decoding."""

    device: str
    steps: int
    total_s: float
    per_step_ms: float


def _peak_cuda_mib() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def benchmark_forward(
    model: torch.nn.Module,
    *,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    steps: int = 3,
) -> ForwardBenchmark:
    """
    Time ``model.forward`` on random ids.  ``steps`` runs, the median is
    reported; CUDA peak memory is measured across the runs.
    """
    model.eval()
    model = model.to(device)
    ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    ctx = (
        torch.autocast(device_type=_device_type(device), dtype=dtype)
        if dtype != torch.float32
        else torch.no_grad()
    )
    with torch.no_grad(), ctx:
        for _ in range(steps):
            t0 = time.perf_counter()
            model(ids, ids, torch.zeros_like(ids))
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)
    samples.sort()
    peak = _peak_cuda_mib()
    return ForwardBenchmark(
        device=device,
        batch_size=batch_size,
        seq_len=seq_len,
        dtype=str(dtype),
        elapsed_s=samples[len(samples) // 2],
        peak_mem_mib=peak,
        backend=model.__class__.__name__,
    )


def benchmark_decode(
    model: torch.nn.Module,
    engine_fn: Callable,
    *,
    text: str,
    max_len: int,
    device: str = "cpu",
    steps: int = 2,
) -> DecodeBenchmark:
    """
    Time ``engine_fn(text, max_len=...)`` (an inference engine) end-to-end;
    reports total and per-step latency.
    """
    samples: list[float] = []
    for _ in range(steps):
        t0 = time.perf_counter()
        engine_fn(text, max_len=max_len)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    median = samples[len(samples) // 2]
    return DecodeBenchmark(
        device=device,
        steps=max_len,
        total_s=median,
        per_step_ms=median * 1000.0 / max(max_len, 1),
    )


def _device_type(device: str) -> str:
    return "cuda" if device.startswith("cuda") else device.split(":")[0]
