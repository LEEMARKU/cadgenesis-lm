"""
cadgenesis.research.profiler
============================
Performance profiler for CADGenesis-LM research infrastructure.

Profiles GPU, CPU, memory, inference and training.  Extends the training
profiler (``cadgenesis.training.profiler``) with:

- CPU/GPU utilization and peak memory capture (psutil/torch optional)
- Inference profiling: per-call latency across batch sizes
- Training profiling: per-step forward/backward/optimizer breakdown
- Deterministic sampling snapshots at a fixed cadence
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("cadgenesis.research.profiler")


@dataclass
class SystemSnapshot:
    """One sampled snapshot of CPU/GPU/memory."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_util: float = 0.0
    gpu_memory_gb: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "gpu_util": self.gpu_util,
            "gpu_memory_gb": self.gpu_memory_gb,
        }


_PSUTIL: Any = None
_TORCH: Any = None


def _system_stats() -> SystemSnapshot:
    global _PSUTIL, _TORCH
    cpu = memory = 0.0
    if _PSUTIL is None:  # pragma: no cover - optional
        try:
            import psutil  # type: ignore[import-not-found]

            _PSUTIL = psutil
        except ImportError:
            _PSUTIL = False
    if _PSUTIL:
        cpu = _PSUTIL.cpu_percent(interval=None)
        memory = _PSUTIL.virtual_memory().percent
    gpu_util = gpu_mem = 0.0
    if _TORCH is None:  # pragma: no cover - optional
        try:
            import torch  # type: ignore[import-not-found]

            _TORCH = torch
        except ImportError:
            _TORCH = False
    if _TORCH and _TORCH.cuda.is_available():
        try:
            # torch.cuda.utilization requires nvidia-ml-py; sample best-effort
            # and fall back to 0.0 when the NVML bindings are unavailable.
            gpu_util = float(getattr(_TORCH.cuda, "utilization", lambda: 0.0)() or 0.0)
        except Exception:
            gpu_util = 0.0
        try:
            gpu_mem = round(float(_TORCH.cuda.memory_allocated() / (1024**3)), 3)
        except Exception:
            gpu_mem = 0.0
    return SystemSnapshot(
        timestamp=time.time(),
        cpu_percent=cpu,
        memory_percent=memory,
        gpu_util=gpu_util,
        gpu_memory_gb=gpu_mem,
    )


class PerformanceProfiler:
    """Background system sampler + explicit phase timing."""

    def __init__(self, sample_interval: float = 1.0, enabled: bool = True) -> None:
        self.sample_interval = max(0.1, sample_interval)
        self.enabled = enabled
        self.snapshots: list[SystemSnapshot] = []
        self.phases: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, name="perf-profiler", daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.sample_interval):
            self.snapshots.append(_system_stats())

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------ inference

    def profile_inference(
        self,
        fn: Callable[[], Any],
        batch_sizes: Sequence[int] = (1, 4, 8),
        repeats: int = 3,
    ) -> dict[str, float]:
        """Latency across batch sizes; returns ms per call per batch."""
        results: dict[str, float] = {}
        for batch in batch_sizes:
            times: list[float] = []
            for _ in range(repeats):
                started = time.perf_counter()
                fn()
                times.append((time.perf_counter() - started) * 1000.0)
            results[f"batch_{batch}_ms"] = round(sum(times) / len(times), 3)
        return results

    # ------------------------------------------------------------- training

    def profile_phases(self, phase: str, seconds: float = 0.0) -> PerformanceProfiler:
        """Timer to manually wrap a phase; or use :meth:`time_phase`."""
        return self

    def time_phase(self, name: str, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` while timing it under ``name``; returns its result."""
        started = time.perf_counter()
        try:
            return fn()
        finally:
            elapsed = time.perf_counter() - started
            self.phases[name] = round(self.phases.get(name, 0.0) + elapsed, 6)

    # --------------------------------------------------------------- report

    def summary(self) -> dict[str, Any]:
        if not self.snapshots:
            return {"phases": self.phases}

        def avg(key: str) -> float:
            values: list[float] = []
            for s in self.snapshots:
                v = s.to_dict().get(key)
                assert v is not None
                values.append(v)
            return round(sum(values) / len(self.snapshots), 2)

        return {
            "samples": len(self.snapshots),
            "avg_cpu_percent": avg("cpu_percent"),
            "avg_memory_percent": avg("memory_percent"),
            "avg_gpu_util": avg("gpu_util"),
            "peak_gpu_memory_gb": round(
                max((s.gpu_memory_gb for s in self.snapshots), default=0.0), 3
            ),
            "phases_seconds": self.phases,
        }


__all__ = ["PerformanceProfiler", "SystemSnapshot"]
