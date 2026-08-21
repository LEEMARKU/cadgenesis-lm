"""
cadgenesis.training.profiler
============================
Training profiler for CADGenesis-LM.

``TrainingProfiler`` times forward/backward/optimizer-steps and data
loading, tracks throughput (tokens/sec), and records a per-step trace
that can be dumped to JSON for offline analysis.  Torch optional: the
profiler works with or without ``torch.profiler`` installed.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Self

import torch

try:
    import torch

    try:
        from torch.profiler import ProfilerActivity, profile, record_function
    except ImportError:  # pragma: no cover - torch.profiler unavailable
        profile = None  # type: ignore[misc, assignment]  # optional import fallback
        record_function = None  # type: ignore[misc, assignment]  # optional import fallback
        ProfilerActivity = None  # type: ignore[misc, assignment]  # optional import fallback
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore[assignment]
    profile = None  # type: ignore[misc, assignment]  # optional import fallback
    record_function = None  # type: ignore[misc, assignment]  # optional import fallback
    ProfilerActivity = None  # type: ignore[misc, assignment]  # optional import fallback

logger = logging.getLogger("cadgenesis.training.profiler")


@dataclass
class ProfilerStats:
    """Aggregate timing statistics."""

    steps: int = 0
    total_seconds: float = 0.0
    forward_seconds: float = 0.0
    backward_seconds: float = 0.0
    optimizer_seconds: float = 0.0
    data_seconds: float = 0.0
    tokens_processed: int = 0

    def tokens_per_second(self) -> float:
        if self.total_seconds <= 0.0:
            return 0.0
        return self.tokens_processed / self.total_seconds

    def as_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "total_seconds": round(self.total_seconds, 6),
            "forward_seconds": round(self.forward_seconds, 6),
            "backward_seconds": round(self.backward_seconds, 6),
            "optimizer_seconds": round(self.optimizer_seconds, 6),
            "data_seconds": round(self.data_seconds, 6),
            "tokens_processed": self.tokens_processed,
            "tokens_per_second": round(self.tokens_per_second(), 6),
        }


class TrainingProfiler:
    """Step-level profiler wrapping the training loop phases."""

    def __init__(self, enabled: bool = True, torch_profile: bool = False) -> None:
        self.enabled = enabled
        self.torch_profile = torch_profile and profile is not None
        self.stats = ProfilerStats()
        self.trace: list[dict[str, float]] = []
        self._last: dict[str, float] = {}
        self._torch_profiler: Any = None
        self._phase: str | None = None
        self._start: float = 0.0

    # ------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Begin profiling; allocates the torch.profiler session if enabled."""
        if not self.enabled:
            return
        self.stats = ProfilerStats()
        self.trace = []
        if self.torch_profile and torch is not None:
            self._torch_profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
                if torch.cuda.is_available()
                else [ProfilerActivity.CPU],
                record_shapes=False,
                profile_memory=True,
            )
            self._torch_profiler.start()

    def stop(self) -> None:
        """Stop the profiler and (optionally) dump the trace."""
        if self._torch_profiler is not None:
            self._torch_profiler.stop()
            self._torch_profiler = None

    def step_done(self, tokens: int = 0) -> None:
        """Finalize one training step and record it in the trace."""
        if not self.enabled:
            return
        self.stats.steps += 1
        self.stats.tokens_processed += int(tokens)
        self._last["tokens"] = float(tokens)
        self.trace.append(dict(self._last))
        self._last = {}
        self._last["tokens_per_second"] = self.stats.tokens_per_second()

    # -------------------------------------------------------------- phases

    def phase(self, name: str) -> Any:
        """Context manager timing one phase (data/forward/backward/optimizer)."""
        if not self.enabled:
            if self.torch_profile and record_function is not None:
                return record_function(name)
            return _nullcontext()
        return _PhaseTimer(self, name, torch_profiler=self._torch_profiler)

    # --------------------------------------------------------------- output

    def summary(self) -> str:
        """Human-readable one-line summary."""
        return ", ".join(f"{k}={v}" for k, v in self.stats.as_dict().items())

    def save_trace(self, path: str) -> None:
        """Write the per-step trace as JSON lines."""
        if not self.enabled:
            return
        with open(path, "w", encoding="utf-8") as handle:
            for row in self.trace:
                handle.write(json.dumps(row) + "\n")

    def _on_phase_end(self, name: str, elapsed: float) -> None:
        self._last[name] = round(elapsed, 6)
        if name == "data":
            self.stats.data_seconds += elapsed
        elif name == "forward":
            self.stats.forward_seconds += elapsed
        elif name == "backward":
            self.stats.backward_seconds += elapsed
        elif name == "optimizer":
            self.stats.optimizer_seconds += elapsed
        self.stats.total_seconds += elapsed


class _PhaseTimer:
    """Records elapsed wall time for one profiled phase."""

    def __init__(self, profiler: TrainingProfiler, name: str, torch_profiler: Any = None) -> None:
        self.profiler = profiler
        self.name = name
        self.torch_profiler = torch_profiler
        self._start: float = 0.0

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.profiler._on_phase_end(self.name, time.perf_counter() - self._start)


class _nullcontext:
    """Bare-bones context manager when profiling is disabled."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


__all__ = ["ProfilerStats", "TrainingProfiler"]
