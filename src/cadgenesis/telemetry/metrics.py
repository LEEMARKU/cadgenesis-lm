"""cadgenesis.telemetry.metrics
============================
Metrics collection for CADGenesis-LM v6.0: counters, gauges and histograms
behind a thread-safe registry with snapshot serialization.
"""

from __future__ import annotations

import enum
import itertools
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, TypeVar

from typing_extensions import Self


class MetricType(str, enum.Enum):
    """Supported metric families."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


class Metric:
    """Base class for all metrics."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.labels = dict(labels or {})
        self._lock = threading.Lock()

    @property
    def type(self) -> MetricType:
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        raise NotImplementedError


class Counter(Metric):
    """Monotonically increasing counter."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: dict[str, str] | None = None,
    ) -> None:
        super().__init__(name, description, labels)
        self._value: float = 0.0

    @property
    def type(self) -> MetricType:
        return MetricType.COUNTER

    def inc(self, amount: float = 1.0) -> None:
        if amount < 0:
            raise ValueError("counter increments must be non-negative")
        with self._lock:
            self._value += amount

    def reset(self) -> None:
        with self._lock:
            self._value = 0.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "type": self.type.value,
                "labels": self.labels,
                "value": self._value,
            }


class Gauge(Metric):
    """Value that can be set and adjusted (CPU, memory, queue depth)."""

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: dict[str, str] | None = None,
    ) -> None:
        super().__init__(name, description, labels)
        self._value: float = 0.0

    @property
    def type(self) -> MetricType:
        return MetricType.GAUGE

    def set(self, value: float) -> None:
        with self._lock:
            self._value = float(value)

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "type": self.type.value,
                "labels": self.labels,
                "value": self._value,
            }


@dataclass
class HistogramBucket:
    """Summary of observations falling below ``upper_bound``."""

    upper_bound: float
    count: int = 0


class Histogram(Metric):
    """Histogram with configurable buckets; tracks sum and observation count."""

    DEFAULT_BOUNDS = [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float("inf")]

    def __init__(
        self,
        name: str,
        description: str = "",
        labels: dict[str, str] | None = None,
        buckets: list[float] | None = None,
    ) -> None:
        super().__init__(name, description, labels)
        bounds = buckets if buckets is not None else self.DEFAULT_BOUNDS
        invalid_bounds = (
            not bounds
            or any(not math.isfinite(b) for b in bounds[:-1])
            or bounds[-1] != float("inf")
        )
        if invalid_bounds:
            raise ValueError("histogram buckets must end with float('inf') and be finite otherwise")
        if any(a >= b for a, b in itertools.pairwise(bounds)):
            raise ValueError("histogram buckets must be strictly increasing")
        self.bounds = bounds
        self._counts = [0] * len(bounds)
        self._sum = 0.0
        self._n = 0

    @property
    def type(self) -> MetricType:
        return MetricType.HISTOGRAM

    def observe(self, value: float) -> None:
        with self._lock:
            self._sum += value
            self._n += 1
            for index, bound in enumerate(self.bounds):
                if value <= bound:
                    self._counts[index] += 1
                    break

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            cumulative = []
            running = 0
            for bound, count in zip(self.bounds, self._counts, strict=True):
                running += count
                cumulative.append({"le": bound, "count": running})
            return {
                "name": self.name,
                "type": self.type.value,
                "labels": self.labels,
                "count": self._n,
                "sum": self._sum,
                "buckets": cumulative,
            }

    @property
    def mean(self) -> float:
        with self._lock:
            return self._sum / self._n if self._n else 0.0


MetricT = TypeVar("MetricT", bound=Metric)


class MetricsRegistry:
    """Thread-safe registry of named metrics.

    Usage::

        registry = MetricsRegistry()
        counter = registry.counter("inference.requests")
        counter.inc()
        registry.snapshot()
    """

    def __init__(self, prefix: str = "cadgenesis") -> None:
        self.prefix = prefix
        self._metrics: dict[str, Metric] = {}
        self._lock = threading.Lock()

    def _full_name(self, name: str) -> str:
        return name if name.startswith(self.prefix) else f"{self.prefix}.{name}"

    def register(self, metric: MetricT) -> MetricT:
        with self._lock:
            key = self._full_name(metric.name)
            metric.name = key
            if key in self._metrics:
                raise ValueError(f"metric '{key}' already registered")
            self._metrics[key] = metric
            return metric

    def counter(
        self,
        name: str,
        description: str = "",
        labels: dict[str, str] | None = None,
    ) -> Counter:
        return self.register(Counter(name, description, labels))

    def gauge(
        self,
        name: str,
        description: str = "",
        labels: dict[str, str] | None = None,
    ) -> Gauge:
        return self.register(Gauge(name, description, labels))

    def histogram(
        self,
        name: str,
        description: str = "",
        labels: dict[str, str] | None = None,
        buckets: list[float] | None = None,
    ) -> Histogram:
        return self.register(Histogram(name, description, labels, buckets))

    def get(self, name: str) -> Metric | None:
        return self._metrics.get(self._full_name(name))

    def names(self) -> list[str]:
        return sorted(self._metrics)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            metrics = [metric.snapshot() for metric in self._metrics.values()]
        metrics.sort(key=lambda m: m["name"])
        return {
            "registry": self.prefix,
            "timestamp": time.time(),
            "metrics": metrics,
        }

    def clear(self) -> None:
        with self._lock:
            self._metrics.clear()


class StepTimer:
    """Utility timing a recurring operation and recording it into a histogram.

    Usage::

        timer = StepTimer(registry.histogram("step.time"))
        with timer:
            run_step()
    """

    def __init__(self, histogram: Histogram) -> None:
        self.histogram = histogram

    def __enter__(self) -> Self:
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        if exc_type is None:
            self.histogram.observe(time.perf_counter() - self._start)
