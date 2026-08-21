"""cadgenesis.utils.math
=====================
Numerical helpers for CADGenesis-LM v6.0: geometry math, streaming statistics,
EMA, percentile estimation, and numeric formatting helpers.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable, Sequence


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` into the inclusive interval ``[low, high]``."""
    return max(low, min(high, value))


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between ``a`` and ``b`` with weight ``t``."""
    return a + (b - a) * t


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide guarding against zero/NaN denominators, returning ``default``."""
    if denominator == 0 or math.isnan(denominator):
        return default
    return numerator / denominator


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Hermite interpolation on the interval ``[edge0, edge1]``."""
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def round_to(value: float, precision: float = 0.01) -> float:
    """Round ``value`` to a multiple of ``precision``."""
    if precision <= 0:
        return value
    return round(value / precision) * precision


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean of a finite iterable (NaN-safe)."""
    items = [v for v in values if not isinstance(v, (bool,)) and not math.isnan(v)]
    if not items:
        return 0.0
    return statistics.fmean(items)


class RunningStats:
    """Online mean / variance / standard-deviation (Welford's algorithm)."""

    def __init__(self) -> None:
        self._count: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0

    def update(self, value: float) -> None:
        """Accumulate a single observation."""
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2

    def update_many(self, values: Iterable[float]) -> None:
        for value in values:
            self.update(value)

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def variance(self) -> float:
        if self._count < 2:
            return 0.0
        return self._m2 / (self._count - 1)

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)

    def merge(self, other: RunningStats) -> None:
        """Merge another RunningStats instance (parallel combination)."""
        if other.count == 0:
            return
        count = self._count + other.count
        if count == 0:
            return
        delta = other.mean - self._mean
        self._mean += delta * (other.count / count)
        self._m2 += other._m2 + delta * delta * (self._count * other.count / count)
        self._count = count

    def as_dict(self) -> dict[str, float]:
        return {
            "count": self._count,
            "mean": self.mean,
            "std": self.stddev,
            "variance": self.variance,
        }


class ExponentialMovingAverage:
    """Exponentially-weighted moving average with optional warmup bias correction."""

    def __init__(self, alpha: float, warmup_steps: int = 0) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self._value: float = 0.0
        self._steps: int = 0

    def update(self, value: float) -> float:
        self._steps += 1
        if self._steps == 1:
            self._value = value
        else:
            self._value = self.alpha * value + (1.0 - self.alpha) * self._value
        if self._steps <= self.warmup_steps:
            correction = 1.0 - (1.0 - self.alpha) ** self._steps
            return self._value / correction
        return self._value

    @property
    def value(self) -> float:
        return self._value


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of a *sorted* sequence (0 <= q <= 100)."""
    if not sorted_values:
        raise ValueError("percentile of empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100]; got {q}")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = q / 100.0 * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(sorted_values[lower])
    weight = rank - lower
    return lerp(float(sorted_values[lower]), float(sorted_values[upper]), weight)


def median(values: Iterable[float]) -> float:
    """Median of an iterable of numbers."""
    sorted_values = sorted(v for v in values if not math.isnan(v))
    if not sorted_values:
        return 0.0
    return percentile(sorted_values, 50.0)


def pct_change(current: float, reference: float) -> float:
    """Percentage change between ``reference`` and ``current`` (relative)."""
    return safe_div(current - reference, abs(reference)) * 100.0


def norm_angle_deg(angle: float) -> float:
    """Wrap an angle in degrees into ``(-180, 180]``."""
    return ((angle + 180.0) % 360.0) - 180.0


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance between two equally-sized points/vectors."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def bbox_volume(min_corner: Sequence[float], max_corner: Sequence[float]) -> float:
    """Volume of an axis-aligned bounding box."""
    dims = [max_c - min_c for min_c, max_c in zip(min_corner, max_corner, strict=True)]
    volume = 1.0
    for d in dims:
        volume *= max(0.0, d)
    return volume


def aspect_ratio(width: float, height: float) -> float:
    """Aspect ratio with NaN-safe fallback to 1.0."""
    if width <= 0 or height <= 0 or math.isnan(width) or math.isnan(height):
        return 1.0
    return width / height
