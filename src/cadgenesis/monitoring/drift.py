"""cadgenesis.monitoring.drift
===========================
Input/output distribution drift detection for CADGenesis-LM v6.0.

Provides Population Stability Index (PSI), KL-divergence and Jensen-Shannon
divergence between a reference distribution and the current streaming
distribution, plus a ``FeatureDriftMonitor`` that accumulates observations over
time.
"""

from __future__ import annotations

import enum
import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from cadgenesis.utils.math import safe_div


class DriftMetric(str, enum.Enum):
    """Drift scoring method."""

    PSI = "psi"
    KL = "kl"
    JS = "js"


@dataclass
class DriftReport:
    """Result of a drift computation for one feature."""

    feature: str
    metric: DriftMetric
    score: float
    threshold: float
    drifted: bool
    samples_reference: int
    samples_current: int
    computed_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "metric": self.metric.value,
            "score": round(self.score, 6),
            "threshold": self.threshold,
            "drifted": self.drifted,
            "samples_reference": self.samples_reference,
            "samples_current": self.samples_current,
            "computed_at": self.computed_at,
        }


def _discretize(values: Sequence[float], bins: int, low: float, high: float) -> list[float]:
    """Normalised histogram of ``values`` over ``[low, high]`` with ``bins`` bins."""
    hist = [0.0] * bins
    if not values:
        return hist
    for value in values:
        if math.isnan(value) or value < low or value > high:
            continue
        index = min(int((value - low) / (high - low) * bins), bins - 1)
        hist[index] += 1.0
    total = sum(hist)
    if total > 0:
        hist = [c / total for c in hist]
    return hist


def _smoothed(probabilities: Sequence[float], epsilon: float = 1e-6) -> list[float]:
    return [max(p, epsilon) for p in probabilities]


def _kl(reference: Sequence[float], current: Sequence[float]) -> float:
    return sum(
        r * math.log(safe_div(r, c, default=1e-12))
        for r, c in zip(reference, current, strict=True)
        if r > 0
    )


def compute_drift(
    reference: Sequence[float],
    current: Sequence[float],
    bins: int = 10,
    metric: DriftMetric = DriftMetric.PSI,
    low: float | None = None,
    high: float | None = None,
    epsilon: float = 1e-6,
) -> float:
    """Score distributional divergence between two value sequences.

    Args:
        reference: Reference (baseline) observations.
        current: Current observations to compare.
        bins: Number of histogram bins.
        metric: PSI, KL or JS divergence.
        low/high: Binning range; defaults to data-driven min/max.
        epsilon: Smoothing constant for empty bins.

    Returns:
        A non-negative divergence score (0 = identical distributions).
    """
    if not current:
        return 0.0
    if low is None or high is None:
        all_values = list(reference) + list(current)
        low = low if low is not None else min(all_values)
        high = high if high is not None else max(all_values)
        if high == low:
            high = low + 1.0
    ref_hist = _smoothed(_discretize(reference, bins, low, high), epsilon)
    cur_hist = _smoothed(_discretize(current, bins, low, high), epsilon)

    if metric == DriftMetric.PSI:
        return sum(
            (r - c) * math.log(safe_div(r, c, default=1e-12))
            for r, c in zip(ref_hist, cur_hist, strict=True)
        )
    if metric == DriftMetric.KL:
        return _kl(ref_hist, cur_hist)
    if metric == DriftMetric.JS:
        mid = [(r + c) / 2.0 for r, c in zip(ref_hist, cur_hist, strict=True)]
        return 0.5 * _kl(ref_hist, mid) + 0.5 * _kl(cur_hist, mid)
    raise ValueError(f"unknown drift metric: {metric!r}")


class FeatureDriftMonitor:
    """Tracks reference vs streaming distributions per feature.

    Usage::

        monitor = FeatureDriftMonitor(reference={"length": [1, 2, 3]}, threshold=0.2)
        monitor.update("length", [1.1, 1.9, 3.2])
        report = monitor.evaluate()["length"]
    """

    def __init__(
        self,
        reference: dict[str, Sequence[float]],
        threshold: float = 0.2,
        bins: int = 10,
        metric: DriftMetric = DriftMetric.PSI,
        limits: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0; got {threshold}")
        self.reference = {k: list(v) for k, v in reference.items()}
        self.threshold = threshold
        self.bins = bins
        self.metric = metric
        self.limits = dict(limits or {})
        self._current: dict[str, list[float]] = {k: [] for k in reference}
        self._lock = threading.Lock()

    def update(self, feature: str, values: Sequence[float]) -> None:
        """Accumulate streaming observations for ``feature``."""
        if feature not in self.reference:
            raise KeyError(f"unknown feature {feature!r}; register it in the constructor")
        with self._lock:
            self._current[feature].extend(values)

    def reset(self, feature: str | None = None) -> None:
        with self._lock:
            if feature is None:
                for key in self._current:
                    self._current[key] = []
            else:
                self._current[feature] = []

    def score(self, feature: str) -> float:
        """Compute the current drift score for ``feature`` (non-mutating)."""
        if feature not in self.reference:
            raise KeyError(f"unknown feature {feature!r}")
        with self._lock:
            reference = list(self.reference[feature])
            current = list(self._current[feature])
        low, high = self.limits.get(feature, (None, None))
        return compute_drift(
            reference,
            current,
            bins=self.bins,
            metric=self.metric,
            low=low,
            high=high,
        )

    def evaluate(self) -> dict[str, DriftReport]:
        """Return per-feature drift reports."""
        reports: dict[str, DriftReport] = {}
        for feature in self.reference:
            with self._lock:
                reference_n = len(self.reference[feature])
                current_n = len(self._current[feature])
            score = self.score(feature)
            reports[feature] = DriftReport(
                feature=feature,
                metric=self.metric,
                score=score,
                threshold=self.threshold,
                drifted=score > self.threshold,
                samples_reference=reference_n,
                samples_current=current_n,
            )
        return reports

    def any_drifted(self) -> bool:
        return any(report.drifted for report in self.evaluate().values())
