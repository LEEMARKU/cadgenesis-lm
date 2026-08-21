"""
cadgenesis.research.stats
=========================
Statistical evaluation for CADGenesis-LM research infrastructure.

- Confidence intervals: normal-approx and bootstrap (percentile)
- Hypothesis testing: paired/unpaired Welch t-test (pure-Python)
- Significance testing & effect size: Cohen's d
- Reproducibility helpers: mean/std/median/IQR over repeated runs

All pure Python (no scipy dependency); results are plain dataclasses.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceInterval:
    """Point estimate with a confidence interval."""

    estimate: float
    lower: float
    upper: float
    level: float
    method: str  # "normal" | "bootstrap"

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass(frozen=True)
class HypothesisTestResult:
    """Outcome of a two-sample significance test."""

    statistic: float
    p_value: float
    significant: bool
    alpha: float
    method: str
    degrees_of_freedom: float | None = None

    def __str__(self) -> str:
        flag = "significant" if self.significant else "not significant"
        return (
            f"{self.method}: t={self.statistic:.4f}, p={self.p_value:.4f}"
            f" ({flag} at alpha={self.alpha})"
        )


def _t_critical(alpha: float, df: float, two_sided: bool = True) -> float:
    """Approximate two-sided Student-t critical value (Abramowitz-Stegun)."""
    if df <= 0:
        df = 1e-6
    if df > 300:
        z = _z_critical(alpha if two_sided else alpha / 2)
        return z
    a = alpha if two_sided else alpha / 2
    z = _z_critical(a)
    g1 = 0.25 * (z**3 + z)
    g2 = (1 / 96) * (5 * z**5 + 16 * z**3 + 3 * z)
    g3 = (1 / 384) * (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z)
    return z + g1 / df + g2 / (df**2) + g3 / (df**3)


def _z_critical(a: float) -> float:
    """Two-sided normal quantile for tail probability ``a/2`` (inverse erf)."""
    if a <= 0 or a >= 1:
        raise ValueError("alpha must be in (0, 1)")
    p = 1.0 - a / 2
    if p <= 0.5:
        raise ValueError("alpha out of range")
    # Acklam's approximation
    a0 = -3.969683028665376e1
    a1 = 2.209460984245205e2
    a2 = -2.759285104469687e2
    a3 = 1.383577518672690e2
    a4 = -3.066479806614716e1
    a5 = 2.506628277459239e0
    b1 = -5.447609879822406e1
    b2 = 1.615858368580409e2
    b3 = -1.556989798598866e2
    b4 = 6.680131188771972e1
    b5 = -1.328068155288572e1
    c0 = -7.784894002430293e-3
    c1 = -3.223964580411365e-1
    c2 = -2.400758277161838
    c3 = -2.549732539343734
    c4 = 4.374664141464968
    c5 = 2.938163982698783
    d1 = 7.784695709041462e-3
    d2 = 3.224671290700398e-1
    d3 = 2.445134137142996
    d4 = 3.754408661907416
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        num = ((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5
        den = (((d1 * q + d2) * q + d3) * q + d4) * q + 1
        return num / den
    if p <= phigh:
        q = p - 0.5
        r = q * q
        num = (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q
        den = ((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1
        return num / den
    q = math.sqrt(-2 * math.log(1 - p))
    num = ((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5
    den = (((d1 * q + d2) * q + d3) * q + d4) * q + 1
    return -(num / den)


def mean_ci_normal(
    samples: Sequence[float],
    level: float = 0.95,
    method: str = "normal",
) -> ConfidenceInterval:
    """Normal-approximation CI for the mean."""
    n = len(samples)
    if n < 2:
        raise ValueError("need at least 2 samples")
    mean = statistics.fmean(samples)
    sem = statistics.stdev(samples) / math.sqrt(n)
    z = _z_critical(1.0 - level)
    return ConfidenceInterval(mean, mean - z * sem, mean + z * sem, level, method)


def bootstrap_ci(
    samples: Sequence[float],
    level: float = 0.95,
    resamples: int = 2000,
    seed: int | None = None,
    method: str = "bootstrap",
) -> ConfidenceInterval:
    """Percentile bootstrap CI for the mean."""
    if len(samples) < 2:
        raise ValueError("need at least 2 samples")
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choices(samples, k=len(samples))) for _ in range(resamples)]
    means.sort()
    lower_index = round(((1 - level) / 2) * resamples)
    upper_index = round((1 - (1 - level) / 2) * resamples)
    return ConfidenceInterval(
        statistics.fmean(samples),
        means[max(0, lower_index - 1)],
        means[min(resamples - 1, upper_index - 1)],
        level,
        method,
    )


def welch_t_test(
    group_a: Sequence[float],
    group_b: Sequence[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> HypothesisTestResult:
    """Welch's t-test (unequal variances) with approximate p-value."""
    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError("each group needs at least 2 samples")
    na, nb = len(group_a), len(group_b)
    mean_a, mean_b = statistics.fmean(group_a), statistics.fmean(group_b)
    var_a, var_b = statistics.variance(group_a), statistics.variance(group_b)
    se = math.sqrt(var_a / na + var_b / nb)
    statistic = (mean_a - mean_b) / se if se > 0 else 0.0
    denom = (var_a / na) ** 2 / (na - 1) + (var_b / nb) ** 2 / (nb - 1)
    df = ((var_a / na + var_b / nb) ** 2 / denom) if denom > 0 else max(na, nb) - 1
    p_value = _two_sided_p(statistic, df) if se > 0 else 1.0  # identical means
    if alternative == "greater":
        p_value = p_value / 2
    elif alternative == "less":
        p_value = 1.0 - p_value / 2
    return HypothesisTestResult(
        statistic=statistic,
        p_value=min(1.0, max(0.0, p_value)),
        significant=p_value < alpha,
        alpha=alpha,
        method="welch_t_test",
        degrees_of_freedom=df,
    )


def _two_sided_p(t: float, df: float) -> float:
    """Two-sided t-distribution survival probability.

    Integrates the t pdf over [0, |t|] via Simpson's rule and derives the
    tail probability from symmetry: p = 2 * (1 - F(|t|)).
    """
    if df > 200:
        return 2 * (1 - _normal_cdf(abs(t)))
    z = abs(t)
    if z == 0:
        return 1.0
    f = lambda x: (1 + (x**2) / df) ** (-(df + 1) / 2)  # noqa: E731
    n_steps = 4000
    h = z / n_steps
    area = f(0.0) + f(z)
    for i in range(1, n_steps):
        coeff = 4 if i % 2 else 2
        area += coeff * f(i * h)
    area *= h / 3  # integral of pdf over [0, z]
    norm = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    cdf = 0.5 + norm * area
    return min(1.0, max(0.0, 2 * (1.0 - cdf)))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


def cohens_d(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """Standardized effect size (pooled SD)."""
    if len(group_a) < 2 or len(group_b) < 2:
        raise ValueError("each group needs at least 2 samples")
    pooled = math.sqrt(
        (
            (len(group_a) - 1) * statistics.variance(group_a)
            + (len(group_b) - 1) * statistics.variance(group_b)
        )
        / (len(group_a) + len(group_b) - 2)
    )
    if pooled == 0:
        return 0.0
    return (statistics.fmean(group_a) - statistics.fmean(group_b)) / pooled


def describe(samples: Iterable[float]) -> dict[str, float]:
    """Summary statistics over repeated runs (reproducibility reporting)."""
    values = list(samples)
    if not values:
        return {}
    ordered = sorted(values)
    n = len(ordered)
    lower_q = ordered[n // 4] if n > 3 else ordered[0]
    upper_q = ordered[3 * n // 4] if n > 3 else ordered[-1]
    return {
        "n": float(n),
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if n > 1 else 0.0,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "iqr": upper_q - lower_q,
    }


__all__ = [
    "ConfidenceInterval",
    "HypothesisTestResult",
    "bootstrap_ci",
    "cohens_d",
    "describe",
    "mean_ci_normal",
    "welch_t_test",
]
