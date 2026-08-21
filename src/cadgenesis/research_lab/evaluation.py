"""
Evaluation Framework - A/B testing, statistical testing, regression testing, benchmark
comparison.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Any

import numpy as np
from scipy import stats


class TestType(str, Enum):
    AB_TEST = "ab_test"
    STATISTICAL = "statistical"
    REGRESSION = "regression"
    BENCHMARK_COMPARISON = "benchmark_comparison"


@dataclass
class ABTestConfig:
    test_id: str
    name: str
    control_config: dict[str, Any]
    treatment_config: dict[str, Any]
    metric: str
    sample_size: int
    confidence_level: float = 0.95
    minimum_effect: float = 0.01


@dataclass
class ABTestResult:
    test_id: str
    control_mean: float
    treatment_mean: float
    p_value: float
    confidence_interval: tuple[float, float]
    significant: bool
    effect_size: float
    recommendation: str  # "implement", "reject", "inconclusive"


@dataclass
class StatisticalTestConfig:
    test_id: str
    name: str
    test_type: str  # t_test, wilcoxon, mann_whitney, chi2, anova
    samples: list[list[float]]
    alternative: str = "two-sided"  # two-sided, less, greater
    alpha: float = 0.05


@dataclass
class StatisticalTestResult:
    test_id: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float | None = None
    confidence_interval: tuple[float, float] | None = None


@dataclass
class RegressionTestConfig:
    test_id: str
    name: str
    baseline_metrics: dict[str, float]
    current_metrics: dict[str, float]
    tolerance: dict[str, float]  # metric -> max allowed regression


@dataclass
class RegressionTestResult:
    test_id: str
    regressions: dict[str, bool]  # metric -> is_regression
    severity: dict[str, str]  # metric -> none/warning/critical
    passed: bool


@dataclass
class BenchmarkComparisonConfig:
    test_id: str
    name: str
    baseline_results: dict[str, dict[str, float]]  # model -> {metric: value}
    candidate_results: dict[str, dict[str, float]]  # model -> {metric: value}
    metrics: list[str]


@dataclass
class BenchmarkComparisonResult:
    test_id: str
    improvements: dict[str, dict[str, float]]  # model -> {metric: improvement}
    significant_improvements: dict[str, list[str]]  # model -> [metrics]
    summary: str


class EvaluationFramework:
    def __init__(self):
        self._tests: dict[str, Any] = {}
        self._lock = RLock()

    def run_ab_test(
        self, config: ABTestConfig, run_fn: Callable[[dict[str, Any]], float]
    ) -> ABTestResult:
        """Run A/B test comparing control vs treatment."""
        control_samples = []
        treatment_samples = []

        for _ in range(config.sample_size):
            control_samples.append(run_fn(config.control_config))
            treatment_samples.append(run_fn(config.treatment_config))

        control_mean = np.mean(control_samples)
        treatment_mean = np.mean(treatment_samples)

        # Two-sample t-test
        _t_stat, p_value = stats.ttest_ind(treatment_samples, control_samples, equal_var=False)

        # Confidence interval for difference
        diff_mean = treatment_mean - control_mean
        se = np.sqrt(
            np.var(control_samples, ddof=1) / len(control_samples)
            + np.var(treatment_samples, ddof=1) / len(treatment_samples)
        )
        ci = stats.t.interval(
            config.confidence_level,
            len(control_samples) + len(treatment_samples) - 2,
            loc=diff_mean,
            scale=se,
        )

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.var(control_samples, ddof=1) + np.var(treatment_samples, ddof=1)) / 2
        )
        effect_size = diff_mean / pooled_std if pooled_std > 0 else 0

        significant = p_value < (1 - config.confidence_level)
        if significant and diff_mean > config.minimum_effect:
            recommendation = "implement"
        elif significant and diff_mean < -config.minimum_effect:
            recommendation = "reject"
        else:
            recommendation = "inconclusive"

        result = ABTestResult(
            test_id=config.test_id,
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            p_value=p_value,
            confidence_interval=ci,
            significant=significant,
            effect_size=effect_size,
            recommendation=recommendation,
        )

        with self._lock:
            self._tests[config.test_id] = result

        return result

    def run_statistical_test(self, config: StatisticalTestConfig) -> StatisticalTestResult:
        """Run various statistical tests."""
        samples = config.samples

        if config.test_type == "t_test" and len(samples) == 2:
            stat, p = stats.ttest_ind(samples[0], samples[1], alternative=config.alternative)
        elif config.test_type == "wilcoxon" and len(samples) == 2:
            stat, p = stats.wilcoxon(samples[0], samples[1], alternative=config.alternative)
        elif config.test_type == "mann_whitney" and len(samples) == 2:
            stat, p = stats.mannwhitneyu(samples[0], samples[1], alternative=config.alternative)
        elif config.test_type == "chi2":
            stat, p, _, _ = stats.chi2_contingency(samples)
        elif config.test_type == "anova":
            stat, p = stats.f_oneway(*samples)
        else:
            raise ValueError(f"Unsupported test type: {config.test_type}")

        significant = p < config.alpha

        # Effect size for t-test
        effect_size = None
        ci = None
        if config.test_type == "t_test" and len(samples) == 2:
            n1, n2 = len(samples[0]), len(samples[1])
            var1, var2 = np.var(samples[0], ddof=1), np.var(samples[1], ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            effect_size = (
                (np.mean(samples[0]) - np.mean(samples[1])) / pooled_std if pooled_std > 0 else 0
            )
            diff = np.mean(samples[0]) - np.mean(samples[1])
            se = pooled_std * np.sqrt(1 / n1 + 1 / n2)
            ci = stats.t.interval(1 - config.alpha, n1 + n2 - 2, loc=diff, scale=se)

        result = StatisticalTestResult(
            test_id=config.test_id,
            statistic=stat,
            p_value=p,
            significant=significant,
            effect_size=effect_size,
            confidence_interval=ci,
        )

        with self._lock:
            self._tests[config.test_id] = result

        return result

    def run_regression_test(self, config: RegressionTestConfig) -> RegressionTestResult:
        """Check for performance regressions."""
        regressions = {}
        severity = {}

        for metric, baseline in config.baseline_metrics.items():
            current = config.current_metrics.get(metric)
            if current is None:
                regressions[metric] = False
                severity[metric] = "none"
                continue

            tolerance = config.tolerance.get(metric, 0.01)
            relative_change = (current - baseline) / abs(baseline) if baseline != 0 else 0

            is_regression = relative_change < -tolerance
            regressions[metric] = is_regression

            if is_regression:
                if relative_change < -0.1:
                    severity[metric] = "critical"
                elif relative_change < -0.05:
                    severity[metric] = "warning"
                else:
                    severity[metric] = "minor"
            else:
                severity[metric] = "none"

        passed = not any(regressions.values())

        result = RegressionTestResult(
            test_id=config.test_id,
            regressions=regressions,
            severity=severity,
            passed=passed,
        )

        with self._lock:
            self._tests[config.test_id] = result

        return result

    def run_benchmark_comparison(
        self, config: BenchmarkComparisonConfig
    ) -> BenchmarkComparisonResult:
        """Compare benchmark results across models."""
        improvements = {}
        significant_improvements = {}

        for model in config.candidate_results:
            if model not in config.baseline_results:
                continue

            model_improvements = {}
            sig_metrics = []

            for metric in config.metrics:
                baseline = config.baseline_results[model].get(metric, 0)
                candidate = config.candidate_results[model].get(metric, 0)

                improvement = (candidate - baseline) / abs(baseline) if baseline != 0 else 0

                model_improvements[metric] = improvement

                # Consider >5% improvement significant
                if improvement > 0.05:
                    sig_metrics.append(metric)

            improvements[model] = model_improvements
            significant_improvements[model] = sig_metrics

        # Generate summary
        total_sig = sum(len(m) for m in significant_improvements.values())
        summary = (
            f"Compared {len(config.candidate_results)} candidates against baseline. "
            f"{total_sig} significant improvements across {len(config.metrics)} metrics."
        )

        result = BenchmarkComparisonResult(
            test_id=config.test_id,
            improvements=improvements,
            significant_improvements=significant_improvements,
            summary=summary,
        )

        with self._lock:
            self._tests[config.test_id] = result

        return result

    def get_test(self, test_id: str) -> Any | None:
        with self._lock:
            return self._tests.get(test_id)

    def list_tests(self) -> list[str]:
        with self._lock:
            return list(self._tests.keys())
