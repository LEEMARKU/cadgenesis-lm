"""
Statistical Analyzer - Confidence intervals, hypothesis testing, significance testing,
regression detection.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any

import numpy as np
from scipy import stats
from scipy.stats import bootstrap


class StatisticalTestType(str, Enum):
    T_TEST = "t_test"
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann_whitney"
    CHI2 = "chi2"
    ANOVA = "anova"
    BOOTSTRAP_CI = "bootstrap_ci"
    REGRESSION_DETECTION = "regression_detection"


@dataclass
class StatisticalTestConfig:
    test_id: str
    name: str
    test_type: StatisticalTestType
    data: dict[str, Any]  # varies by test type
    alpha: float = 0.05
    alternative: str = "two-sided"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalResult:
    test_id: str
    test_type: StatisticalTestType
    statistic: float
    p_value: float
    significant: bool
    confidence_interval: tuple[float, float] | None = None
    effect_size: float | None = None
    interpretation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalReport:
    report_id: str
    experiment_id: str
    results: list[StatisticalResult] = field(default_factory=list)
    summary: str = ""
    created_at: float = field(default_factory=time.time)


class StatisticalAnalyzer:
    """Performs statistical analysis on experimental results."""

    def __init__(self):
        self._reports: dict[str, StatisticalReport] = {}
        self._lock = RLock()

    def run_test(self, config: StatisticalTestConfig) -> StatisticalResult:
        """Run a statistical test based on configuration."""
        test_type = config.test_type

        if test_type == StatisticalTestType.T_TEST:
            return self._t_test(config)
        elif test_type == StatisticalTestType.WILCOXON:
            return self._wilcoxon(config)
        elif test_type == StatisticalTestType.MANN_WHITNEY:
            return self._mann_whitney(config)
        elif test_type == StatisticalTestType.CHI2:
            return self._chi2(config)
        elif test_type == StatisticalTestType.ANOVA:
            return self._anova(config)
        elif test_type == StatisticalTestType.BOOTSTRAP_CI:
            return self._bootstrap_ci(config)
        elif test_type == StatisticalTestType.REGRESSION_DETECTION:
            return self._regression_detection(config)
        else:
            raise ValueError(f"Unknown test type: {test_type}")

    def _t_test(self, config: StatisticalTestConfig) -> StatisticalResult:
        sample1 = np.array(config.data["sample1"])
        sample2 = np.array(config.data.get("sample2", []))

        if len(sample2) > 0:
            stat, p = stats.ttest_ind(
                sample1, sample2, alternative=config.alternative, equal_var=False
            )
            # Effect size (Cohen's d)
            n1, n2 = len(sample1), len(sample2)
            var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
            pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            effect = (np.mean(sample1) - np.mean(sample2)) / pooled if pooled > 0 else 0
            # CI
            diff = np.mean(sample1) - np.mean(sample2)
            se = pooled * np.sqrt(1 / n1 + 1 / n2)
            ci = stats.t.interval(1 - config.alpha, n1 + n2 - 2, loc=diff, scale=se)
        else:
            # One-sample t-test
            popmean = config.data.get("popmean", 0)
            stat, p = stats.ttest_1samp(sample1, popmean, alternative=config.alternative)
            effect = (
                (np.mean(sample1) - popmean) / np.std(sample1, ddof=1) if np.std(sample1) > 0 else 0
            )
            n = len(sample1)
            se = np.std(sample1, ddof=1) / np.sqrt(n)
            ci = stats.t.interval(1 - config.alpha, n - 1, loc=np.mean(sample1) - popmean, scale=se)

        significant = p < config.alpha
        interp = (
            f"Significant difference (p={p:.4f})"
            if significant
            else f"No significant difference (p={p:.4f})"
        )

        return StatisticalResult(
            test_id=config.test_id,
            test_type=config.test_type,
            statistic=stat,
            p_value=p,
            significant=significant,
            confidence_interval=ci,
            effect_size=effect,
            interpretation=interp,
        )

    def _wilcoxon(self, config: StatisticalTestConfig) -> StatisticalResult:
        sample1 = np.array(config.data["sample1"])
        sample2 = np.array(config.data.get("sample2", []))

        if len(sample2) > 0:
            stat, p = stats.wilcoxon(sample1, sample2, alternative=config.alternative)
        else:
            stat, p = stats.wilcoxon(sample1, alternative=config.alternative)

        significant = p < config.alpha
        return StatisticalResult(
            test_id=config.test_id,
            test_type=config.test_type,
            statistic=stat,
            p_value=p,
            significant=significant,
            interpretation=f"{'Significant' if significant else 'Not significant'} (p={p:.4f})",
        )

    def _mann_whitney(self, config: StatisticalTestConfig) -> StatisticalResult:
        sample1 = np.array(config.data["sample1"])
        sample2 = np.array(config.data["sample2"])

        stat, p = stats.mannwhitneyu(sample1, sample2, alternative=config.alternative)
        significant = p < config.alpha

        # Effect size (rank-biserial correlation)
        n1, n2 = len(sample1), len(sample2)
        effect = 1 - (2 * stat) / (n1 * n2)

        return StatisticalResult(
            test_id=config.test_id,
            test_type=config.test_type,
            statistic=stat,
            p_value=p,
            significant=significant,
            effect_size=effect,
            interpretation=f"{'Significant' if significant else 'Not significant'} (p={p:.4f})",
        )

    def _chi2(self, config: StatisticalTestConfig) -> StatisticalResult:
        table = np.array(config.data["contingency_table"])
        stat, p, dof, expected = stats.chi2_contingency(table)
        significant = p < config.alpha

        return StatisticalResult(
            test_id=config.test_id,
            test_type=config.test_type,
            statistic=stat,
            p_value=p,
            significant=significant,
            metadata={"dof": dof, "expected": expected.tolist()},
            interpretation=(
                f"{'Significant association' if significant else 'No significant association'} "
                f"(p={p:.4f})"
            ),
        )

    def _anova(self, config: StatisticalTestConfig) -> StatisticalResult:
        samples = [np.array(s) for s in config.data["samples"]]
        stat, p = stats.f_oneway(*samples)
        significant = p < config.alpha

        # Effect size (eta-squared)
        all_data = np.concatenate(samples)
        grand_mean = np.mean(all_data)
        ss_between = sum(len(s) * (np.mean(s) - grand_mean) ** 2 for s in samples)
        ss_total = np.sum((all_data - grand_mean) ** 2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        return StatisticalResult(
            test_id=config.test_id,
            test_type=config.test_type,
            statistic=stat,
            p_value=p,
            significant=significant,
            effect_size=eta_squared,
            interpretation=(
                f"{'Significant difference between groups' if significant else 'No significant'} "
                f"difference (p={p:.4f})"
            ),
        )

    def _bootstrap_ci(self, config: StatisticalTestConfig) -> StatisticalResult:
        sample = np.array(config.data["sample"])
        statistic = config.data.get("statistic", np.mean)
        n_resamples = config.data.get("n_resamples", 10000)

        # Use scipy bootstrap
        res = bootstrap(
            (sample,), statistic, n_resamples=n_resamples, confidence_level=1 - config.alpha
        )
        ci = (res.confidence_interval.low, res.confidence_interval.high)

        return StatisticalResult(
            test_id=config.test_id,
            test_type=config.test_type,
            statistic=statistic(sample),
            p_value=0.0,  # Not applicable
            significant=True,
            confidence_interval=ci,
            interpretation=f"Bootstrap {100 * (1 - config.alpha)}% CI: [{ci[0]:.4f}, {ci[1]:.4f}]",
        )

    def _regression_detection(self, config: StatisticalTestConfig) -> StatisticalResult:
        """Detect performance regression using sequential testing."""
        baseline = np.array(config.data["baseline"])
        current = np.array(config.data["current"])
        metric_name = config.data.get("metric", "metric")

        # Compare means
        stat, p = stats.ttest_ind(
            current, baseline, alternative="less"
        )  # One-sided: is current worse?

        # Effect size
        pooled = np.sqrt((np.var(baseline, ddof=1) + np.var(current, ddof=1)) / 2)
        effect = (np.mean(current) - np.mean(baseline)) / pooled if pooled > 0 else 0

        # Regression if significant and negative effect
        is_regression = p < config.alpha and effect < 0

        return StatisticalResult(
            test_id=config.test_id,
            test_type=config.test_type,
            statistic=stat,
            p_value=p,
            significant=is_regression,
            effect_size=effect,
            interpretation=f"Regression detected: {is_regression} (p={p:.4f}, effect={effect:.4f})",
            metadata={
                "metric": metric_name,
                "baseline_mean": float(np.mean(baseline)),
                "current_mean": float(np.mean(current)),
            },
        )

    def create_report(
        self, experiment_id: str, results: list[StatisticalResult]
    ) -> StatisticalReport:
        report = StatisticalReport(
            report_id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            results=results,
        )

        # Generate summary
        sig_count = sum(1 for r in results if r.significant)
        report.summary = (
            f"Ran {len(results)} statistical tests. {sig_count} significant at "
            f"alpha={results[0].metadata.get('alpha', 0.05) if results else 0.05}."
        )

        with self._lock:
            self._reports[report.report_id] = report

        return report

    def get_report(self, report_id: str) -> StatisticalReport | None:
        with self._lock:
            return self._reports.get(report_id)
