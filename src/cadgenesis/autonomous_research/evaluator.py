"""
Benchmark Evaluator - Measure accuracy, CAD quality, engineering correctness, memory efficiency,
inference speed, GPU utilization.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class BenchmarkCategory(str, Enum):
    ACCURACY = "accuracy"
    CAD_QUALITY = "cad_quality"
    ENGINEERING_CORRECTNESS = "engineering_correctness"
    MEMORY_EFFICIENCY = "memory_efficiency"
    INFERENCE_SPEED = "inference_speed"
    GPU_UTILIZATION = "gpu_utilization"
    ROBUSTNESS = "robustness"


@dataclass
class BenchmarkMetric:
    """A single benchmark metric."""

    name: str
    category: BenchmarkCategory
    value: float
    unit: str
    higher_is_better: bool = True
    baseline: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""

    report_id: str
    experiment_id: str
    model_id: str
    metrics: list[BenchmarkMetric] = field(default_factory=list)
    summary: str = ""
    passed: bool = True
    threshold_failures: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_metric(self, metric: BenchmarkMetric) -> None:
        self.metrics.append(metric)

    def get_metrics_by_category(self, category: BenchmarkCategory) -> list[BenchmarkMetric]:
        return [m for m in self.metrics if m.category == category]

    def compute_overall_score(self) -> float:
        if not self.metrics:
            return 0.0
        scores = []
        for m in self.metrics:
            if m.baseline is not None and m.baseline != 0:
                ratio = m.value / m.baseline
                scores.append(min(ratio, 2.0) if m.higher_is_better else max(2.0 - ratio, 0.0))
            else:
                scores.append(1.0)
        return sum(scores) / len(scores)


@dataclass
class _BenchmarkEntry:
    """Registered benchmark metadata."""

    function: Callable[[Any], float]
    category: BenchmarkCategory
    higher_is_better: bool
    baseline: float | None
    unit: str


class BenchmarkEvaluator:
    """Evaluates models against benchmarks."""

    def __init__(self):
        self._benchmarks: dict[str, _BenchmarkEntry] = {}
        self._thresholds: dict[str, float] = {}
        self._reports: dict[str, EvaluationReport] = {}
        self._lock = RLock()

    def register_benchmark(
        self,
        name: str,
        category: BenchmarkCategory,
        function: Callable[[Any], float],
        higher_is_better: bool = True,
        baseline: float | None = None,
        unit: str = "",
    ) -> None:
        """Register a benchmark function."""
        with self._lock:
            self._benchmarks[name] = _BenchmarkEntry(
                function=function,
                category=category,
                higher_is_better=higher_is_better,
                baseline=baseline,
                unit=unit,
            )

    def set_threshold(self, name: str, threshold: float) -> None:
        """Set minimum threshold for a benchmark."""
        with self._lock:
            self._thresholds[name] = threshold

    def evaluate(
        self,
        model: Any,
        experiment_id: str,
        model_id: str,
        benchmark_names: list[str] | None = None,
    ) -> EvaluationReport:
        """Run benchmarks on a model."""
        report = EvaluationReport(
            report_id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            model_id=model_id,
        )

        benchmarks_to_run = benchmark_names or list(self._benchmarks.keys())

        for name in benchmarks_to_run:
            bench = self._benchmarks.get(name)
            if not bench:
                continue

            try:
                value = bench.function(model)
                metric = BenchmarkMetric(
                    name=name,
                    category=bench.category,
                    value=value,
                    unit=bench.unit,
                    higher_is_better=bench.higher_is_better,
                    baseline=bench.baseline,
                )
                report.add_metric(metric)

                # Check threshold
                threshold = self._thresholds.get(name)
                if threshold is not None:
                    if bench.higher_is_better and value < threshold:
                        report.threshold_failures.append(f"{name}: {value} < {threshold}")
                        report.passed = False
                    elif not bench.higher_is_better and value > threshold:
                        report.threshold_failures.append(f"{name}: {value} > {threshold}")
                        report.passed = False

            except Exception as e:
                report.threshold_failures.append(f"{name}: ERROR - {e}")
                report.passed = False

        # Generate summary
        overall = report.compute_overall_score()
        report.summary = (
            f"Overall score: {overall:.3f}. Passed: {report.passed}. "
            f"Metrics: {len(report.metrics)}."
        )

        with self._lock:
            self._reports[report.report_id] = report

        return report

    def get_report(self, report_id: str) -> EvaluationReport | None:
        with self._lock:
            return self._reports.get(report_id)

    def list_reports(self, experiment_id: str | None = None) -> list[EvaluationReport]:
        with self._lock:
            reports = list(self._reports.values())
            if experiment_id:
                reports = [r for r in reports if r.experiment_id == experiment_id]
            return reports
