"""
System Benchmark - CAD quality, reasoning, planning, retrieval, simulation, latency, throughput,
memory efficiency, GPU utilization, reliability, scalability.
"""

from __future__ import annotations

import contextlib
import statistics
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class BenchmarkCategory(str, Enum):
    CAD_QUALITY = "cad_quality"
    REASONING = "reasoning"
    PLANNING = "planning"
    RETRIEVAL = "retrieval"
    SIMULATION = "simulation"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_EFFICIENCY = "memory_efficiency"
    GPU_UTILIZATION = "gpu_utilization"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    benchmark_id: str
    name: str
    category: BenchmarkCategory
    description: str
    function: Callable[[], dict[str, float]]
    iterations: int = 10
    warmup_iterations: int = 3
    timeout: float = 300.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    result_id: str
    benchmark_id: str
    category: BenchmarkCategory
    metrics: dict[str, float] = field(default_factory=dict)  # metric -> value
    statistics: dict[str, dict[str, float]] = field(
        default_factory=dict
    )  # metric -> {mean, std, min, max, median}
    iterations_completed: int = 0
    total_time: float = 0.0
    status: str = "completed"
    error: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class BenchmarkSuite:
    """A suite of benchmarks."""

    suite_id: str
    name: str
    benchmarks: list[BenchmarkConfig] = field(default_factory=list)
    results: list[BenchmarkResult] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class SystemBenchmark:
    """Comprehensive system benchmarking."""

    def __init__(self):
        self._benchmarks: dict[str, BenchmarkConfig] = {}
        self._suites: dict[str, BenchmarkSuite] = {}
        self._results: dict[str, BenchmarkResult] = {}
        self._lock = RLock()

    def register_benchmark(
        self,
        benchmark_id: str,
        name: str,
        category: BenchmarkCategory,
        description: str,
        function: Callable[[], dict[str, float]],
        iterations: int = 10,
        warmup_iterations: int = 3,
        timeout: float = 300.0,
    ) -> BenchmarkConfig:
        """Register a benchmark."""
        config = BenchmarkConfig(
            benchmark_id=benchmark_id,
            name=name,
            category=category,
            description=description,
            function=function,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
            timeout=timeout,
        )
        with self._lock:
            self._benchmarks[benchmark_id] = config
        return config

    def create_suite(self, name: str, benchmark_ids: list[str]) -> BenchmarkSuite:
        """Create a benchmark suite."""
        benchmarks = [self._benchmarks[bid] for bid in benchmark_ids if bid in self._benchmarks]
        suite = BenchmarkSuite(
            suite_id=str(uuid.uuid4()),
            name=name,
            benchmarks=benchmarks,
        )
        with self._lock:
            self._suites[suite.suite_id] = suite
        return suite

    def run_benchmark(self, benchmark_id: str) -> BenchmarkResult:
        """Run a single benchmark."""
        config = self._benchmarks.get(benchmark_id)
        if not config:
            raise ValueError(f"Benchmark {benchmark_id} not found")

        # Warmup
        for _ in range(config.warmup_iterations):
            with contextlib.suppress(Exception):
                config.function()

        # Actual runs
        all_metrics: dict[str, list[float]] = {}
        start_time = time.time()
        completed = 0

        for _ in range(config.iterations):
            time.time()
            try:
                metrics = config.function()
                for k, v in metrics.items():
                    if k not in all_metrics:
                        all_metrics[k] = []
                    all_metrics[k].append(v)
                completed += 1
            except Exception:
                pass
            # Check timeout
            if time.time() - start_time > config.timeout:
                break

        total_time = time.time() - start_time

        # Compute statistics
        stats = {}
        for metric, values in all_metrics.items():
            if values:
                stats[metric] = {
                    "mean": statistics.mean(values),
                    "stdev": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "median": statistics.median(values),
                }

        result = BenchmarkResult(
            result_id=str(uuid.uuid4()),
            benchmark_id=benchmark_id,
            category=config.category,
            metrics={k: v["mean"] for k, v in stats.items()},
            statistics=stats,
            iterations_completed=completed,
            total_time=total_time,
        )

        with self._lock:
            self._results[result.result_id] = result

        return result

    def run_suite(self, suite_id: str) -> BenchmarkSuite:
        """Run all benchmarks in a suite."""
        suite = self._suites.get(suite_id)
        if not suite:
            raise ValueError(f"Suite {suite_id} not found")

        for benchmark in suite.benchmarks:
            result = self.run_benchmark(benchmark.benchmark_id)
            suite.results.append(result)

        return suite

    def get_result(self, result_id: str) -> BenchmarkResult | None:
        with self._lock:
            return self._results.get(result_id)

    def list_results(self, category: BenchmarkCategory | None = None) -> list[BenchmarkResult]:
        with self._lock:
            results = list(self._results.values())
            if category:
                results = [r for r in results if r.category == category]
            return results

    def compare_results(self, result_ids: list[str]) -> dict[str, Any]:
        """Compare multiple benchmark results."""
        results = [self._results[rid] for rid in result_ids if rid in self._results]
        if not results:
            return {}

        all_metrics: set[str] = set()
        for r in results:
            all_metrics.update(r.metrics.keys())

        comparison = {}
        for metric in all_metrics:
            comparison[metric] = {r.benchmark_id: r.metrics.get(metric, 0) for r in results}

        return comparison
