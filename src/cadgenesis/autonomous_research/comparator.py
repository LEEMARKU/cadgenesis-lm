"""
Architecture Comparator - Compare transformers, adapters, memory systems, reasoning engines,
multimodal encoders.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


@dataclass
class ComparisonConfig:
    """Configuration for architecture comparison."""

    comparison_id: str
    name: str
    baseline: dict[str, Any]  # architecture config
    candidates: list[dict[str, Any]]  # list of architecture configs
    metrics: list[str]
    benchmark_fn: Callable[[dict[str, Any]], dict[str, float]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Result of architecture comparison."""

    comparison_id: str
    baseline_metrics: dict[str, float]
    candidate_results: list[dict[str, Any]]  # each has config, metrics, improvements
    best_candidate: int | None = None  # index
    summary: str = ""
    created_at: float = field(default_factory=time.time)


class ArchitectureComparator:
    """Compares different architectures against a baseline."""

    def __init__(self):
        self._comparisons: dict[str, ComparisonResult] = {}
        self._lock = RLock()

    def compare(
        self,
        config: ComparisonConfig,
    ) -> ComparisonResult:
        """Run comparison between baseline and candidates."""
        # Evaluate baseline
        baseline_metrics = config.benchmark_fn(config.baseline)

        # Evaluate candidates
        candidate_results: list[dict[str, Any]] = []
        for i, candidate in enumerate(config.candidates):
            try:
                metrics = config.benchmark_fn(candidate)
                improvements = {}
                for metric in config.metrics:
                    if (
                        metric in baseline_metrics
                        and metric in metrics
                        and baseline_metrics[metric] != 0
                    ):
                        improvements[metric] = (metrics[metric] - baseline_metrics[metric]) / abs(
                            baseline_metrics[metric]
                        )
                    else:
                        improvements[metric] = 0

                candidate_results.append(
                    {
                        "index": i,
                        "config": candidate,
                        "metrics": metrics,
                        "improvements": improvements,
                    }
                )
            except Exception as e:
                candidate_results.append(
                    {
                        "index": i,
                        "config": candidate,
                        "metrics": {},
                        "improvements": {},
                        "error": str(e),
                    }
                )

        # Find best candidate
        best_idx = None
        best_score = -float("inf")
        for cr in candidate_results:
            if "error" not in cr:
                score = sum(cr["improvements"].values())
                if score > best_score:
                    best_score = score
                    best_idx = cr["index"]

        result = ComparisonResult(
            comparison_id=config.comparison_id,
            baseline_metrics=baseline_metrics,
            candidate_results=candidate_results,
            best_candidate=best_idx,
        )

        # Generate summary
        n_improved = sum(
            1
            for cr in candidate_results
            if "error" not in cr and any(v > 0 for v in cr["improvements"].values())
        )
        result.summary = (
            f"Compared {len(config.candidates)} candidates. {n_improved} "
            f"showed improvements. Best: candidate {best_idx}."
        )

        with self._lock:
            self._comparisons[config.comparison_id] = result

        return result

    def get_comparison(self, comparison_id: str) -> ComparisonResult | None:
        with self._lock:
            return self._comparisons.get(comparison_id)

    def list_comparisons(self) -> list[ComparisonResult]:
        with self._lock:
            return list(self._comparisons.values())
