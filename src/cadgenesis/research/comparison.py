"""
cadgenesis.research.comparison
==============================
Model comparison framework for CADGenesis-LM research infrastructure.

Compares architectures, checkpoints, datasets and adapters side by side
on a shared metric set, with statistical validation (CIs + significance).

``ComparisonReport`` holds per-variant metric tables and an optional
significance matrix; ``ModelComparator`` orchestrates runs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.research.stats import HypothesisTestResult, mean_ci_normal, welch_t_test

logger = logging.getLogger("cadgenesis.research.comparison")

VariantRunner = Callable[[str, Mapping[str, Any]], Mapping[str, float]]

COMPARISON_DIMENSIONS = ("architecture", "checkpoint", "dataset", "adapter")


@dataclass
class VariantResult:
    """Metrics for one variant, with CI when repeated runs exist."""

    variant: str
    dimension: str
    metrics: dict[str, float]
    repeats: list[dict[str, float]] = field(default_factory=list)
    confidence_intervals: dict[str, Any] = field(default_factory=dict)

    def compute_intervals(self, level: float = 0.95) -> dict[str, Any]:
        """Normal-approx CI over repeats for each metric."""
        if len(self.repeats) < 2:
            return {}
        keys = self.repeats[0].keys()
        self.confidence_intervals = {}
        for key in keys:
            values = [r[key] for r in self.repeats if key in r]
            if len(values) < 2:
                continue
            ci = mean_ci_normal(values, level=level)
            self.confidence_intervals[key] = {
                "lower": ci.lower,
                "upper": ci.upper,
                "estimate": ci.estimate,
            }
        return self.confidence_intervals

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "dimension": self.dimension,
            "metrics": dict(self.metrics),
            "repeats": [dict(r) for r in self.repeats],
            "confidence_intervals": dict(self.confidence_intervals),
        }


@dataclass
class ComparisonReport:
    """Full comparison: variants + significance matrix."""

    dimension: str
    variants: list[VariantResult] = field(default_factory=list)
    significance: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, result: VariantResult) -> None:
        self.variants.append(result)

    def rankings(self, metric: str, minimize: bool = True) -> list[tuple[str, float]]:
        """Variants ranked by a metric (best first)."""
        scored = [
            (v.variant, v.metrics.get(metric, float("inf")))
            for v in self.variants
            if metric in v.metrics
        ]
        return sorted(scored, key=lambda pair: pair[1], reverse=not minimize)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "variants": [v.to_dict() for v in self.variants],
            "significance": dict(self.significance),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ModelComparator:
    """Runs variants across a dimension and validates differences."""

    def __init__(self, runner: VariantRunner, repeats: int = 1, seed: int = 42) -> None:
        self.runner = runner
        self.repeats = max(1, repeats)
        self.seed = seed

    def compare(
        self,
        dimension: str,
        variants: Iterable[str],
        configs: Mapping[str, Mapping[str, Any]] | None = None,
        metric: str = "loss",
        minimize: bool = True,
        alpha: float = 0.05,
    ) -> ComparisonReport:
        if dimension not in COMPARISON_DIMENSIONS:
            raise ValueError(f"unknown dimension {dimension!r}; expected {COMPARISON_DIMENSIONS}")
        report = ComparisonReport(dimension=dimension)
        results: dict[str, VariantResult] = {}
        for variant in variants:
            config = dict((configs or {}).get(variant, {}))
            repeats: list[dict[str, float]] = []
            for _ in range(self.repeats):
                runs = {k: float(v) for k, v in self.runner(variant, config).items()}
                repeats.append(runs)
            aggregated = {
                key: sum(r[key] for r in repeats if key in r)
                / len([r for r in repeats if key in r])
                for key in repeats[0]
            }
            result = VariantResult(
                variant=variant, dimension=dimension, metrics=aggregated, repeats=repeats
            )
            result.compute_intervals()
            results[variant] = result
            report.add(result)
            logger.info(
                "compared %s: %s -> %s",
                dimension,
                variant,
                {k: round(v, 4) for k, v in aggregated.items()},
            )
        report.significance = self._significance_matrix(results, metric, minimize, alpha)
        return report

    @staticmethod
    def _significance_matrix(
        results: Mapping[str, VariantResult],
        metric: str,
        minimize: bool,
        alpha: float,
    ) -> dict[str, dict[str, Any]]:
        matrix: dict[str, dict[str, Any]] = {}
        names = list(results)
        for left in names:
            matrix[left] = {}
            for right in names:
                if left == right:
                    matrix[left][right] = {"comparison": "self"}
                    continue
                a = [r[metric] for r in results[left].repeats if metric in r]
                b = [r[metric] for r in results[right].repeats if metric in r]
                if len(a) < 2 or len(b) < 2:
                    matrix[left][right] = {"comparison": "insufficient_repeats"}
                    continue
                test: HypothesisTestResult = welch_t_test(a, b, alpha=alpha)
                better = (
                    (test.statistic < 0 if minimize else test.statistic > 0)
                    if test.significant
                    else "tie"
                )
                matrix[left][right] = {
                    "t_statistic": round(test.statistic, 4),
                    "p_value": round(test.p_value, 6),
                    "significant": test.significant,
                    "better": better,
                }
        return matrix


__all__ = ["COMPARISON_DIMENSIONS", "ComparisonReport", "ModelComparator", "VariantResult"]
