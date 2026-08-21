"""Tests for Pillar 8 optimization + cost estimation."""

from __future__ import annotations

from cadgenesis.execution import (
    OBJECTIVES,
    CostEstimator,
    OptimizationEngine,
    OptimizationReport,
)


class TestOptimization:
    def test_objectives_exposed(self) -> None:
        for objective in ("weight", "material", "complexity", "print_time", "cost", "structural"):
            assert objective in OBJECTIVES

    def test_optimize_returns_scores(self) -> None:
        report = OptimizationEngine().optimize(
            {"name": "part", "feature_count": 4, "weight_kg": 2.5}
        )
        assert isinstance(report, OptimizationReport)
        assert report.passed
        assert report.scores
        assert report.best_index is not None

    def test_optimize_scores_span_objectives(self) -> None:
        report = OptimizationEngine().optimize(
            {"name": "part", "feature_count": 4}, objectives=["weight", "cost"]
        )
        names = {s["name"] for s in report.scores}
        assert {"part-weight", "part-cost"} <= names

    def test_unknown_objective_rejected(self) -> None:
        report = OptimizationEngine().optimize({"name": "part"}, objectives=["quantum"])
        assert not report.passed
        assert "no recognized objectives" in report.messages

    def test_suggest_lean_design(self) -> None:
        suggestions = OptimizationEngine().suggest({"name": "lean"})
        assert suggestions == ["design is already lean; no immediate structural savings found"]

    def test_suggest_wall_reduction(self) -> None:
        suggestions = OptimizationEngine().suggest({"wall_thickness_mm": 8.0})
        assert any("wall" in s for s in suggestions)

    def test_summary(self) -> None:
        assert "objectives" in OptimizationEngine().summary()


class TestCostEstimator:
    def test_machining_cost(self) -> None:
        breakdown = CostEstimator().estimate(
            {
                "processes": ["machining"],
                "feature_count": 2,
                "volume_m3": 150e-9,
            }
        )
        assert breakdown.total > 0
        assert breakdown.machining_usd > 0

    def test_printing_cost(self) -> None:
        breakdown = CostEstimator().estimate({"processes": ["fdm_print"], "volume_m3": 150e-9})
        assert breakdown.printing_usd > 0

    def test_no_processes(self) -> None:
        breakdown = CostEstimator().estimate({})
        assert breakdown.total == 0.0

    def test_summary_and_to_dict(self) -> None:
        estimator = CostEstimator()
        breakdown = estimator.estimate({"processes": ["machining"]})
        summary = estimator.summary()
        assert "total" in breakdown.to_dict()
        assert "materials" in summary
