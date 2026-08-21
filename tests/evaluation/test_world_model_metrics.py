"""tests/evaluation/test_world_model_metrics.py
================================================
Unit tests for the Pillar-4 world-model metrics.
"""

from __future__ import annotations

from cadgenesis.evaluation import (
    accuracy,
    affordance_coverage_with,
    assembly_integrity,
    planning_success,
    run_world_benchmark,
    safety_margin,
)
from cadgenesis.world_model import AffordanceMapper, make_object


class TestMetrics:
    def test_accuracy(self):
        assert accuracy([True, False], [True, False]) == 1.0
        assert accuracy([], []) == 0.0

    def test_safety_margin(self):
        assert safety_margin([3.0, 2.0], [2.0, 2.0]) == 0.5

    def test_assembly_integrity(self):
        checks = [
            [{"passed": True}, {"passed": True}],
            [{"passed": True}, {"passed": False}],
        ]
        assert assembly_integrity(checks) == 0.5

    def test_affordance_coverage(self):
        mapper = AffordanceMapper()
        hole = make_object("hole", "h", {"radius": 3, "depth": 8})
        cylinder = make_object("cylinder", "c", {"radius": 5, "height": 20})
        cov = affordance_coverage_with(mapper, [hole, cylinder], [["insert"], ["rotate"]])
        assert cov == 1.0

    def test_planning_success(self):
        assert planning_success([{"all_passed": True}, {"all_passed": False}]) == 0.5

    def test_run_world_benchmark(self):
        report = run_world_benchmark(
            spatial_checks=[(True, True)],
            safety_checks=[(2.5, 2.0)],
            assembly_checks=[[{"passed": True}]],
            path_checks=[(False, False)],
            plan_outcomes=[{"all_passed": True}],
        )
        assert report["spatial_accuracy"] == 1.0
        assert report["assembly_integrity"] == 1.0
        assert report["planning_success"] == 1.0
