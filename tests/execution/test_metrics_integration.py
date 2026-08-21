"""Tests for execution evaluation metrics + integration wiring."""

from __future__ import annotations

import pytest

from cadgenesis.cad.integration.execution_bridge import ExecutionBridge
from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.evaluation import (
    confidence_agreement,
    cost_error,
    geometry_validity_rate,
    manufacturability_rate,
    repair_rate,
    run_execution_benchmark,
    safety_factor_pass_rate,
    simulation_pass_rate,
)
from cadgenesis.execution import CADExecutionEngine, CADExecutionResult


def _result(**kwargs) -> CADExecutionResult:
    defaults = dict(
        is_valid_geometry=True,
        is_manufacturable=True,
        safety_factor=2.0,
        confidence_score=0.9,
        estimated_cost_usd=10.0,
        simulation_report={"passed": True},
        repair_report={"attempted": True, "fixed": True},
        errors=[],
        suggestions=[],
    )
    defaults.update(kwargs)
    result = CADExecutionResult()
    for key, value in defaults.items():
        setattr(result, key, value)
    return result


class TestMetrics:
    def test_geometry_validity_rate(self) -> None:
        results = [_result(), _result(is_valid_geometry=False)]
        assert geometry_validity_rate(results) == 0.5
        assert geometry_validity_rate([]) == 0.0

    def test_manufacturability_rate(self) -> None:
        results = [_result(), _result(is_manufacturable=False)]
        assert manufacturability_rate(results) == 0.5

    def test_safety_factor_pass_rate(self) -> None:
        results = [_result(safety_factor=2.0), _result(safety_factor=1.0)]
        assert safety_factor_pass_rate(results) == 0.5
        assert safety_factor_pass_rate(results, required=0.5) == 1.0

    def test_simulation_pass_rate(self) -> None:
        results = [
            _result(simulation_report={"passed": True}),
            _result(simulation_report={"passed": False}),
        ]
        assert simulation_pass_rate(results) == 0.5

    def test_confidence_agreement(self) -> None:
        results = [_result(confidence_score=0.8), _result(is_valid_geometry=False)]
        assert confidence_agreement(results) == 0.8

    def test_repair_rate(self) -> None:
        results = [
            _result(repair_report={"attempted": True}),
            _result(repair_report={"attempted": False}),
        ]
        assert repair_rate(results) == 0.5

    def test_cost_error(self) -> None:
        assert cost_error([10.0, 20.0], [10.0, 40.0]) == pytest.approx(0.25)
        assert cost_error([], []) == 0.0

    def test_benchmark_report(self) -> None:
        report = run_execution_benchmark(
            [_result(), _result(is_valid_geometry=False)],
            actual_costs=[10.0, 20.0],
        )
        assert report["geometry_validity_rate"] == 0.5
        assert report["checks"]["results"] == 2
        assert "cost_error" in report


class TestIntegrationWiring:
    def test_bridge_execute_design(self) -> None:
        bridge = ExecutionBridge()
        result = bridge.execute_design(
            {"mesh": Mesh.box().to_dict(), "name": "box", "volume_m3": 150e-9}
        )
        assert result.is_valid_geometry
        summary = bridge.summary(result)
        assert summary["is_valid_geometry"] is True

    def test_bridge_execute_design_export(self, tmp_path) -> None:
        bridge = ExecutionBridge()
        bridge.execute_design(
            {"mesh": Mesh.box().to_dict(), "name": "box"},
            export_fmt="stl",
            export_path=str(tmp_path / "bridge.stl"),
        )
        assert (tmp_path / "bridge.stl").exists()

    def test_agents_execution_adapter(self) -> None:
        from cadgenesis.agents.integration import ExecutionAdapter

        adapter = ExecutionAdapter()
        result = adapter.execute(["PRIM_BOX"])
        assert result.is_valid_geometry
        design_result = adapter.execute_design({"mesh": Mesh.box().to_dict(), "name": "box"})
        assert design_result.is_manufacturable

    def test_agents_init_exports_adapter(self) -> None:
        import cadgenesis.agents

        assert hasattr(cadgenesis.agents, "ExecutionAdapter")

    def test_pipeline_flag_off(self) -> None:
        from cadgenesis.cad.integration.pipeline import CADIntelligencePipeline

        result = CADIntelligencePipeline().run({"material": "ABS"}, name="x")
        assert result.execution == {}

    def test_pipeline_flag_on(self) -> None:
        from cadgenesis.cad.integration.pipeline import CADIntelligencePipeline

        pipeline = CADIntelligencePipeline(execution=True)
        result = pipeline.run({"mesh": Mesh.box().to_dict(), "name": "box"}, name="x")
        assert result.execution["is_valid_geometry"] is True
        assert "is_manufacturable" in result.execution
        assert "execution" in result.summary()


class TestWorldModelPlannerWiring:
    def test_planner_without_engine_unchanged(self) -> None:
        from cadgenesis.world_model import WorldModelSystem

        wm = WorldModelSystem()
        plan = wm.reason("plan", goal="assemble a bracket")
        result = wm.reason("execute_plan", plan=plan)
        assert result.all_passed

    def test_planner_with_engine_validates(self) -> None:
        from cadgenesis.reasoning.planner import TaskPlanner
        from cadgenesis.world_model.objects import ObjectGraph
        from cadgenesis.world_model.planning import WorldModelPlanner

        planner = WorldModelPlanner(execution=CADExecutionEngine())
        plan = TaskPlanner().create_plan("assemble a bracket")
        result = planner.execute(plan, ObjectGraph())
        assert result.all_passed
