"""tests/reasoning/test_validator.py
===================================
Unit tests for cadgenesis.reasoning.validator (integration orchestrator).
"""

from __future__ import annotations

import pytest

from cadgenesis.reasoning.constraint_solver import Constraint, ConstraintSolver, Variable
from cadgenesis.reasoning.geometry_reasoner import GeometryReasoner, Primitive
from cadgenesis.reasoning.manufacturing_rules import ManufacturingRules
from cadgenesis.reasoning.planner import CADPlan, PlanningStep
from cadgenesis.reasoning.rule_engine import RuleEngine, make_rule
from cadgenesis.reasoning.topology import TopologyAnalyzer
from cadgenesis.reasoning.validator import CheckResult, DesignValidator, ValidationReport


def _build_validator() -> DesignValidator:
    rules = RuleEngine(
        [
            make_rule(
                "wall_minimum",
                lambda ctx: ctx.get("wall_thickness", 0) >= 0.8,
                severity="info",
            ),
            make_rule(
                "draft_angle_ok",
                lambda ctx: ctx.get("draft_angle", 0) >= 1.0,
                severity="error",
                meta={"recommendation": "Add draft."},
            ),
        ]
    )
    return DesignValidator(
        rule_engine=rules,
        manufacturing_rules=ManufacturingRules(),
        geometry_reasoner=GeometryReasoner,
        constraint_solver=ConstraintSolver(),
        topology_analyzer=TopologyAnalyzer,
    )


class TestCheckResult:
    def test_invalid_category(self):
        with pytest.raises(ValueError):
            CheckResult("nope", "x", True)


class TestValidationReport:
    def test_empty_report_passes(self):
        report = ValidationReport()
        assert report.passed

    def test_counts(self):
        report = ValidationReport(
            results=[
                CheckResult("rule", "r1", True),
                CheckResult("rule", "r2", False, severity="warning"),
                CheckResult("geometry", "g1", False, severity="error"),
            ]
        )
        assert not report.passed
        assert len(report.errors) == 1
        assert len(report.warnings) == 1
        assert set(report.by_category()) == {"rule", "geometry"}
        assert report.summary()["total"] == 3

    def test_to_dict(self):
        report = ValidationReport(results=[CheckResult("rule", "r", True)])
        data = report.to_dict()
        assert data["passed"] is True
        assert len(data["results"]) == 1


class TestDesignValidator:
    def test_empty_context_passes(self):
        validator = _build_validator()
        report = validator.validate({})
        assert isinstance(report, ValidationReport)

    def test_rule_check(self):
        validator = _build_validator()
        report = validator.validate({"wall_thickness": 1.2, "draft_angle": 0.5})
        rule_results = [r for r in report.results if r.category == "rule"]
        assert {r.name for r in rule_results} == {"wall_minimum"}
        assert rule_results[0].passed

    def test_rule_error_check(self):
        validator = _build_validator()
        report = validator.validate({"wall_thickness": 0.4, "draft_angle": 0.5})
        rule_results = [r for r in report.results if r.category == "rule"]
        # neither rule triggers: wall too thin, draft too small
        assert rule_results == []

    def test_geometry_check(self):
        validator = _build_validator()
        primitives = [
            Primitive("box", {"length": 2, "width": 2, "height": 2}),
            Primitive("box", {"length": 2, "width": 2, "height": 2}, position=(5, 0, 0)),
        ]
        report = validator.validate({"primitives": primitives})
        geometry_results = [r for r in report.results if r.category == "geometry"]
        assert all(r.passed for r in geometry_results)

    def test_interference_detected(self):
        validator = _build_validator()
        a = Primitive("box", {"length": 4, "width": 4, "height": 4})
        b = Primitive("box", {"length": 4, "width": 4, "height": 4}, position=(2, 0, 0))
        report = validator.validate({"interference_pairs": [(a, b)]})
        interferences = [r for r in report.results if r.category == "geometry"]
        assert interferences and not interferences[0].passed

    def test_constraint_check(self):
        validator = _build_validator()
        variables = [Variable("w"), Variable("d")]
        constraints = [Constraint("c", {"w": 1.0, "d": -2.0}, "==", 0.0)]
        report = validator.validate({"constraint_variables": variables, "constraints": constraints})
        constraint_results = [r for r in report.results if r.category == "constraint"]
        assert constraint_results and constraint_results[0].passed

    def test_manufacturing_check(self):
        validator = _build_validator()
        report = validator.validate(
            {"part": {"min_wall_thickness": 0.3, "processes": ["machining"]}}
        )
        mfg_results = [r for r in report.results if r.category == "manufacturing"]
        assert mfg_results and not mfg_results[0].passed

    def test_topology_check(self):
        validator = _build_validator()
        report = validator.validate(
            {"topology": {"vertices": 8, "edges": 12, "faces": 6, "solids": 1}}
        )
        topology_results = [r for r in report.results if r.category == "topology"]
        assert topology_results and topology_results[0].passed

    def test_custom_check(self):
        validator = _build_validator()

        def custom_check(ctx):
            return [CheckResult("custom", "always_ok", True)]

        validator.add_check(custom_check)
        report = validator.validate({})
        assert any(r.category == "custom" for r in report.results)

    def test_validate_plan(self):
        validator = _build_validator()
        plan = CADPlan(goal="x")
        plan.add_step(PlanningStep("a", "model"))
        report = validator.validate_plan(plan)
        assert report.passed
