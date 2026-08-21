"""tests/reasoning/test_planner.py
================================
Unit tests for cadgenesis.reasoning.planner.
"""

from __future__ import annotations

import pytest

from cadgenesis.reasoning.planner import CADPlan, PlanningStep, TaskPlanner


class TestPlanningStep:
    def test_invalid_action(self):
        with pytest.raises(ValueError):
            PlanningStep(id="s", action="build")

    def test_empty_id(self):
        with pytest.raises(ValueError):
            PlanningStep(id="", action="model")


class TestCADPlan:
    @pytest.fixture
    def plan(self) -> CADPlan:
        plan = CADPlan(goal="box")
        plan.add_step(PlanningStep("a", "sketch", depends_on=[]))
        plan.add_step(PlanningStep("b", "model", depends_on=["a"]))
        plan.add_step(PlanningStep("c", "validate", depends_on=["b"]))
        return plan

    def test_order(self, plan):
        assert plan.topological_order() == ["a", "b", "c"]

    def test_not_cyclic(self, plan):
        assert not plan.is_cyclic()

    def test_cycle_detection(self):
        plan = CADPlan(goal="x")
        plan.add_step(PlanningStep("a", "model", depends_on=["b"]))
        plan.add_step(PlanningStep("b", "model", depends_on=["a"]))
        assert plan.is_cyclic()
        with pytest.raises(ValueError):
            plan.topological_order()

    def test_unknown_dependency_problem(self):
        plan = CADPlan(goal="x")
        plan.add_step(PlanningStep("a", "model", depends_on=["ghost"]))
        problems = plan.validate()
        assert any("ghost" in p for p in problems)

    def test_duplicate_step(self):
        plan = CADPlan(goal="x")
        plan.add_step(PlanningStep("a", "model"))
        with pytest.raises(ValueError):
            plan.add_step(PlanningStep("a", "model"))

    def test_critical_path(self, plan):
        assert plan.critical_path() == ["a", "b", "c"]

    def test_depends_on(self, plan):
        assert plan.depends_on("b") == ["a"]

    def test_round_trip(self, plan):
        rebuilt = CADPlan.from_dict(plan.to_dict())
        assert rebuilt.goal == "box"
        assert rebuilt.step_count == 3
        assert rebuilt.topological_order() == ["a", "b", "c"]


class TestTaskPlanner:
    def test_box_template(self):
        planner = TaskPlanner()
        plan = planner.create_plan("box")
        assert plan.step_count == 4
        assert plan.validate() == []
        assert plan.topological_order()[0] == "s1"

    def test_assembly_template(self):
        planner = TaskPlanner()
        plan = planner.create_plan("assembly")
        assert [s.action for s in plan.steps] == ["model", "model", "assemble", "simulate"]

    def test_unknown_goal_falls_back(self):
        planner = TaskPlanner()
        plan = planner.create_plan("custom widget")
        assert plan.step_count > 0

    def test_empty_goal_rejected(self):
        planner = TaskPlanner()
        with pytest.raises(ValueError):
            planner.create_plan("")

    def test_refine_without_rules_unchanged(self):
        planner = TaskPlanner()
        plan = planner.create_plan("box")
        refined = planner.refine(plan, {})
        assert refined is plan

    def test_refine_with_rules(self):
        from cadgenesis.reasoning.rule_engine import Rule, RuleEngine

        def add_step_action(ctx):
            plan = ctx["plan"]
            plan.add_step(PlanningStep("extra", "validate", depends_on=[]))
            return None

        engine = RuleEngine()
        engine.add_rule(Rule(name="add_check", condition=lambda c: True, action=add_step_action))
        planner = TaskPlanner(rules=engine)
        plan = planner.create_plan("box")
        refined = planner.refine(plan, {})
        assert refined.get_step("extra") is not None
