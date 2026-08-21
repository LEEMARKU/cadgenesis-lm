"""Tests for Pillar 7 symbolic planner."""

from __future__ import annotations

import pytest

from cadgenesis.reasoning import (
    PlanningOperator,
    SymbolicPlanner,
    TaskPlanner,
)


def _build_planner() -> SymbolicPlanner:
    return SymbolicPlanner(
        [
            PlanningOperator(
                "sketch",
                precondition=lambda s: not s.get("sketched"),
                effect=lambda s: {**s, "sketched": True},
                action="sketch",
            ),
            PlanningOperator(
                "model",
                precondition=lambda s: s.get("sketched") and not s.get("modeled"),
                effect=lambda s: {**s, "modeled": True},
                action="model",
            ),
            PlanningOperator(
                "constrain",
                precondition=lambda s: s.get("modeled") and not s.get("constrained"),
                effect=lambda s: {**s, "constrained": True},
                action="constrain",
            ),
            PlanningOperator(
                "validate",
                precondition=lambda s: s.get("constrained") and not s.get("validated"),
                effect=lambda s: {**s, "validated": True},
                action="validate",
            ),
        ]
    )


def test_planner_solves_goal() -> None:
    planner = _build_planner()
    plan = planner.plan("build", {}, lambda s: s.get("validated"))
    assert plan.solved
    assert plan.operators == ["sketch", "model", "constrain", "validate"]
    assert plan.cost == pytest.approx(4.0)
    assert plan.explored >= 4


def test_planner_no_solution() -> None:
    planner = SymbolicPlanner([PlanningOperator("a", lambda s: False, lambda s: s)])
    plan = planner.plan("impossible", {}, lambda s: True, max_depth=3)
    assert not plan.solved
    assert plan.operators == []


def test_planner_rejects_bad_args() -> None:
    planner = _build_planner()
    with pytest.raises(ValueError):
        planner.plan("", {}, lambda s: True)
    with pytest.raises(ValueError):
        planner.plan("x", {}, lambda s: True, max_depth=0)


def test_operator_validation() -> None:
    with pytest.raises(ValueError):
        PlanningOperator("", lambda s: True, lambda s: s)
    with pytest.raises(ValueError):
        PlanningOperator("x", lambda s: True, lambda s: s, cost=-1.0)
    with pytest.raises(ValueError):
        PlanningOperator("x", lambda s: True, lambda s: s, action="teleport")
    with pytest.raises(TypeError):
        PlanningOperator("x", "not callable", lambda s: s)


def test_planner_registry() -> None:
    planner = _build_planner()
    assert len(planner) == 4
    assert planner.get("model") is not None
    with pytest.raises(ValueError):
        planner.register(PlanningOperator("model", lambda s: True, lambda s: s))


def test_planner_state_chain() -> None:
    planner = _build_planner()
    plan = planner.plan("build", {"sketched": True}, lambda s: s.get("validated"))
    assert plan.solved
    assert plan.operators == ["model", "constrain", "validate"]
    assert len(plan.states) == 4
    assert plan.states[0]["sketched"] is True


def test_plan_dependency_graph() -> None:
    planner = _build_planner()
    plan = planner.plan("build", {}, lambda s: s.get("validated"))
    edges = plan.dependency_graph()
    assert isinstance(edges, list)
    assert all(isinstance(e, tuple) and len(e) == 2 for e in edges)


def test_plan_to_cad_plan() -> None:
    planner = _build_planner()
    plan = planner.plan("build", {}, lambda s: s.get("validated"))
    cad_plan = plan.to_cad_plan("build")
    assert cad_plan.goal == "build"
    assert cad_plan.step_count == 4
    assert not cad_plan.validate()
    order = cad_plan.topological_order()
    assert order == ["step1", "step2", "step3", "step4"]


def test_decompose_returns_chain() -> None:
    planner = _build_planner()
    chain = planner.decompose("build", lambda s: s.get("validated"), {})
    assert chain == ["sketch", "model", "constrain", "validate"]


def test_plan_summary_and_execution_order() -> None:
    planner = _build_planner()
    plan = planner.plan("build", {}, lambda s: s.get("validated"))
    summary = plan.summary()
    assert summary["solved"] is True
    assert summary["steps"] == 4
    assert plan.execution_order() == plan.operators


def test_task_planner_still_works() -> None:
    planner = TaskPlanner()
    plan = planner.create_plan("box")
    assert plan.step_count >= 3
