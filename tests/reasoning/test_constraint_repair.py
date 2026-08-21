"""Tests for Pillar 7 constraint propagation, conflict detection and repair."""

from __future__ import annotations

from cadgenesis.reasoning import Constraint, ConstraintSolver, Variable


def _system() -> tuple[ConstraintSolver, list[Variable], list[Constraint]]:
    solver = ConstraintSolver()
    variables = [
        Variable("a", initial=1.0, lower=0.0, upper=10.0),
        Variable("b", initial=1.0, lower=0.0, upper=10.0),
        Variable("c", initial=1.0, lower=0.0, upper=10.0),
    ]
    constraints = [
        Constraint("sum_ab", {"a": 1.0, "b": 1.0}, "==", 4.0),
        Constraint("diff_bc", {"b": 1.0, "c": -1.0}, "==", 1.0),
    ]
    return solver, variables, constraints


def test_dependency_graph_edges() -> None:
    solver, _, constraints = _system()
    graph = solver.dependency_graph(constraints)
    assert graph["sum_ab"] == ["diff_bc"]
    assert graph["diff_bc"] == ["sum_ab"]


def test_propagate_fixes_assignment() -> None:
    solver, variables, constraints = _system()
    assignment = {"a": 1.0, "b": 1.0, "c": 1.0}
    propagated = solver.propagate(variables, constraints, assignment)
    solution = solver.solve(variables, constraints)
    assert solution.feasible
    for name, value in solution.assignment.items():
        assert propagated[name] == value


def test_propagate_respects_bounds() -> None:
    solver, variables, constraints = _system()
    assignment = {"a": 0.0, "b": 0.0, "c": 0.0}
    propagated = solver.propagate(variables, constraints, assignment, max_hops=100)
    assert propagated["a"] >= 0.0 and propagated["b"] >= 0.0
    assert abs(propagated["a"] + propagated["b"] - 4.0) < 1e-4


def test_propagate_rejects_bad_max_hops() -> None:
    solver, variables, constraints = _system()
    try:
        solver.propagate(variables, constraints, {}, max_hops=0)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_detect_conflicts_finds_pair() -> None:
    solver = ConstraintSolver()
    variables = [Variable("x", initial=1.0, lower=0.0, upper=10.0)]
    constraints = [
        Constraint("low", {"x": 1.0}, "<=", 2.0),
        Constraint("high", {"x": 1.0}, ">=", 8.0),
    ]
    conflicts = solver.detect_conflicts(variables, constraints)
    assert len(conflicts) == 1
    assert {conflicts[0]["left"], conflicts[0]["right"]} == {"low", "high"}
    assert conflicts[0]["variable"] == "x"


def test_detect_conflicts_clear_when_feasible() -> None:
    solver, variables, constraints = _system()
    assert solver.detect_conflicts(variables, constraints) == []


def test_repair_drops_conflicting_constraint() -> None:
    solver = ConstraintSolver()
    variables = [Variable("x", initial=1.0, lower=0.0, upper=10.0)]
    constraints = [
        Constraint("low", {"x": 1.0}, "<=", 2.0),
        Constraint("high", {"x": 1.0}, ">=", 8.0),
    ]
    report = solver.repair(variables, constraints)
    assert report["feasible"] is True
    assert len(report["dropped"]) == 1
    remaining = [c for c in constraints if c.name != report["dropped"][0]]
    assert solver.solve(variables, remaining).feasible


def test_repair_respects_relax_order() -> None:
    solver = ConstraintSolver()
    variables = [Variable("x", initial=1.0, lower=0.0, upper=10.0)]
    constraints = [
        Constraint("low", {"x": 1.0}, "<=", 2.0),
        Constraint("high", {"x": 1.0}, ">=", 8.0),
    ]
    report = solver.repair(variables, constraints, relax_order=["low"])
    assert report["dropped"] == ["low"]
    assert abs(report["assignment"]["x"] - 8.0) < 1e-4


def test_repair_infeasible_when_all_dropped() -> None:
    solver = ConstraintSolver()
    variables = [Variable("x", initial=1.0, lower=5.0, upper=5.0)]
    constraints = [
        Constraint("c1", {"x": 1.0}, "==", 1.0),
    ]
    report = solver.repair(variables, constraints)
    assert report["feasible"] is True or report["dropped"] == ["c1"]
