"""tests/reasoning/test_constraint_solver.py
===========================================
Unit tests for cadgenesis.reasoning.constraint_solver.
"""

from __future__ import annotations

import pytest

from cadgenesis.reasoning.constraint_solver import (
    Constraint,
    ConstraintSolver,
    Variable,
)


class TestVariable:
    def test_clamps_initial(self):
        var = Variable("x", initial=10.0, lower=0.0, upper=5.0)
        assert var.initial == 5.0

    def test_empty_range_rejected(self):
        with pytest.raises(ValueError):
            Variable("x", lower=5.0, upper=0.0)

    def test_clamp(self):
        var = Variable("x", lower=0.0, upper=10.0)
        assert var.clamp(-5.0) == 0.0
        assert var.clamp(15.0) == 10.0
        assert var.clamp(4.0) == 4.0


class TestConstraint:
    def test_validation(self):
        with pytest.raises(ValueError):
            Constraint("c", {"x": 1.0}, "!=", 0.0)
        with pytest.raises(ValueError):
            Constraint("c", {}, "==", 0.0)

    def test_residual(self):
        c = Constraint("c", {"x": 2.0, "y": 1.0}, "==", 10.0)
        assert c.residual({"x": 3.0, "y": 2.0}) == pytest.approx(-2.0)
        assert c.satisfied({"x": 3.0, "y": 4.0})

    def test_inequality_residual(self):
        c = Constraint("c", {"x": 1.0}, "<=", 5.0)
        assert c.satisfied({"x": 3.0})
        assert not c.satisfied({"x": 7.0})


class TestConstraintSolver:
    def test_equality_solved(self):
        solver = ConstraintSolver()
        variables = [Variable("x"), Variable("y")]
        constraints = [
            Constraint("c1", {"x": 1.0, "y": 1.0}, "==", 10.0),
            Constraint("c2", {"x": 1.0, "y": -1.0}, "==", 2.0),
        ]
        solution = solver.solve(variables, constraints)
        assert solution.feasible
        assert solution.assignment["x"] == pytest.approx(6.0, abs=1e-4)
        assert solution.assignment["y"] == pytest.approx(4.0, abs=1e-4)

    def test_bounds_respected(self):
        solver = ConstraintSolver()
        variables = [Variable("x", lower=0.0, upper=5.0)]
        constraints = [Constraint("c", {"x": 1.0}, "==", 100.0)]
        solution = solver.solve(variables, constraints)
        assert not solution.feasible
        assert solution.assignment["x"] == pytest.approx(5.0)

    def test_inequality(self):
        solver = ConstraintSolver()
        variables = [Variable("x", initial=10.0)]
        constraints = [Constraint("c", {"x": 1.0}, "<=", 5.0)]
        solution = solver.solve(variables, constraints)
        assert solution.feasible
        assert solution.assignment["x"] <= 5.0 + 1e-4

    def test_solved_within_iterations(self):
        solver = ConstraintSolver(max_iterations=500)
        variables = [Variable("w"), Variable("d")]
        constraints = [Constraint("c", {"w": 1.0, "d": -2.0}, "==", 0.0)]
        solution = solver.solve(variables, constraints)
        assert solution.feasible
        assert solution.iterations <= 500

    def test_empty_variables(self):
        solver = ConstraintSolver()
        assert solver.solve([], []).feasible

    def test_unknown_variable_rejected(self):
        solver = ConstraintSolver()
        with pytest.raises(KeyError):
            solver.solve([Variable("x")], [Constraint("c", {"y": 1.0}, "==", 0.0)])

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            ConstraintSolver(tolerance=0)
        with pytest.raises(ValueError):
            ConstraintSolver(max_iterations=0)

    def test_check_consistency(self):
        solver = ConstraintSolver()
        variables = [Variable("x", initial=5.0)]
        constraints = [Constraint("c", {"x": 1.0}, ">=", 3.0)]
        assert solver.check_consistency(variables, constraints)
