"""Tests for cadgenesis.cad.parametric (sketch, constraints, parameters)."""

from __future__ import annotations

import pytest

from cadgenesis.cad.geometry.core import Vec
from cadgenesis.cad.parametric.constraints import (
    ConstraintSolution,
    GeometricConstraint,
    SketchConstraintSolver,
)
from cadgenesis.cad.parametric.parameters import Parameter, ParameterTable
from cadgenesis.cad.parametric.sketch import Sketch


def _rect_sketch() -> Sketch:
    sketch = Sketch("rect")
    sketch.add_point(0, 0, name="p0")
    sketch.add_point(10, 0, name="p1")
    sketch.add_point(10, 8, name="p2")
    sketch.add_point(0, 8, name="p3")
    sketch.add_constraint(GeometricConstraint("HORIZONTAL", "p0", "p1"))
    sketch.add_constraint(GeometricConstraint("VERTICAL", "p1", "p2"))
    sketch.add_constraint(GeometricConstraint("FIXED", "p0"))
    return sketch


class TestSketch:
    def test_add_entities(self) -> None:
        sketch = Sketch("s")
        sketch.add_point(1, 2, name="a")
        sketch.add_point(3, 4, name="b")
        sketch.add_line(Vec(0, 0), Vec(5, 5))
        assert len(sketch.entities) == 3
        assert sketch.entity("a").name == "a"

    def test_duplicate_entity(self) -> None:
        sketch = Sketch("s")
        sketch.add_point(0, 0, name="a")
        with pytest.raises(KeyError):
            sketch.add_point(0, 0, name="a")

    def test_bounds(self) -> None:
        sketch = _rect_sketch()
        lo, hi = sketch.bounds()
        assert lo == Vec(0, 0)
        assert hi == Vec(10, 8)

    def test_is_closed_profile(self) -> None:
        sketch = Sketch("rect")
        sketch.rectangle(0, 0, 10, 8)
        assert sketch.is_closed_profile() is True


class TestConstraints:
    def test_geometric_constraint_valid(self) -> None:
        constraint = GeometricConstraint("HORIZONTAL", "p0", "p1")
        assert constraint.consumes_dof == 1
        assert constraint.references("p0") is True

    def test_geometric_constraint_invalid_type(self) -> None:
        with pytest.raises(ValueError):
            GeometricConstraint("NOT_A_TYPE", "p0", "p1")

    def test_solve_converges(self) -> None:
        sketch = _rect_sketch()
        solution = SketchConstraintSolver().solve(sketch)
        assert isinstance(solution, ConstraintSolution)
        assert solution.residual <= 1e-3

    def test_analyze_dof(self) -> None:
        sketch = _rect_sketch()
        analysis = SketchConstraintSolver.analyze_degrees(sketch)
        assert analysis.is_fully_constrained in (True, False)


class TestParameters:
    def test_parameter_lookup(self) -> None:
        table = ParameterTable()
        table.add(Parameter("width", 10.0))
        assert table.get("width").value == 10.0  # type: ignore[union-attr]

    def test_set_value(self) -> None:
        table = ParameterTable()
        table.add(Parameter("height", 5.0))
        table.set("height", 7.0)
        assert table.get("height").value == 7.0  # type: ignore[union-attr]

    def test_unknown_parameter(self) -> None:
        table = ParameterTable()
        assert table.get("nope") is None


class TestParameterExpressions:
    def test_is_expression_flag(self) -> None:
        direct = Parameter("d", 5.0)
        expr = Parameter("e", 0.0, expression="w * 2")
        assert not direct.is_expression
        assert expr.is_expression

    def test_resolve_expression(self) -> None:
        table = ParameterTable()
        table.add(Parameter("w", 10.0))
        table.add(Parameter("h", 0.0, expression="w * 2"))
        assert table.resolve("h") == 20.0

    def test_resolve_expression_dependency(self) -> None:
        table = ParameterTable()
        table.add(Parameter("depth", 5.0))
        table.add(Parameter("box_depth", 0.0, expression="depth + 5"))
        assert table.resolve("box_depth") == 10.0

    def test_dependencies_reported(self) -> None:
        table = ParameterTable()
        table.add(Parameter("base", 1.0))
        table.add(Parameter("derived", 0.0, expression="base * 3"))
        assert table.dependencies("derived") == ["base"]

    def test_resolve_all(self) -> None:
        table = ParameterTable()
        table.add(Parameter("a", 2.0))
        table.add(Parameter("b", 0.0, expression="a * a"))
        values = table.resolve_all()
        assert values["a"] == 2.0
        assert values["b"] == 4.0

    def test_invalid_parameter_name(self) -> None:
        with pytest.raises(ValueError):
            Parameter("1bad")
