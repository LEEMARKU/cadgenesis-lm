"""Tests for cadgenesis.cad.assembly (hierarchy, mates, DOF)."""

from __future__ import annotations

import pytest

from cadgenesis.cad.assembly.assembly import Assembly, Component
from cadgenesis.cad.assembly.mates import (
    MATE_TYPES,
    AssemblyConstraint,
    MateSolver,
    Reference,
)
from cadgenesis.cad.geometry.core import Transform


class TestComponent:
    def test_part_vs_assembly(self) -> None:
        part = Component("body", is_assembly=False)
        sub = Component("sub", is_assembly=True)
        assert part.is_part is True
        assert sub.is_part is False

    def test_add_child_requires_assembly(self) -> None:
        part = Component("body", is_assembly=False)
        with pytest.raises(ValueError):
            part.add_child(Component("child"))

    def test_depth(self) -> None:
        root = Component("root", is_assembly=True)
        child = Component("child", is_assembly=True)
        root.add_child(child)
        child.add_child(Component("leaf"))
        assert root.depth() == 3


class TestAssembly:
    def test_add_part(self) -> None:
        assembly = Assembly("asm")
        assembly.add_part("screw", part_id="screw_m4")
        assert assembly.part_count() == 1

    def test_add_subassembly(self) -> None:
        assembly = Assembly("asm")
        sub = assembly.add_subassembly("sub")
        assembly.add_part("gear", part_id="gear_1", parent=sub)
        assert assembly.part_count() == 1

    def test_unique_part_ids(self) -> None:
        assembly = Assembly("asm")
        assembly.add_part("a", part_id="p1")
        assembly.add_part("b", part_id="p1")
        assert assembly.unique_part_ids() == {"p1"}

    def test_find(self) -> None:
        assembly = Assembly("asm")
        assembly.add_part("a", part_id="p1")
        assert assembly.find("a") is not None
        assert assembly.find("missing") is None

    def test_world_transform(self) -> None:
        from cadgenesis.cad.geometry.core import Vec

        assembly = Assembly("asm")
        sub = assembly.add_subassembly("sub", transform=Transform.translation(10, 0, 0))
        assembly.add_part("part", part_id="p", parent=sub, transform=Transform.translation(0, 5, 0))
        world = assembly.world_transform("part")
        assert world.apply(Vec(0, 0, 0)) == Vec(10, 5, 0)

    def test_round_trip(self) -> None:
        assembly = Assembly("asm")
        assembly.add_part("screw", part_id="screw_m4")
        restored = Assembly.from_dict(assembly.to_dict())
        assert restored.part_count() == 1


class TestMates:
    def test_mate_types(self) -> None:
        assert "COINCIDENT" in MATE_TYPES
        assert "CONCENTRIC" in MATE_TYPES

    def test_constraint_dof(self) -> None:
        constraint = AssemblyConstraint(
            name="c1",
            mate_type="CONCENTRIC",
            reference_a=Reference("a", "axis1", "axis"),
            reference_b=Reference("b", "axis1", "axis"),
        )
        assert constraint.removes_dof == 4

    def test_invalid_mate_type(self) -> None:
        with pytest.raises(ValueError):
            AssemblyConstraint(
                name="c1",
                mate_type="NOT_A_MATE",
                reference_a=Reference("a", "f1"),
                reference_b=Reference("b", "f1"),
            )

    def test_empty_name(self) -> None:
        with pytest.raises(ValueError):
            AssemblyConstraint(
                name="",
                mate_type="COINCIDENT",
                reference_a=Reference("a", "f1"),
                reference_b=Reference("b", "f1"),
            )

    def test_mate_solver(self) -> None:
        solver = MateSolver()
        constraints = [
            AssemblyConstraint(
                name="m1",
                mate_type="CONCENTRIC",
                reference_a=Reference("a", "axis", "axis"),
                reference_b=Reference("b", "axis", "axis"),
            )
        ]
        analysis = solver.analyze_component("a", constraints)
        assert analysis.dof_removed == 4
        assert analysis.dof == 2

    def test_total_dof(self) -> None:
        solver = MateSolver()
        constraints = [
            AssemblyConstraint(
                name="m1",
                mate_type="RIGID",
                reference_a=Reference("a", "f1"),
                reference_b=Reference("b", "f1"),
            )
        ]
        assert solver.total_dof(["a", "b"], constraints) == 0
        assert solver.is_rigid(["a", "b"], constraints) is True

    def test_constraint_roundtrip(self) -> None:
        constraint = AssemblyConstraint(
            name="m1",
            mate_type="PARALLEL",
            reference_a=Reference("a", "f1", "face"),
            reference_b=Reference("b", "f1", "face"),
            offset=1.5,
        )
        restored = AssemblyConstraint.from_dict(constraint.to_dict())
        assert restored.offset == 1.5
