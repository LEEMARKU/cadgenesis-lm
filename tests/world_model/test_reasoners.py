"""tests/world_model/test_reasoners.py
======================================
Unit tests for the Pillar-4 reasoners (spatial, mechanical, functional,
assembly, affordances, simulator, planning).
"""

from __future__ import annotations

import pytest

from cadgenesis.cad.geometry.core import Transform
from cadgenesis.cad.mechanisms.joints import Joint, Mechanism
from cadgenesis.world_model import (
    AffordanceMapper,
    AssemblyValidator,
    FunctionalReasoner,
    MechanicalReasoner,
    MotionSimulator,
    SpatialReasoner,
    WorldAssembly,
    WorldModelSystem,
    make_object,
)
from cadgenesis.world_model.objects import BoundaryCondition, LoadCase


class TestSpatialReasoner:
    def test_clearance_separated(self):
        reasoner = SpatialReasoner()
        base = make_object("block", "base", {"length": 100, "width": 60, "height": 8})
        top = make_object(
            "block",
            "top",
            {"length": 40, "width": 30, "height": 50},
            pose=Transform.translation(0, 0, 60),
        )
        report = reasoner.clearance_report(base, top, minimum=2.0, axis="z")
        assert report.passed
        assert report.checks[0]["value"] == pytest.approx(31.0)

    def test_clearance_fails_on_overlap(self):
        reasoner = SpatialReasoner()
        a = make_object("block", "a", {"length": 10, "width": 10, "height": 10})
        b = make_object("block", "b", {"length": 10, "width": 10, "height": 10})
        assert reasoner.overlap(a, b)
        assert reasoner.fits_inside(a, b)

    def test_fits_inside(self):
        reasoner = SpatialReasoner()
        outer = make_object("block", "outer", {"length": 100, "width": 100, "height": 100})
        inner = make_object("block", "inner", {"length": 10, "width": 10, "height": 10})
        assert reasoner.fits_inside(outer, inner)
        assert not reasoner.fits_inside(inner, outer)

    def test_world_bounds_rotated(self):
        reasoner = SpatialReasoner()
        obj = make_object(
            "block",
            "beam",
            {"length": 100, "width": 10, "height": 10},
            pose=Transform.rotation(
                0.5, __import__("cadgenesis.cad.geometry.core", fromlist=["Vec"]).Vec(0, 0, 1)
            ),
        )
        lo, hi = reasoner.world_bounds(obj)
        assert hi.x > 0 and lo.x < 0


class TestMechanicalReasoner:
    def test_factor_of_safety(self):
        reasoner = MechanicalReasoner()
        bracket = make_object(
            "block", "bracket", {"length": 40, "width": 30, "height": 50}, material="steel"
        )
        load = LoadCase("static", [BoundaryCondition(kind="force", magnitude=5000.0)])
        result = reasoner.check_load(bracket, load, target_safety_factor=2.0)
        assert result.passed
        assert result.values["factor_of_safety"] > 2.0

    def test_overload_fails(self):
        reasoner = MechanicalReasoner()
        bracket = make_object(
            "block", "bracket", {"length": 40, "width": 30, "height": 50}, material="plastic"
        )
        load = LoadCase("static", [BoundaryCondition(kind="force", magnitude=100000.0)])
        assert not reasoner.check_load(bracket, load).passed

    def test_stability_slender_fails(self):
        reasoner = MechanicalReasoner()
        pillar = make_object("cylinder", "pillar", {"radius": 2, "height": 500})
        assert not reasoner.stability(pillar).passed

    def test_mass_budget(self):
        reasoner = MechanicalReasoner()
        plate = make_object(
            "block", "plate", {"length": 100, "width": 60, "height": 8}, material="steel"
        )
        assert reasoner.mass_budget([plate], limit_kg=10.0).passed


class TestFunctionalReasoner:
    def test_dof(self):
        reasoner = FunctionalReasoner()
        obj = make_object("block", "free")
        assert reasoner.requires_dof(obj, 6).passed

    def test_envelope(self):
        reasoner = FunctionalReasoner()
        obj = make_object("block", "part", {"length": 50, "width": 20, "height": 10})
        assert reasoner.fits_in_box(obj, 60, 30, 20).passed
        assert not reasoner.fits_in_box(obj, 40, 30, 20).passed


class TestAssemblyValidator:
    def test_valid_assembly(self):
        validator = AssemblyValidator()
        base = make_object("block", "base")
        cover = make_object("block", "cover")
        assembly = WorldAssembly(
            name="box",
            parts=[base, cover],
            mates=[{"type": "COINCIDENT", "part_a": "base", "part_b": "cover"}],
        )
        checks = validator.validate(assembly)
        assert all(c.passed for c in checks)

    def test_bad_mate_type(self):
        validator = AssemblyValidator()
        assembly = WorldAssembly(
            name="bad",
            parts=[make_object("block", "a")],
            mates=[{"type": "GLUED", "part_a": "a", "part_b": "a"}],
        )
        checks = validator.validate(assembly)
        assert not checks[0].passed

    def test_disconnected(self):
        validator = AssemblyValidator()
        assembly = WorldAssembly(
            name="split", parts=[make_object("block", "a"), make_object("block", "b")]
        )
        checks = validator.validate(assembly)
        assert not all(c.passed for c in checks)


class TestAffordanceMapper:
    def test_hole_insert(self):
        mapper = AffordanceMapper()
        hole = make_object("hole", "h", {"radius": 3, "depth": 8})
        assert mapper.supports(hole, "insert") is not None
        assert mapper.supports(hole, "roll") is None

    def test_cylinder_rotate(self):
        mapper = AffordanceMapper()
        cylinder = make_object("cylinder", "c", {"radius": 5, "height": 20})
        assert "rotate" in [a.action for a in mapper.affordances(cylinder)]


class TestSimulator:
    def test_forward_kinematics(self):
        sim = MotionSimulator()
        mech = Mechanism("arm")
        for name in ("l0", "l1", "l2"):
            mech.add_link(name)
        mech.add_joint(Joint("j1", "REVOLUTE", "l0", "l1"))
        mech.add_joint(Joint("j2", "REVOLUTE", "l1", "l2"))
        straight = sim.simulate(mech, {"j1": 0.0, "j2": 0.0}, link_offsets={"l1": 100, "l2": 100})
        bent = sim.simulate(mech, {"j1": 1.5708, "j2": 0.0}, link_offsets={"l1": 100, "l2": 100})
        sx, _sy, sz = straight.position_of("l2")
        _bx, by, bz = bent.position_of("l2")
        assert sx == pytest.approx(200.0)
        assert by == pytest.approx(200.0, abs=1e-3)
        assert abs(sz) < 1e-9 and abs(bz) < 1e-9

    def test_trajectory(self):
        sim = MotionSimulator()
        mech = Mechanism("arm")
        mech.add_link("l0")
        mech.add_link("l1")
        mech.add_joint(Joint("j1", "REVOLUTE", "l0", "l1"))
        traj = sim.simulate_trajectory(
            mech, {"j1": 0.0}, {"j1": 1.5708}, steps=4, link_offsets={"l1": 100}
        )
        assert len(traj) == 5
        assert traj[0].position_of("l1")[0] == pytest.approx(100.0)


class TestPlanner:
    def test_plan_and_execute(self):
        wm = WorldModelSystem()
        plan = wm.reason("plan", goal="assemble a bracket")
        result = wm.reason("execute_plan", plan=plan)
        assert result.all_passed
        assert len(wm.graph) >= 2
