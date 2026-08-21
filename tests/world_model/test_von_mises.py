"""tests/world_model/test_von_mises.py
======================================
Unit tests for the von Mises stress evaluation in the mechanical reasoner
(the physics hook used by the FEAStressAgent reinforcement loop).
"""

from __future__ import annotations

import pytest

from cadgenesis.world_model import MechanicalReasoner, make_object
from cadgenesis.world_model.objects import BoundaryCondition, LoadCase


def test_von_mises_pure_axial_equals_normal_stress():
    reasoner = MechanicalReasoner()
    bracket = make_object(
        "block", "bracket", {"length": 40, "width": 30, "height": 50}, material="steel"
    )
    load = LoadCase("static", [BoundaryCondition(kind="force", magnitude=5000.0)])
    normal = reasoner.working_stress_mpa(bracket, load.conditions[0])
    assert reasoner.von_mises_stress_mpa(bracket, load) == pytest.approx(normal)


def test_von_mises_torque_raises_equivalent_stress():
    reasoner = MechanicalReasoner()
    shaft = make_object("cylinder", "shaft", {"radius": 10, "height": 100}, material="steel")
    axial = LoadCase("axial", [BoundaryCondition(kind="force", magnitude=1000.0)])
    combined = LoadCase(
        "combined",
        [
            BoundaryCondition(kind="force", magnitude=1000.0),
            BoundaryCondition(kind="torque", magnitude=50.0),
        ],
    )
    assert reasoner.von_mises_stress_mpa(shaft, combined) > reasoner.von_mises_stress_mpa(
        shaft, axial
    )
    assert reasoner.von_mises_stress_mpa(shaft, axial) == pytest.approx(2.5)


def test_check_von_mises_passes_under_yield_target():
    reasoner = MechanicalReasoner()
    bracket = make_object(
        "block", "bracket", {"length": 40, "width": 30, "height": 50}, material="steel"
    )
    load = LoadCase("light", [BoundaryCondition(kind="force", magnitude=5000.0)])
    result = reasoner.check_von_mises(bracket, load, target_safety_factor=2.0)
    assert result.passed
    assert result.values["factor_of_safety"] > 2.0
    assert result.model == "von_mises_first_order"


def test_check_von_mises_fails_when_stress_exceeds_yield_target():
    reasoner = MechanicalReasoner()
    bracket = make_object(
        "block", "bracket", {"length": 40, "width": 30, "height": 50}, material="steel"
    )
    heavy = LoadCase("heavy", [BoundaryCondition(kind="force", magnitude=300000.0)])
    result = reasoner.check_von_mises(bracket, heavy, target_safety_factor=2.0)
    assert not result.passed
    # sigma_vm * target > sigma_yield  ->  reinforcement trigger condition
    assert (
        result.values["sigma_vm_mpa"] * result.values["target_safety_factor"]
        > result.values["sigma_yield_mpa"]
    )


def test_worst_von_mises_picks_strictest_load_case():
    reasoner = MechanicalReasoner()
    bracket = make_object(
        "block", "bracket", {"length": 40, "width": 30, "height": 50}, material="steel"
    )
    light = LoadCase("light", [BoundaryCondition(kind="force", magnitude=5000.0)])
    heavy = LoadCase("heavy", [BoundaryCondition(kind="force", magnitude=300000.0)])
    worst = reasoner.worst_von_mises(bracket, [light, heavy], target_safety_factor=2.0)
    assert not worst.passed
    assert worst.name == "von_mises.heavy"


def test_worst_von_mises_requires_loads():
    reasoner = MechanicalReasoner()
    bracket = make_object("block", "bracket", material="steel")
    with pytest.raises(ValueError):
        reasoner.worst_von_mises(bracket, [])
