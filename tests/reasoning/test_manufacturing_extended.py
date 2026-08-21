"""Tests for Pillar 7 extended manufacturing rules (casting/welding/tooling/tolerance)."""

from __future__ import annotations

import pytest

from cadgenesis.reasoning import ManufacturingRules


def test_casting_wall_range() -> None:
    rules = ManufacturingRules()
    checks = rules.check_casting({"wall_thickness": 5.0})
    assert any(c.check == "cast_wall_thickness" and c.passed for c in checks)
    checks = rules.check_casting({"wall_thickness": 0.5})
    assert any(c.check == "cast_wall_thickness" and not c.passed for c in checks)


def test_casting_draft_and_defects() -> None:
    rules = ManufacturingRules()
    checks = rules.check_casting({"draft_angle": 2.0})
    assert all(c.passed for c in checks)
    checks = rules.check_casting({"has_isolated_thick_section": True})
    assert any(not c.passed for c in checks)
    checks = rules.check_casting({"sharp_corners": True})
    assert any(c.check == "cast_sharp_corners" for c in checks)


def test_casting_empty_params() -> None:
    rules = ManufacturingRules()
    checks = rules.check_casting({})
    assert checks == [c for c in checks if c.check == "cast_no_params"]


def test_welding_throat_and_gap() -> None:
    rules = ManufacturingRules()
    checks = rules.check_welding({"weld_throat": 2.0, "root_gap": 1.0})
    assert all(c.passed for c in checks)
    checks = rules.check_welding({"weld_throat": 0.5})
    assert any(c.check == "weld_throat_size" and not c.passed for c in checks)
    checks = rules.check_welding({"root_gap": 5.0})
    assert any(c.check == "weld_root_gap" and not c.passed for c in checks)


def test_welding_dissimilar_warning() -> None:
    rules = ManufacturingRules()
    checks = rules.check_welding({"dissimilar_metals": True})
    assert any(c.check == "weld_dissimilar" for c in checks)


def test_tooling_corner_radius_and_aspect() -> None:
    rules = ManufacturingRules()
    checks = rules.check_tooling({"min_corner_radius": 1.0})
    assert all(c.passed for c in checks)
    checks = rules.check_tooling({"min_corner_radius": 0.1})
    assert any(not c.passed for c in checks)
    checks = rules.check_tooling({"pocket_depth": 30.0, "pocket_width": 3.0})
    assert any(c.check == "tool_pocket_aspect" and not c.passed for c in checks)


def test_tooling_undercut() -> None:
    rules = ManufacturingRules()
    checks = rules.check_tooling({"undercut_requires_special_tool": True})
    assert any(c.check == "tool_undercut" for c in checks)


def test_tolerance_feasibility() -> None:
    rules = ManufacturingRules()
    checks = rules.check_tolerance({"nominal_size": 100.0, "tolerance_um": 50.0})
    assert any(c.check == "tolerance_feasibility" and c.passed for c in checks)
    checks = rules.check_tolerance({"nominal_size": 100.0, "tolerance_um": 0.0001})
    assert any(c.check == "tolerance_feasibility" and not c.passed for c in checks)


def test_tolerance_datum_warning() -> None:
    rules = ManufacturingRules()
    checks = rules.check_tolerance({"tolerance_um": 25.0})
    assert any(c.check == "tolerance_datum" for c in checks)
    checks = rules.check_tolerance({"tolerance_um": 25.0, "gd_t_datum": "A"})
    assert all(c.passed for c in checks)


def test_assess_selects_new_processes() -> None:
    rules = ManufacturingRules()
    part = {"processes": ["casting", "welding", "tooling", "tolerance"]}
    assessment = rules.assess(part)
    checks = {c.check for c in assessment.checks}
    assert "cast_no_params" in checks
    assert "weld_no_params" in checks
    assert "tool_no_params" in checks
    assert "tolerance_no_params" in checks


def test_assess_default_processes_unchanged() -> None:
    rules = ManufacturingRules()
    part = {"processes": ["machining", "injection_molding", "3d_printing", "sheet_metal"]}
    assessment = rules.assess(part)
    assert not any(
        c.check.startswith(("cast_", "weld_", "tool_", "tolerance_")) for c in assessment.checks
    )


def test_assess_unknown_process_still_rejected() -> None:
    rules = ManufacturingRules()
    with pytest.raises(ValueError):
        rules.assess({"processes": ["quantum_machining"]})


def test_custom_thresholds() -> None:
    rules = ManufacturingRules(cast_min_wall=10.0)
    checks = rules.check_casting({"wall_thickness": 5.0})
    assert any(not c.passed for c in checks)
