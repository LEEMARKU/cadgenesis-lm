"""tests/reasoning/test_manufacturing_rules.py
==============================================
Unit tests for cadgenesis.reasoning.manufacturing_rules.
"""

from __future__ import annotations

import pytest

from cadgenesis.reasoning.manufacturing_rules import (
    ManufacturingAssessment,
    ManufacturingRules,
    MfgCheck,
)


class TestMachining:
    def test_thin_wall_fails(self):
        rules = ManufacturingRules()
        checks = rules.check_machining({"min_wall_thickness": 0.5})
        assert not checks[0].passed
        assert checks[0].severity == "error"

    def test_ok_wall(self):
        rules = ManufacturingRules()
        checks = rules.check_machining({"min_wall_thickness": 2.0})
        assert checks[0].passed

    def test_deep_hole_warns(self):
        rules = ManufacturingRules()
        checks = rules.check_machining({"hole_depth": 60, "hole_diameter": 6})
        names = {c.check: c for c in checks}
        assert not names["machining_depth_ratio"].passed
        assert names["machining_depth_ratio"].severity == "warning"

    def test_no_params_ok(self):
        rules = ManufacturingRules()
        checks = rules.check_machining({})
        assert checks[0].passed


class TestInjectionMolding:
    def test_wall_out_of_range(self):
        rules = ManufacturingRules()
        checks = rules.check_injection_molding({"wall_thickness": 8.0})
        assert not checks[0].passed

    def test_insufficient_draft(self):
        rules = ManufacturingRules()
        checks = rules.check_injection_molding({"draft_angle": 0.3})
        assert not checks[0].passed

    def test_undercut(self):
        rules = ManufacturingRules()
        checks = rules.check_injection_molding({"has_undercut": True})
        assert any(not c.passed for c in checks)


class Test3dPrinting:
    def test_thin_wall(self):
        rules = ManufacturingRules()
        checks = rules.check_3d_printing({"min_wall_thickness": 0.2})
        assert not checks[0].passed

    def test_steep_overhang(self):
        rules = ManufacturingRules()
        checks = rules.check_3d_printing({"max_overhang_angle": 60})
        assert not checks[0].passed


class TestSheetMetal:
    def test_tight_bend_radius(self):
        rules = ManufacturingRules()
        checks = rules.check_sheet_metal({"bend_radius": 0.5, "material_thickness": 2.0})
        assert not checks[0].passed
        assert checks[0].severity == "warning"

    def test_ok_bend(self):
        rules = ManufacturingRules()
        checks = rules.check_sheet_metal({"bend_radius": 3.0, "material_thickness": 2.0})
        assert checks[0].passed


class TestAssess:
    def test_assessment(self):
        rules = ManufacturingRules()
        assessment = rules.assess({"min_wall_thickness": 0.3, "processes": ["machining"]})
        assert isinstance(assessment, ManufacturingAssessment)
        assert isinstance(assessment.checks[0], MfgCheck)
        assert not assessment.passed
        assert len(assessment.errors) == 1

    def test_unknown_process(self):
        rules = ManufacturingRules()
        with pytest.raises(ValueError):
            rules.assess({"processes": ["quantum_machining"]})

    def test_summary(self):
        rules = ManufacturingRules()
        assessment = rules.assess({"min_wall_thickness": 2.0, "processes": ["machining"]})
        summary = assessment.summary()
        assert summary["passed"] is True
        assert summary["errors"] == 0
