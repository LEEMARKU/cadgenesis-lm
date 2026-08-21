"""Tests for cadgenesis.cad.validation (pipeline, checks, report)."""

from __future__ import annotations

from cadgenesis.cad.gdt import Datum, FeatureControlFrame, GDTSpecification
from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.cad.modeling.brep import BRepSolid
from cadgenesis.cad.parametric.constraints import GeometricConstraint, SketchConstraintSolver
from cadgenesis.cad.parametric.sketch import Sketch
from cadgenesis.cad.validation.checks import (
    check_brep_solid,
    check_constraints,
    check_design_consistency,
    check_gdt_spec,
    check_manufacturability,
    check_material,
    check_mesh_quality,
    check_mesh_topology,
)
from cadgenesis.cad.validation.pipeline import CadValidator
from cadgenesis.cad.validation.report import CadCheckResult, CadValidationReport


class _MeshDesign:
    def __init__(self) -> None:
        self.mesh = Mesh.box(10, 5, 3)
        self.vertices = self.mesh.vertices
        self.faces = self.mesh.faces


class _BreapDesign:
    def __init__(self) -> None:
        self.solid = BRepSolid.from_prism(10, 5, 3)

    def validate(self) -> list[str]:
        return self.solid.validate()


class TestReport:
    def test_empty_report_passes(self) -> None:
        report = CadValidationReport()
        assert report.passed is True

    def test_mixed_results(self) -> None:
        report = CadValidationReport(
            results=[
                CadCheckResult(name="a", passed=True),
                CadCheckResult(name="b", passed=False, severity="warning"),
            ]
        )
        assert report.passed is False
        assert len(report.warnings) == 1
        assert len(report.errors) == 0

    def test_summary(self) -> None:
        report = CadValidationReport(results=[CadCheckResult(name="a", passed=True)])
        summary = report.summary()
        assert summary["passed"] is True
        assert summary["total"] == 1


class TestChecks:
    def test_check_mesh_topology(self) -> None:
        mesh = Mesh.box()
        results = check_mesh_topology(mesh.faces)
        assert results
        assert all(r.passed for r in results)

    def test_check_mesh_quality(self) -> None:
        mesh = Mesh.box()
        results = check_mesh_quality(mesh.vertices, mesh.faces)
        assert all(r.passed for r in results)

    def test_check_brep_solid(self) -> None:
        solid = BRepSolid.from_prism(10, 5, 3)
        results = check_brep_solid(solid)
        assert results and results[0].passed

    def test_check_gdt_spec(self) -> None:
        spec = GDTSpecification(
            datums=[Datum(identifier="A")],
            control_frames=[FeatureControlFrame(characteristic="FLATNESS", tolerance=0.02)],
        )
        results = check_gdt_spec(spec)
        assert all(r.passed for r in results)

    def test_check_material(self) -> None:
        results = check_material({"density_kg_m3": 7850.0, "yield_strength_pa": 2e8})
        assert all(r.passed for r in results)

    def test_check_manufacturability(self) -> None:
        results = check_manufacturability({"processes": ["machining"], "min_wall_thickness": 3.0})
        assert results

    def test_check_constraints_fully_constrained(self) -> None:
        sketch = Sketch("c")
        sketch.add_point(0.0, 0.0, name="p0")
        sketch.add_point(10.0, 0.0, name="p1")
        sketch.add_point(10.0, 8.0, name="p2")
        sketch.add_constraint(GeometricConstraint("HORIZONTAL", "p0", "p1"))
        sketch.add_constraint(GeometricConstraint("PERPENDICULAR", "p1", "p2"))
        sketch.add_constraint(GeometricConstraint("FIXED", "p0"))
        results = check_constraints(sketch)
        assert results
        assert all(r.passed for r in results)

    def test_check_constraints_underconstrained_reports(self) -> None:
        sketch = Sketch("u")
        sketch.add_point(0.0, 0.0, name="p0")
        sketch.add_point(5.0, 5.0, name="p1")
        solution = SketchConstraintSolver().solve(sketch)
        assert solution.status == "under"
        results = check_constraints(sketch)
        assert results
        assert any(r.name.endswith("status") for r in results)

    def test_check_design_consistency_duplicate_names(self) -> None:
        class Design:
            feature_tree = [{"name": "a"}, {"name": "a"}]

        results = check_design_consistency(Design())
        assert any(not r.passed for r in results)

    def test_check_design_consistency_unique_names(self) -> None:
        class Design:
            feature_tree = [{"name": "a"}, {"name": "b"}]
            parameters = {"d": 5.0}

        results = check_design_consistency(Design())
        assert all(r.passed for r in results)

    def test_check_design_consistency_nonfinite_parameter(self) -> None:
        class Design:
            parameters = {"bad": float("nan")}

        results = check_design_consistency(Design())
        assert any(not r.passed for r in results)


class TestValidator:
    def test_mesh_design(self) -> None:
        report = CadValidator().validate(_MeshDesign())
        assert report.passed is True
        assert report.summary()["total"] > 0

    def test_brep_design(self) -> None:
        report = CadValidator().validate(_BreapDesign())
        assert report.passed is True

    def test_disabled_checks(self) -> None:
        validator = CadValidator(
            analyze_topology=False,
            check_material=False,
            check_gdt=False,
            check_manufacturing=False,
        )
        report = validator.validate(_MeshDesign())
        assert report.passed is True

    def test_custom_check(self) -> None:
        validator = CadValidator()

        def extra(design) -> list[CadCheckResult]:
            return [CadCheckResult(name="custom", passed=hasattr(design, "mesh"))]

        validator.add_check(extra)
        report = validator.validate(_MeshDesign())
        assert all(r.passed for r in report.results)
