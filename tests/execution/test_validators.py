"""Tests for Pillar 8 validators: geometry, topology, manufacturing."""

from __future__ import annotations

from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.execution import (
    GeometryValidator,
    ManufacturabilityAnalyzer,
    ManufacturingCheck,
    ManufacturingReport,
    TopologyAnalyzer,
)

STEEL = {
    "name": "steel",
    "yield_strength_pa": 250e6,
    "density_kg_m3": 7800.0,
    "machinability": 0.8,
}


class TestGeometryValidator:
    def test_box_valid(self) -> None:
        report = GeometryValidator().validate_mesh(Mesh.box())
        assert report.valid
        assert report.summary()["errors"] == 0

    def test_open_mesh_invalid(self) -> None:
        mesh = Mesh.box()
        mesh.faces.pop()
        report = GeometryValidator().validate_mesh(mesh)
        assert not report.valid
        assert "mesh:watertight" in report.summary()["failed"]

    def test_degenerate_face_detected(self) -> None:
        mesh = Mesh.from_vertices_faces(
            [(0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 1, 0)],
            [(0, 1, 2), (0, 2, 3)],
        )
        report = GeometryValidator().validate_mesh(mesh)
        assert not report.valid

    def test_self_intersection_detected(self) -> None:
        mesh = Mesh.from_vertices_faces(
            [
                (0, 0, 0),
                (10, 0, 0),
                (0, 10, 0),  # plane z=0
                (2, 2, -5),
                (2, 2, 5),
                (2, 8, 0),  # vertical blade
            ],
            [(0, 1, 2), (3, 4, 5)],
        )
        report = GeometryValidator().validate_mesh(mesh)
        assert not report.valid
        assert "mesh:self_intersections" in report.summary()["failed"]

    def test_shared_vertex_touch_not_intersection(self) -> None:
        report = GeometryValidator().validate_mesh(Mesh.box())
        assert report.checks[-1].passed

    def test_report_to_dict(self) -> None:
        report = GeometryValidator().validate_mesh(Mesh.box())
        data = report.to_dict()
        assert data["valid"] is True
        assert len(data["checks"]) == 4

    def test_min_face_area_filter(self) -> None:
        validator = GeometryValidator(min_face_area=1.0)
        tiny = Mesh.from_vertices_faces(
            [(0, 0, 0), (0.001, 0, 0), (0, 0.001, 0)],
            [(0, 1, 2)],
        )
        report = validator.validate_mesh(tiny)
        assert "mesh:degenerate_faces" in report.summary()["failed"]


class TestTopologyAnalyzer:
    def test_box_topology(self) -> None:
        report = TopologyAnalyzer().analyze_mesh(Mesh.box())
        assert report.valid
        assert report.summary()["failed"] == []

    def test_holes_detected(self) -> None:
        mesh = Mesh.box()
        mesh.faces.pop()
        report = TopologyAnalyzer().analyze_mesh(mesh)
        assert not report.valid
        assert "mesh:closed" in report.summary()["failed"]
        assert "mesh:edge_usage" in report.summary()["failed"]

    def test_face_reuse(self) -> None:
        mesh = Mesh.box()
        mesh.faces.append(mesh.faces[0])
        report = TopologyAnalyzer().analyze_mesh(mesh)
        assert not report.valid


class TestManufacturabilityAnalyzer:
    def test_machining_ok(self) -> None:
        analyzer = ManufacturabilityAnalyzer()
        report = analyzer.assess(
            {"processes": ["machining"], "material": STEEL, "feature_count": 2}
        )
        assert report.passed
        assert isinstance(report, ManufacturingReport)

    def test_thin_wall_warning(self) -> None:
        analyzer = ManufacturabilityAnalyzer()
        report = analyzer.assess(
            {
                "processes": ["machining"],
                "material": STEEL,
                "min_wall_thickness": 0.2,
            }
        )
        assert not report.passed
        assert any(isinstance(c, ManufacturingCheck) and not c.passed for c in report.checks)

    def test_overhang_failure(self) -> None:
        analyzer = ManufacturabilityAnalyzer()
        report = analyzer.assess(
            {
                "processes": ["3d_printing"],
                "material": STEEL,
                "max_overhang_angle": 80,
            }
        )
        assert not report.passed
        assert "print_overhang" in report.summary()["failed"]

    def test_unknown_process(self) -> None:
        report = ManufacturabilityAnalyzer().assess({"processes": ["quantum_engraving"]})
        assert not report.passed

    def test_summary_roundtrip(self) -> None:
        report = ManufacturabilityAnalyzer().assess({"processes": ["machining"], "material": STEEL})
        summary = report.summary()
        assert "passed" in summary
        assert "failed" in summary
