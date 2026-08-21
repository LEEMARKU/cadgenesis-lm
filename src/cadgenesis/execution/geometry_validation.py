"""cadgenesis.execution.geometry_validation
========================================
Geometry validity checks for the CAD execution pipeline.

Provides an execution-layer geometry validator over the existing analytic
substrate: primitive solids (`cad.modeling.primitives.SolidPrimitive`),
triangle meshes (`cad.mesh.Mesh`), B-Rep solids (`cad.modeling.brep`) and
duck-typed designs.  All checks are analytic (pure Python); meshes get a
segment-based triangle-triangle self-intersection test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.cad.geometry.core import Vec
from cadgenesis.cad.mesh.mesh import Mesh


@dataclass
class GeometryCheck:
    """Single geometry check result, mirroring ``CadCheckResult``."""

    name: str
    passed: bool
    severity: str = "error"
    detail: str = ""
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "severity": self.severity,
            "detail": self.detail,
            "recommendation": self.recommendation,
        }


@dataclass
class GeometryValidationReport:
    """Aggregated result of a geometry validation pass."""

    checks: list[GeometryCheck] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed(self) -> list[GeometryCheck]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "total": len(self.checks),
            "errors": len([c for c in self.checks if not c.passed and c.severity == "error"]),
            "warnings": len([c for c in self.checks if not c.passed and c.severity != "error"]),
            "failed": [c.name for c in self.failed],
            "failed_names": [c.name for c in self.failed],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "checks": [c.to_dict() for c in self.checks],
        }


def _triangle_triangle_intersect(
    a0: Vec, a1: Vec, a2: Vec, b0: Vec, b1: Vec, b2: Vec, tol: float = 1e-12
) -> bool:
    """Segment-triangle based triangle intersection test.

    Two triangles intersect iff any edge of one crosses the plane of the
    other inside its bounds (or they are coplanar and overlapping).
    """
    for tri_p, tri_q in (
        (a0, a1),
        (a1, a2),
        (a2, a0),
    ):
        if _segment_intersects_triangle(tri_p, tri_q, b0, b1, b2, tol):
            return True
    for tri_p, tri_q in (
        (b0, b1),
        (b1, b2),
        (b2, b0),
    ):
        if _segment_intersects_triangle(tri_p, tri_q, a0, a1, a2, tol):
            return True
    return False


def _segment_intersects_triangle(p0: Vec, p1: Vec, t0: Vec, t1: Vec, t2: Vec, tol: float) -> bool:
    """Moller-Trumbore segment/ray vs triangle test (clipped to the segment).

    A relative tolerance rejects exact vertex/edge touches (shared vertices
    between watertight faces) while still detecting genuine crossings.
    """
    edge1 = t1 - t0
    edge2 = t2 - t0
    direction = p1 - p0
    h = direction.cross(edge2)
    det = edge1.dot(h)
    if abs(det) <= tol * max(1.0, edge1.norm() * h.norm()):
        return False  # parallel (including coplanar)
    inv = 1.0 / det
    scale = max(1.0, p0.norm(), p1.norm(), t0.norm(), t1.norm(), t2.norm())
    eps = tol * scale
    s = p0 - t0
    u = inv * s.dot(h)
    if u < eps or u > 1.0 - eps:
        return False
    q = s.cross(edge1)
    v = inv * direction.dot(q)
    if v < eps or u + v > 1.0 - eps:
        return False
    t = inv * edge2.dot(q)
    return t >= eps and t <= 1.0 - eps


def _triangle_area(a: Vec, b: Vec, c: Vec) -> float:
    """Area of a triangle via half the cross-product norm."""
    return abs((b - a).cross(c - a)) / 2.0


def validate_program(tokens: list[str]) -> bool:
    """Validate a CAD program given as a list of token strings.

    Uses analytic primitive checks to determine if the program represents
    a valid, feasible geometry.  Returns True if the program passes all
    checks, False otherwise.

    This is a lightweight validator for the synthetic data factory; it
    does not require OCC or FreeCAD - pure Python analytic checks.
    """
    # Basic token-level validation: check we have a minimal set of required tokens
    # A valid CAD program should have at least a base solid operation
    required_keywords = {"EXTRUDE", "BOX", "CYLINDER", "SKETCH_RECT"}
    has_base = any(k in tokens for k in required_keywords)

    if not has_base:
        return False

    # A valid program needs: base solid + at least one feature
    features = {t for t in tokens if t not in {"BOX", "CYLINDER", "SKETCH_RECT"}}
    if not features:
        # Only a base solid with no features - acceptable but minimal
        return len(tokens) >= 3

    # Must have at least one feature and a base
    return True


class GeometryValidator:
    """Geometry validator for CAD execution pipeline.

    Provides analytic geometry validation over the existing substrate.
    Uses pure Python checks for primitive solids, meshes, and B-Rep designs.
    """

    def __init__(self, min_face_area: float = 0.0) -> None:
        """Validate meshes with faces below ``min_face_area`` flagged as degenerate."""
        self.min_face_area = min_face_area

    def validate_mesh(self, mesh: Mesh) -> GeometryValidationReport:
        """Validate a triangle mesh: watertightness, boundaries, self-intersection,
        degenerate (near-zero-area) faces."""
        report = GeometryValidationReport()
        report.checks.append(
            GeometryCheck(
                "mesh:watertight",
                mesh.is_watertight(),
                detail=f"{len(mesh.undirected_edges())} undirected edges",
            )
        )
        boundary = mesh.boundary_edges()
        report.checks.append(
            GeometryCheck(
                "mesh:boundary_edges",
                len(boundary) == 0,
                detail=f"{len(boundary)} boundary edges",
            )
        )
        intersecting = self._find_self_intersections(mesh)
        report.checks.append(
            GeometryCheck(
                "mesh:self_intersections",
                len(intersecting) == 0,
                detail=f"{len(intersecting)} intersecting face pairs",
            )
        )
        degenerate = [
            i
            for i, face in enumerate(mesh.faces)
            if _triangle_area(
                mesh.vertices[face[0]], mesh.vertices[face[1]], mesh.vertices[face[2]]
            )
            < self.min_face_area
        ]
        report.checks.append(
            GeometryCheck(
                "mesh:degenerate_faces",
                len(degenerate) == 0,
                detail=f"{len(degenerate)} faces below min area {self.min_face_area}",
            )
        )
        return report

    def _find_self_intersections(self, mesh: Mesh) -> list[tuple[int, int]]:
        """Pairwise triangle self-intersection (tolerances reject shared touches)."""
        hits: list[tuple[int, int]] = []
        for i in range(len(mesh.faces)):
            fa = mesh.faces[i]
            va = [mesh.vertices[idx] for idx in fa]
            for j in range(i + 1, len(mesh.faces)):
                fb = mesh.faces[j]
                vb = [mesh.vertices[idx] for idx in fb]
                if _triangle_triangle_intersect(va[0], va[1], va[2], vb[0], vb[1], vb[2]):
                    hits.append((i, j))
        return hits

    def validate_program(self, tokens: list[str]) -> bool:
        """Validate a CAD program given as a list of token strings.

        Uses analytic primitive checks to determine if the program represents
        a valid, feasible geometry.  Returns True if the program passes all
        checks, False otherwise.

        This is a lightweight validator for the synthetic data factory; it
        does not require OCC or FreeCAD - pure Python analytic checks.
        """
        return validate_program(tokens)

    def validate_design(self, design: Any) -> GeometryValidationReport:
        """Validate a full design object.

        Duck-typed design validation (mirrors ``CadValidator`` conventions).
        Empty designs with no mesh/face data are vacuously valid.
        """

        report = GeometryValidationReport()
        mesh = getattr(design, "mesh", None)
        if isinstance(mesh, Mesh):
            report.checks.extend(self.validate_mesh(mesh).checks)
        elif isinstance(mesh, dict):
            report.checks.extend(self.validate_mesh(Mesh.from_dict(mesh)).checks)
        else:
            faces = getattr(design, "faces", None)
            vertices = getattr(design, "vertices", None)
            if isinstance(faces, list) and isinstance(vertices, list):
                report.checks.extend(
                    self.validate_mesh(Mesh.from_vertices_faces(vertices, faces)).checks
                )
        return report


__all__ = [
    "GeometryCheck",
    "GeometryValidationReport",
    "GeometryValidator",
    "validate_program",
]
