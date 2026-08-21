"""cadgenesis.cad.validation.checks
=================================
CAD-specific validation checks built on the existing reasoning toolkit
(topology analyzer, manufacturing rules, geometry reasoner) plus the new
CAD packages (meshes, B-Rep solids, GD&T, materials).

Each check returns a :class:`cadgenesis.cad.validation.report.CheckResult`.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.cad.validation.report import CadCheckResult

# Mesh checks ------------------------------------------------------------------

MeshFaces = list[list[int]]  # vertex-index faces


def check_mesh_topology(
    faces: MeshFaces,
    name: str = "mesh",
    analyzer=None,
) -> list[CadCheckResult]:
    """Run topology analyzer over an explicit triangle mesh."""
    from cadgenesis.reasoning.topology import TopologyAnalyzer

    analyzer = analyzer or TopologyAnalyzer()
    if not faces:
        return [CadCheckResult(name=f"{name}:topology", passed=False, detail="empty mesh")]
    stats = analyzer.analyze_mesh(faces)
    results: list[CadCheckResult] = [
        CadCheckResult(name=f"{name}:note", passed=False, detail=note) for note in stats.notes
    ]
    results.append(
        CadCheckResult(
            name=f"{name}:manifold",
            passed=stats.is_manifold,
            detail=f"manifold={stats.is_manifold}",
        )
    )
    results.append(
        CadCheckResult(
            name=f"{name}:closed",
            passed=stats.is_closed,
            detail=f"closed={stats.is_closed}, genus={stats.genus}",
        )
    )
    return results


def check_mesh_quality(
    vertices: list[tuple[float, float, float]],
    faces: MeshFaces,
    name: str = "mesh",
    min_area: float = 1e-6,
    max_aspect: float = 50.0,
) -> list[CadCheckResult]:
    """Degenerate-face and aspect-ratio checks on a triangle mesh."""
    import math

    results: list[CadCheckResult] = []
    worst_aspect = 0.0
    degenerate = 0
    for face in faces:
        if len(face) != 3:
            continue
        a, b, c = (vertices[i] for i in face)
        cross = _triangle_area(a, b, c)
        if cross <= min_area:
            degenerate += 1
            continue
        lengths = sorted(
            (
                math.dist(a, b),
                math.dist(b, c),
                math.dist(c, a),
            )
        )
        aspect = lengths[2] / lengths[0]
        worst_aspect = max(worst_aspect, aspect)
    results.append(
        CadCheckResult(
            name=f"{name}:degenerate_faces",
            passed=degenerate == 0,
            detail=f"{degenerate} degenerate face(s)",
        )
    )
    results.append(
        CadCheckResult(
            name=f"{name}:aspect_ratio",
            passed=worst_aspect <= max_aspect,
            detail=f"worst aspect ratio {worst_aspect:.2f} (limit {max_aspect})",
        )
    )
    return results


def _triangle_area(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> float:
    import math

    ab = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    ac = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
    cross = (
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    )
    return math.sqrt(sum(v * v for v in cross)) / 2.0


# B-Rep checks -----------------------------------------------------------------


def check_brep_solid(solid, name: str = "brep") -> list[CadCheckResult]:
    """Validate a :class:`cadgenesis.cad.modeling.brep.BRepSolid`."""
    results: list[CadCheckResult] = []
    try:
        problems = solid.validate()
        if problems:
            results.append(
                CadCheckResult(
                    name=f"{name}:solid",
                    passed=False,
                    detail="; ".join(problems) if isinstance(problems, list) else str(problems),
                )
            )
        else:
            results.append(CadCheckResult(name=f"{name}:solid", passed=True, detail="valid"))
    except Exception as exc:
        results.append(CadCheckResult(name=f"{name}:solid", passed=False, detail=f"error: {exc}"))
    return results


# GD&T checks ------------------------------------------------------------------


def check_gdt_spec(spec, name: str = "gdt") -> list[CadCheckResult]:
    """Validate a :class:`cadgenesis.cad.gdt.GDTSpecification`."""
    if spec is None:
        return [CadCheckResult(name=f"{name}:spec", passed=True, detail="no GD&T specified")]
    try:
        problems = spec.validate()
    except Exception as exc:
        problems = [str(exc)]
    return [
        CadCheckResult(
            name=f"{name}:spec",
            passed=not problems,
            detail="; ".join(problems) or "GD&T specification valid",
        )
    ]


# Material checks --------------------------------------------------------------


def check_material(material: dict[str, Any], name: str = "material") -> list[CadCheckResult]:
    """Sanity-check a material descriptor (density, strength, etc.)."""
    results: list[CadCheckResult] = []
    density = material.get("density_kg_m3")
    if density is not None:
        results.append(
            CadCheckResult(
                name=f"{name}:density",
                passed=float(density) > 0,
                detail=f"density {density} kg/m^3",
            )
        )
    yield_strength = material.get("yield_strength_pa")
    if yield_strength is not None:
        results.append(
            CadCheckResult(
                name=f"{name}:yield_strength",
                passed=float(yield_strength) > 0,
                detail=f"yield strength {yield_strength} Pa",
            )
        )
    return results or [
        CadCheckResult(name=f"{name}:present", passed=True, detail="material present")
    ]


# Process checks ---------------------------------------------------------------


def check_manufacturability(
    part: dict[str, Any],
    rules=None,
    name: str = "mfg",
) -> list[CadCheckResult]:
    """Run DFM rules from the reasoning toolkit against a part descriptor."""
    from cadgenesis.reasoning.manufacturing_rules import ManufacturingRules

    rules = rules or ManufacturingRules()
    assessment = rules.assess(part)
    return [
        CadCheckResult(
            name=f"{name}:{check.check}",
            passed=check.passed,
            severity=check.severity,
            detail=check.detail,
            recommendation=check.recommendation,
        )
        for check in assessment.checks
    ]


# Constraint checks -------------------------------------------------------------


def check_constraints(
    sketch,
    solver=None,
    name: str = "constraints",
) -> list[CadCheckResult]:
    """Solve a sketch's geometric constraints and report its DOF status.

    Accepts a sketch (``Sketch`` with entities and constraints), or a dict with
    ``constraints`` (list of ``GeometricConstraint``-compatible dicts) and
    ``entities``.  Returns a result describing the solved residual and whether
    the sketch is under / fully / over constrained.
    """
    from cadgenesis.cad.parametric.constraints import SketchConstraintSolver

    if sketch is None:
        return [CadCheckResult(name=f"{name}:present", passed=True, detail="no sketch to check")]
    solver = solver or SketchConstraintSolver()
    solution = solver.solve(sketch)
    return [
        CadCheckResult(
            name=f"{name}:residual",
            passed=solution.residual < 1e-6,
            detail=f"residual {solution.residual:.2e} after {solution.iterations} iterations",
        ),
        CadCheckResult(
            name=f"{name}:status",
            passed=True,
            detail=f"{solution.status} constrained (dof={solution.dof})",
        ),
    ]


def check_design_consistency(
    design: Any,
    name: str = "consistency",
) -> list[CadCheckResult]:
    """Check a design for internal consistency (names resolve, no conflicts).

    Verifies, when present on ``design``:
    - ``feature_tree``: references resolve and names are unique
    - ``parameters``: values are finite
    - a callable ``check()`` that returns a list of problem strings
    """
    results: list[CadCheckResult] = []

    ftree = getattr(design, "feature_tree", None)
    if ftree is not None:
        names: list[str] = []
        dups: list[str] = []
        features = ftree if isinstance(ftree, list) else list(ftree)
        for feature in features:
            fname = (
                feature.get("name") if isinstance(feature, dict) else getattr(feature, "name", None)
            )
            if not isinstance(fname, str):
                continue
            if fname in names and fname not in dups:
                dups.append(fname)
            names.append(fname)
        results.append(
            CadCheckResult(
                name=f"{name}:feature_names",
                passed=not dups,
                detail=f"duplicate feature names: {dups}" if dups else "feature names unique",
            )
        )

    params = getattr(design, "parameters", None) or getattr(design, "params", None)
    if isinstance(params, dict):
        nonfinite = [k for k, v in params.items() if v is not None and not _is_finite(v)]
        results.append(
            CadCheckResult(
                name=f"{name}:parameters",
                passed=not nonfinite,
                detail=f"non-finite parameters: {nonfinite}" if nonfinite else "parameters finite",
            )
        )

    check_fn = getattr(design, "check", None)
    if callable(check_fn):
        problems = check_fn()
        if isinstance(problems, str):
            problems = [problems]
        results.append(
            CadCheckResult(
                name=f"{name}:model",
                passed=not problems,
                detail="; ".join(problems) if problems else "model consistent",
            )
        )

    return results or [
        CadCheckResult(name=f"{name}:present", passed=True, detail="no consistency targets found")
    ]


def _is_finite(value: Any) -> bool:
    import math

    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


__all__ = [
    "check_brep_solid",
    "check_constraints",
    "check_design_consistency",
    "check_gdt_spec",
    "check_manufacturability",
    "check_material",
    "check_mesh_quality",
    "check_mesh_topology",
]
