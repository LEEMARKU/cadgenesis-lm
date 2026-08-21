"""cadgenesis.cad.benchmarks.cad_benchmarks
=========================================
Benchmark suite for the Pillar 2 "CAD Intelligence" subsystem.

Measures wall-clock performance of the pure-Python geometry / mesh /
topology / parametric / mechanism kernels:

* primitive volume / area / aabb computation
    * quadric mesh simplification vs face count
    * B-Rep topology validation (Euler / manifold / genus)
    * parametric constraint solving
    * four-bar linkage kinematic sweeps
    * GD&T specification validation
    * CAD intelligence pipeline (generation, tokenize, validate, memorise)
    * assembly tree construction / traversal
    * feature prediction (parametric reconstruction)
    * constraint prediction (geometric solver preconditioning)
    * manufacturability / DFM analysis

Run with::

    python -m cadgenesis.cad.benchmarks.cad_benchmarks              # all
    python -m cadgenesis.cad.benchmarks.cad_benchmarks --section mesh
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from cadgenesis.cad.assembly import Assembly
from cadgenesis.cad.gdt import Datum, DatumReference, FeatureControlFrame, GDTSpecification
from cadgenesis.cad.mechanisms.linkages import FourBarLinkage
from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.cad.mesh.simplify import quadric_simplify
from cadgenesis.cad.modeling.brep import BRepSolid
from cadgenesis.cad.modeling.primitives import (
    make_box,
    make_cylinder,
    make_sphere,
)
from cadgenesis.cad.parametric.constraints import GeometricConstraint, SketchConstraintSolver
from cadgenesis.cad.parametric.sketch import Sketch
from cadgenesis.cad.validation.checks import check_manufacturability

SECTIONS = (
    "geometry",
    "mesh",
    "brep",
    "parametric",
    "mechanisms",
    "gdt",
    "generation",
    "assembly",
    "feature_prediction",
    "constraint_prediction",
    "manufacturability",
)


def time_fn(fn, reps: int) -> float:
    """Mean wall-clock seconds over ``reps`` runs (after one warm-up)."""
    fn()
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def bench_geometry(reps: int) -> None:
    print("\n== geometry primitives (volume + surface + aabb) ==")
    primitives = [
        make_box(10.0, 20.0, 30.0),
        make_cylinder(5.0, 40.0),
        make_sphere(7.5),
    ]

    def work() -> float:
        total = 0.0
        for prim in primitives:
            total += prim.volume() + prim.surface_area()
            _min_v, max_v = prim.aabb()
            total += max_v.x + max_v.y + max_v.z
        return total

    elapsed = time_fn(work, reps)
    print(f"  3 primitives x {reps} reps: {elapsed * 1e6:.1f} us/rep  (result={work():.2f})")


def bench_mesh(reps: int) -> None:
    print("\n== mesh quadric simplification ==")
    mesh = Mesh.uv_sphere(radius=1.0, segments=16, rings=8)
    base_faces = mesh.face_count

    def work() -> int:
        simplified = quadric_simplify(mesh, target_faces=max(24, base_faces // 4))
        return simplified.face_count

    result = work()
    elapsed = time_fn(work, reps)
    print(f"  sphere {base_faces} faces -> {result} faces ({elapsed * 1e6:.1f} us/rep)")


def bench_brep(reps: int) -> None:
    print("\n== B-Rep topology validation ==")
    solid = BRepSolid.from_prism(10.0, 5.0, 3.0)

    def work() -> int:
        solid.validate()
        analysis = solid.analyze()
        return int(analysis["genus"])

    result = work()
    elapsed = time_fn(work, reps)
    print(f"  prism genus={result} (valid) x {reps} reps: {elapsed * 1e6:.1f} us/rep")


def bench_parametric(reps: int) -> None:
    print("\n== parametric constraint solving ==")
    solver = SketchConstraintSolver()

    def build_sketch() -> Sketch:
        sketch = Sketch("bench")
        sketch.add_point(0.0, 0.0, name="p0")
        sketch.add_point(10.0, 0.0, name="p1")
        sketch.add_point(10.0, 8.0, name="p2")
        sketch.add_point(0.0, 8.0, name="p3")
        sketch.add_constraint(GeometricConstraint("HORIZONTAL", "p0", "p1"))
        sketch.add_constraint(GeometricConstraint("PERPENDICULAR", "p1", "p2"))
        sketch.add_constraint(GeometricConstraint("FIXED", "p0"))
        return sketch

    def work() -> float:
        solution = solver.solve(build_sketch())
        return float(solution.residual)

    result = work()
    elapsed = time_fn(work, reps)
    print(f"  sketch residual={result:.2e} x {reps} reps: {elapsed * 1e6:.1f} us/rep")


def bench_mechanisms(reps: int) -> None:
    print("\n== four-bar linkage kinematics ==")
    linkage = FourBarLinkage(ground=60.0, crank=20.0, coupler=70.0, rocker=40.0)

    def work() -> float:
        angles = [a for a in range(0, 361, 2) if linkage.rocker_angle(a) is not None]
        return float(len(angles))

    result = work()
    elapsed = time_fn(work, reps)
    print(
        f"  grashof={linkage.is_grashof} type={linkage.mechanism_type} "
        f"valid angles={result} x {reps} reps: {elapsed * 1e6:.1f} us/rep"
    )


def bench_gdt(reps: int) -> None:
    print("\n== GD&T specification validation ==")
    spec = GDTSpecification(
        datums=[Datum(identifier="A"), Datum(identifier="B")],
        control_frames=[
            FeatureControlFrame(
                characteristic="POSITION",
                tolerance=0.1,
                datums=[DatumReference("A")],
            )
        ],
    )

    def work() -> int:
        problems = spec.validate()
        return len(problems)

    result = work()
    elapsed = time_fn(work, reps)
    print(f"  GD&T problems={result} x {reps} reps: {elapsed * 1e6:.1f} us/rep")


def bench_feature_prediction(reps: int) -> None:
    print("\n== feature prediction (parametric reconstruction) ==")
    solver = SketchConstraintSolver()

    def predict() -> int:
        sketch = Sketch("pred")
        sketch.add_point(0.0, 0.0, name="p0")
        sketch.add_point(100.0, 0.0, name="p1")
        sketch.add_point(100.0, 60.0, name="p2")
        sketch.add_point(0.0, 60.0, name="p3")
        sketch.add_constraint(GeometricConstraint("HORIZONTAL", "p0", "p1"))
        sketch.add_constraint(GeometricConstraint("PARALLEL", "p1", "p2", name="vert"))
        sketch.add_constraint(GeometricConstraint("FIXED", "p0"))
        sketch.add_constraint(GeometricConstraint("COINCIDENT", "p1", "p2"))
        solution = solver.solve(sketch)
        return int(solution.is_fully_constrained or solution.is_over_constrained)

    result = predict()
    elapsed = time_fn(predict, reps)
    print(f"  reconstruction constrained={result} x {reps} reps: {elapsed * 1e6:.1f} us/rep")


def bench_constraint_prediction(reps: int) -> None:
    print("\n== constraint prediction (solver preconditioning) ==")
    solver = SketchConstraintSolver()

    def predict_constraints() -> float:
        sketch = Sketch("constraints")
        sketch.add_point(0.0, 0.0, name="p0")
        sketch.add_point(10.0, 0.0, name="p1")
        sketch.add_point(10.0, 10.0, name="p2")
        sketch.add_constraint(GeometricConstraint("HORIZONTAL", "p0", "p1"))
        sketch.add_constraint(GeometricConstraint("PERPENDICULAR", "p0", "p2"))
        sketch.add_constraint(GeometricConstraint("FIXED", "p0"))
        solution = solver.solve(sketch)
        return float(solution.residual)

    result = predict_constraints()
    elapsed = time_fn(predict_constraints, reps)
    print(f"  constraint residual={result:.2e} x {reps} reps: {elapsed * 1e6:.1f} us/rep")


def build_assembly(n: int) -> Assembly:
    """Build a small nested assembly with ``n`` parts."""
    root = Assembly("assy")
    sub = root.add_subassembly("sub")
    for i in range(n):
        root.add_part(f"part_{i}", part_id=f"P{i}")
        root.add_part(f"subpart_{i}", part_id=f"S{i}", parent=sub)
    return root


def bench_assembly(reps: int) -> None:
    print("\n== assembly tree construction / traversal ==")
    assembly = build_assembly(50)

    def work() -> int:
        return assembly.part_count() + assembly.max_depth()

    result = work()
    elapsed = time_fn(work, reps)
    print(f"  parts={result} x {reps} reps: {elapsed * 1e6:.1f} us/rep")


def bench_generation(reps: int) -> None:
    print("\n== CAD intelligence pipeline (generation/validate/memorise) ==")
    from cadgenesis.cad.integration.pipeline import CADIntelligencePipeline

    pipeline = CADIntelligencePipeline()
    design: dict[str, Any] = {
        "primitives": [
            {"kind": "box", "dims": {"length": 10.0, "width": 5.0, "height": 2.0}},
            {"kind": "cylinder", "dims": {"radius": 1.0, "height": 2.0}},
        ],
        "part": {"material": "Aluminum", "process": "CNC_MILLING"},
    }

    def generate() -> int:
        report = pipeline.run(design, name="bench_design")
        return int(report.validation.passed) if report.validation is not None else 0

    result = generate()
    elapsed = time_fn(generate, reps)
    print(f"  pipeline passed={result} x {reps} reps: {elapsed * 1e6:.1f} us/rep")


def bench_manufacturability(reps: int) -> None:
    print("\n== manufacturability / DFM analysis ==")
    features = [
        {"name": "base", "kind": "BOX", "parameters": {"length": 100.0, "height": 10.0}},
        {"name": "hole", "kind": "HOLE", "parameters": {"radius": 2.0}},
        {"name": "slot", "kind": "SLOT", "parameters": {"width": 5.0}},
    ]

    def work() -> int:
        problems = check_manufacturability({"features": features})
        return len(problems)

    result = work()
    elapsed = time_fn(work, reps)
    print(f"  DFM problems={result} x {reps} reps: {elapsed * 1e6:.1f} us/rep")


SECTIONS_FN = {
    "geometry": bench_geometry,
    "mesh": bench_mesh,
    "brep": bench_brep,
    "parametric": bench_parametric,
    "mechanisms": bench_mechanisms,
    "gdt": bench_gdt,
    "generation": bench_generation,
    "assembly": bench_assembly,
    "feature_prediction": bench_feature_prediction,
    "constraint_prediction": bench_constraint_prediction,
    "manufacturability": bench_manufacturability,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="CAD Intelligence micro-benchmarks")
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--section", choices=SECTIONS, default=None)
    args = parser.parse_args()

    print(f"CAD Intelligence benchmarks (reps={args.reps})")
    sections = [args.section] if args.section else SECTIONS
    for section in sections:
        SECTIONS_FN[section](args.reps)


if __name__ == "__main__":
    main()
