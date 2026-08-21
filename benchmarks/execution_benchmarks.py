"""
benchmarks/execution_benchmarks.py
==================================
Benchmarks for the Pillar 8 (v6.0) CAD execution pipeline.

Measures wall-clock time for:
* the full engine pipeline (validate -> simulate -> optimize -> repair -> export),
* each validator and solver stage in isolation,
* export formats,
* the feedback loop,
* digital-twin materialization / synchronization.

Run with::

    python benchmarks/execution_benchmarks.py                  # all
    python benchmarks/execution_benchmarks.py --sections pipe
"""

from __future__ import annotations

import argparse
import time

from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.digital_twin import DigitalTwinSystem
from cadgenesis.execution import (
    CADExecutionEngine,
    ExportEngine,
    FeedbackLoop,
    GeometryValidator,
    OptimizationEngine,
    SimulationEngine,
    TopologyAnalyzer,
)

SECTIONS = ("pipe", "stages", "export", "feedback", "twin")

BOX = {
    "name": "bench_box",
    "processes": ["machining"],
    "material": {
        "name": "steel",
        "yield_strength_pa": 250e6,
        "density_kg_m3": 7800.0,
    },
    "volume_m3": 150e-9,
    "analysis": {"type": "structural", "load": {"force_n": 1000.0}},
}


def time_fn(fn, reps: int) -> float:
    fn()
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def bench_pipe(engine: CADExecutionEngine, reps: int) -> None:
    print("\n== full engine pipeline (design dict) ==")
    design = {**BOX, "mesh": Mesh.box().to_dict()}
    ms = time_fn(lambda: engine.execute(design=design), reps) * 1e3
    print(f"{'execute':>14} | {ms:>10.3f} ms/call")
    result = engine.execute(design=design)
    print(
        f"  valid={result.is_valid_geometry} mfg={result.is_manufacturable} "
        f"conf={result.confidence_score}"
    )


def bench_stages(reps: int) -> None:
    print("\n== individual stages ==")
    mesh = Mesh.box()
    stages = [
        ("geometry_validate", lambda: GeometryValidator().validate_mesh(mesh)),
        ("topology_analyze", lambda: TopologyAnalyzer().analyze_mesh(mesh)),
        ("simulate_structural", lambda: SimulationEngine().structural(BOX)),
        ("optimize", lambda: OptimizationEngine().optimize(BOX)),
    ]
    print(f"{'stage':>20} | {'ms/call':>10}")
    for name, fn in stages:
        ms = time_fn(fn, reps) * 1e3
        print(f"{name:>20} | {ms:>10.3f}")


def bench_export(engine: CADExecutionEngine, reps: int) -> None:
    print("\n== export formats (10x10x10 box) ==")
    mesh = Mesh.box()
    exporter = ExportEngine()
    print(f"{'format':>10} | {'ms/call':>10}")
    for fmt in ("stl", "obj", "ply", "gltf", "dxf", "step", "igs"):
        ms = time_fn(lambda fmt=fmt: exporter.export(mesh, f"bench.{fmt}", fmt), reps) * 1e3
        print(f"{fmt:>10} | {ms:>10.3f}")


def bench_feedback(engine: CADExecutionEngine, reps: int) -> None:
    print("\n== feedback loop ==")
    result = engine.execute(design={**BOX, "mesh": Mesh.box().to_dict()})
    loop = FeedbackLoop()
    reports = {
        "geometry": result.geometry_report,
        "topology": result.topology_report,
        "manufacturing": result.manufacturing_report,
        "simulation": result.simulation_report,
        "optimization": result.optimization_report,
    }
    ms = time_fn(lambda: loop.collect(reports), reps) * 1e3
    print(f"{'collect':>14} | {ms:>10.3f} ms/call")
    print(f"  items={len(loop.collect(reports))}")


def bench_twin(reps: int) -> None:
    print("\n== digital twin ==")
    twin = DigitalTwinSystem()
    twin.materialize("bench", {"mesh": Mesh.box(), "name": "bench"})
    ms = time_fn(lambda: twin.synchronize("bench"), reps) * 1e3
    print(f"{'synchronize':>14} | {ms:>10.3f} ms/call")
    ms = time_fn(lambda: twin.snapshot("bench"), reps) * 1e3
    print(f"{'snapshot':>14} | {ms:>10.3f} ms/call")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", nargs="+", default=SECTIONS, choices=SECTIONS)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()
    engine = CADExecutionEngine()
    for section in args.sections:
        if section == "pipe":
            bench_pipe(engine, args.reps)
        elif section == "stages":
            bench_stages(args.reps)
        elif section == "export":
            bench_export(engine, args.reps)
        elif section == "feedback":
            bench_feedback(engine, args.reps)
        elif section == "twin":
            bench_twin(args.reps)


if __name__ == "__main__":
    main()
