"""cadgenesis.cad.benchmarks
==========================

Benchmark suite for the Pillar 2 "CAD Intelligence" subsystem.

Run with::

    python -m cadgenesis.cad.benchmarks.cad_benchmarks
"""

from cadgenesis.cad.benchmarks.cad_benchmarks import (
    bench_assembly,
    bench_brep,
    bench_constraint_prediction,
    bench_feature_prediction,
    bench_gdt,
    bench_generation,
    bench_geometry,
    bench_manufacturability,
    bench_mechanisms,
    bench_mesh,
    bench_parametric,
    main,
    time_fn,
)

__all__ = [
    "bench_assembly",
    "bench_brep",
    "bench_constraint_prediction",
    "bench_feature_prediction",
    "bench_gdt",
    "bench_generation",
    "bench_geometry",
    "bench_manufacturability",
    "bench_mechanisms",
    "bench_mesh",
    "bench_parametric",
    "main",
    "time_fn",
]
