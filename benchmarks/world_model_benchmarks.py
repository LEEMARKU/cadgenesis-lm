"""
benchmarks/world_model_benchmarks.py
====================================
Benchmarks for the Pillar 4 (v6.0) world-model reasoning stack.

Measures wall-clock time for:
* the reasoner capabilities (clearance, safety, affordances, assembly),
* forward-kinematics simulation,
* plan creation + execution,
* end-to-end world → multimodal embedding.

Run with::

    python benchmarks/world_model_benchmarks.py            # all
    python benchmarks/world_model_benchmarks.py --sections reason
"""

from __future__ import annotations

import argparse
import time

from cadgenesis.cad.geometry.core import Transform
from cadgenesis.cad.mechanisms.joints import Joint, Mechanism
from cadgenesis.world_model import (
    WorldAssembly,
    WorldModelIntegration,
    WorldModelSystem,
)
from cadgenesis.world_model.objects import BoundaryCondition, LoadCase

SECTIONS = ("reason", "simulate", "plan", "embed")


def time_fn(fn, reps: int) -> float:
    fn()
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def _build_world(n: int = 12) -> WorldModelSystem:
    wm = WorldModelSystem(name="bench")
    wm.add_object("block", "base", {"length": 100, "width": 80, "height": 8}, material="steel")
    for i in range(n):
        wm.add_object(
            "block",
            f"bracket_{i}",
            {"length": 30, "width": 20, "height": 40},
            material="aluminum",
            pose=Transform.translation(0, 0, 60 + i * 50),
        )
        wm.add_object("hole", f"hole_{i}", {"radius": 3, "depth": 8})
    return wm


def bench_reason(wm: WorldModelSystem, reps: int) -> None:
    print("\n== reasoner capabilities (12-part world) ==")
    a, b = wm.graph.objects[0], wm.graph.objects[1]
    load = LoadCase("static", [BoundaryCondition(kind="force", magnitude=2000.0)])
    cases = [
        ("clearance", lambda: wm.reason("clearance", a=a, b=b, minimum=2.0, axis="z")),
        ("safety", lambda: wm.reason("safety", object=b, load_case=load)),
        ("stability", lambda: wm.reason("stability", object=b)),
        ("mass", lambda: wm.reason("mass", limit_kg=50.0)),
        ("affordances", lambda: wm.reason("affordances", object=wm.graph.objects[-1])),
        ("assembly", lambda: wm.assembly_validator.validate(WorldAssembly("w", wm.graph.objects))),
    ]
    print(f"{'capability':>14} | {'ms/call':>10}")
    for name, fn in cases:
        ms = time_fn(fn, reps) * 1e3
        print(f"{name:>14} | {ms:>10.3f}")


def bench_simulate(wm: WorldModelSystem, reps: int) -> None:
    print("\n== forward-kinematics simulation ==")
    mech = Mechanism("arm")
    for i in range(6):
        mech.add_link(f"l{i}")
    for i in range(5):
        mech.add_joint(Joint(f"j{i}", "REVOLUTE", f"l{i}", f"l{i + 1}"))
    states = {f"j{i}": 0.1 * i for i in range(5)}
    offsets = {f"l{i}": 100.0 for i in range(1, 6)}
    ms = time_fn(lambda: wm.simulator.simulate(mech, states, link_offsets=offsets), reps) * 1e3
    print(f"{'simulate':>14} | {ms:>10.3f} ms/call")


def bench_plan(wm: WorldModelSystem, reps: int) -> None:
    print("\n== plan + execute ==")
    ms = (
        time_fn(
            lambda: wm.planner.execute(wm.planner.plan("assemble a bracket"), wm.graph),
            reps,
        )
        * 1e3
    )
    print(f"{'plan+execute':>14} | {ms:>10.3f} ms/call")


def bench_embed(wm: WorldModelSystem, reps: int) -> None:
    print("\n== world -> multimodal embedding ==")
    from cadgenesis.config import MultimodalConfig
    from cadgenesis.multimodal import MultimodalSystem

    sys = MultimodalSystem.from_config(MultimodalConfig())
    integration = WorldModelIntegration()
    ms = time_fn(lambda: integration.embed_world(wm.graph, sys), reps) * 1e3
    print(f"{'embed_world':>14} | {ms:>10.3f} ms/call")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", nargs="+", default=SECTIONS, choices=SECTIONS)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()
    wm = _build_world()
    for section in args.sections:
        if section == "reason":
            bench_reason(wm, args.reps)
        elif section == "simulate":
            bench_simulate(wm, args.reps)
        elif section == "plan":
            bench_plan(wm, args.reps)
        elif section == "embed":
            bench_embed(wm, args.reps)


if __name__ == "__main__":
    main()
