"""
benchmarks/agent_benchmarks.py
==============================
Benchmarks for the Pillar 5 (v6.0) multi-agent platform.

Measures wall-clock time for:
* platform startup (fleet registration + lifecycle start),
* single-agent dispatch latency,
* full 8-stage pipeline runs,
* consensus resolution,
* layered shared-memory access.

Run with::

    python benchmarks/agent_benchmarks.py                  # all
    python benchmarks/agent_benchmarks.py --sections dispatch
"""

from __future__ import annotations

import argparse
import time

from cadgenesis.agents.orchestrator import AgentPlatform

SECTIONS = ("startup", "dispatch", "pipeline", "consensus", "memory")

GOAL = "plan and validate a cost-optimized assembly"


def time_fn(fn, reps: int) -> float:
    fn()
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def bench_startup(reps: int) -> None:
    print("\n== platform startup (18-agent fleet) ==")
    ms = time_fn(lambda: AgentPlatform().load_fleet(), reps) * 1e3
    print(f"{'startup':>14} | {ms:>10.3f} ms")


def bench_dispatch(reps: int) -> None:
    print("\n== single-agent dispatch ==")
    platform = AgentPlatform().load_fleet()
    ms = (
        time_fn(
            lambda: platform.dispatch("cost", "estimate", {"mass_kg": 1.5}),
            reps,
        )
        * 1e3
    )
    print(f"{'dispatch':>14} | {ms:>10.3f} ms")


def bench_pipeline(reps: int) -> None:
    print("\n== full 8-stage pipeline ==")
    platform = AgentPlatform().load_fleet()
    ms = time_fn(lambda: platform.submit_pipeline(GOAL), reps) * 1e3
    print(f"{'pipeline':>14} | {ms:>10.3f} ms")


def bench_consensus(reps: int) -> None:
    print("\n== fleet consensus ==")
    platform = AgentPlatform().load_fleet()
    ms = time_fn(lambda: platform.ask("is the design acceptable?"), reps) * 1e3
    print(f"{'consensus':>14} | {ms:>10.3f} ms")


def bench_memory(reps: int) -> None:
    print("\n== layered shared memory ==")
    platform = AgentPlatform().load_fleet()
    ms = time_fn(lambda: platform.share("working", "k", {"v": 1}), reps) * 1e3
    print(f"{'share':>14} | {ms:>10.3f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", nargs="+", default=SECTIONS, choices=SECTIONS)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()
    for section in args.sections:
        if section == "startup":
            bench_startup(args.reps)
        elif section == "dispatch":
            bench_dispatch(args.reps)
        elif section == "pipeline":
            bench_pipeline(args.reps)
        elif section == "consensus":
            bench_consensus(args.reps)
        elif section == "memory":
            bench_memory(args.reps)


if __name__ == "__main__":
    main()
