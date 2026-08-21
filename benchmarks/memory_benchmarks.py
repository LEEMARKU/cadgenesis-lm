"""
benchmarks/memory_benchmarks.py
===============================
Benchmarks for the Pillar 6 (v6.0) layer-integrated memory extensions.

Measures wall-clock time for:
* retrieval across pools (lexical, graph, symbolic, temporal, hybrid),
* routing (semantic + contextual modes),
* consolidation / summarization / compression,
* persistence (v1 save, v2 save, snapshot, append/replay),
* semantic→neural bridge rendering + attention layer forward pass.

Run with::

    python benchmarks/memory_benchmarks.py                     # all
    python benchmarks/memory_benchmarks.py --sections bridge
"""

from __future__ import annotations

import argparse
import time

import torch

from cadgenesis.memory.bridge import SemanticMemoryBridge
from cadgenesis.memory.compression import (
    EmbeddingCompressor,
    MemoryConsolidator,
    MemorySummarizer,
)
from cadgenesis.memory.long_term_memory import LongTermMemory
from cadgenesis.memory.memory_system import MemorySystem
from cadgenesis.memory.persistence import MemoryPersistence

SECTIONS = (
    "retrieval",
    "routing",
    "compression",
    "persistence",
    "bridge",
)

_QUERY = "tolerance material load stress feature"


def time_ms(fn, reps: int) -> str:
    fn()
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return f"{sum(times) / len(times) * 1000:8.3f} ms"


def report(label: str, fn, reps: int) -> None:
    print(f"{label:>12} | {time_ms(fn, reps)}")


def build_system() -> MemorySystem:
    system = MemorySystem()
    for pool in (
        "working",
        "project",
        "cad",
        "engineering",
        "manufacturing",
        "simulation",
    ):
        store = system.pool(pool)
        for i in range(40):
            store.add(
                f"{pool}:{i}",
                f"record {i} about {pool} tolerance material load stress feature",
                importance=0.5 + (i % 5) * 0.1,
                metadata={"kind": pool, "related": [f"{pool}:{(i + 1) % 40}"]},
            )
    return system


def bench_retrieval(system: MemorySystem, reps: int) -> None:
    retriever = system.retriever
    report("lexical", lambda: retriever.retrieve(_QUERY), reps)
    report("graph", lambda: retriever.graph_search("cad:0", hop_count=2), reps)
    report("symbolic", lambda: retriever.symbolic_search({"kind": "cad"}), reps)
    report(
        "hybrid",
        lambda: retriever.hybrid_retrieve(_QUERY, symbolic={"kind": "cad"}),
        reps,
    )


def bench_routing(system: MemorySystem, reps: int) -> None:
    router = system.router
    report("semantic", lambda: router.route(_QUERY), reps)
    report("context", lambda: router.route_by_context({"text": _QUERY}), reps)
    report("task", lambda: router.route_by_task("simulation"), reps)
    report("confidence", lambda: router.route_by_confidence(_QUERY, 0.9), reps)
    report("agent", lambda: router.route_by_agent("geometry"), reps)


def bench_compression(system: MemorySystem, reps: int) -> None:
    source = system.pool("working")
    target = LongTermMemory()
    summarizer = MemorySummarizer()
    consolidator = MemoryConsolidator()
    compressor = EmbeddingCompressor()
    values = [float(i) for i in range(256)]
    keys = list(source.keys())[:8]
    report("summarize", lambda: summarizer.summarize(source, keys), reps)
    report("consolidate", lambda: consolidator.consolidate(source, target), reps)
    report("embed-compress", lambda: compressor.compress(values, factor=4), reps)


def bench_persistence(system: MemorySystem, reps: int, root: str) -> None:
    store = system.pool("cad")
    stores = list(system.pools.values())
    report("v1-save", lambda: MemoryPersistence.save(store, f"{root}/pool.json"), reps)
    report(
        "v2-snapshot",
        lambda: MemoryPersistence.save_system(stores, f"{root}/sys"),
        reps,
    )
    report("append", lambda: MemoryPersistence.append(store, "k", "v", root), reps)


def bench_bridge(system: MemorySystem, reps: int) -> None:
    bridge = SemanticMemoryBridge(d_model=64)
    result = system.retriever.retrieve(_QUERY, top_k=8)
    layer = torch.nn.MultiheadAttention(64, 4, batch_first=True)
    x = torch.randn(1, 8, 64)
    vectors = bridge.to_vectors(result, batch_size=1)
    report("render", lambda: bridge.to_vectors(result, batch_size=1), reps)
    report("attend", lambda: layer(x, vectors, vectors)[0], reps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", nargs="*", choices=SECTIONS, default=list(SECTIONS))
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()
    system = build_system()
    root = "outputs/benchmarks/memory"
    for section in args.sections:
        print(f"\n== {section} ==")
        if section == "retrieval":
            bench_retrieval(system, args.reps)
        elif section == "routing":
            bench_routing(system, args.reps)
        elif section == "compression":
            bench_compression(system, args.reps)
        elif section == "persistence":
            bench_persistence(system, args.reps, root)
        elif section == "bridge":
            bench_bridge(system, args.reps)


if __name__ == "__main__":
    main()
