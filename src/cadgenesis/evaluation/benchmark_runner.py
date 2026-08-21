"""cadgenesis.evaluation.benchmark_runner
======================================
Benchmark harness runner for the pillar stack.

The helper is intentionally lightweight and dependency-safe. It exercises the
core public entrypoints that are already implemented in the repository and
returns a compact report suitable for local benchmarking or smoke tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

from cadgenesis.config import CADConfig
from cadgenesis.evaluation.memory_metrics import run_memory_benchmark
from cadgenesis.evaluation.reasoning_metrics import run_reasoning_benchmark
from cadgenesis.evaluation.world_model_metrics import run_world_benchmark
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


@dataclass
class BenchmarkSummary:
    """A simple summary payload returned by the benchmark runner."""

    name: str
    passed: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _smoke_transformer() -> dict[str, Any]:
    cfg = CADConfig.mini()
    model = GeometryAwareTransformer(cfg)
    src = torch.randint(0, 32, (2, 8))
    tgt_in = torch.randint(0, 32, (2, 6))
    tgt_type = torch.zeros((2, 6), dtype=torch.long)
    with torch.no_grad():
        logits, confidence = model(src, tgt_in, tgt_type)
    return {
        "logits_shape": list(logits.shape),
        "confidence_shape": list(confidence.shape),
    }


def run_pillar_benchmark() -> BenchmarkSummary:
    """Run a lightweight benchmark over the core pillar entrypoints."""

    checks: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("transformer", _smoke_transformer),
        ("memory", lambda: run_memory_benchmark(retrieval_batches=[])),
        ("reasoning", lambda: run_reasoning_benchmark()),
        (
            "world_model",
            lambda: run_world_benchmark(
                spatial_checks=[],
                safety_checks=[],
                assembly_checks=[],
                path_checks=[],
                plan_outcomes=[],
            ),
        ),
    ]

    results: dict[str, Any] = {}
    for name, fn in checks:
        try:
            results[name] = fn()
        except Exception as exc:  # pragma: no cover - benchmark robustness
            results[name] = {"error": str(exc)}

    passed = all(not isinstance(value, dict) or "error" not in value for value in results.values())
    return BenchmarkSummary(name="pillar_stack_smoke", passed=passed, details=results)
