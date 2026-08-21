"""
cadgenesis.cli.eval
===================
``python -m cadgenesis.cli.eval`` — evaluate a checkpoint.

Runs the implemented evaluation metrics harnesses (reasoning, execution,
memory, world-model, agent, multimodal) over a loaded model/checkpoint and
prints a summary table; optional JSON output.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("cadgenesis.cli.eval")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cadgenesis eval", description="Evaluate model checkpoints"
    )
    parser.add_argument("--model", default=None, help="checkpoint path (default: CADGENESIS_MODEL)")
    parser.add_argument(
        "--suites",
        nargs="+",
        default=["reasoning", "execution", "memory", "world_model"],
        help="suites: reasoning execution memory world_model agent multimodal",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = args.model or os.environ.get("CADGENESIS_MODEL")
    if not checkpoint or not os.path.exists(checkpoint):
        print(f"error: model checkpoint not found: {checkpoint}")
        return 2

    from cadgenesis.evaluation import (
        agent_metrics,
        execution_metrics,
        memory_metrics,
        multimodal_metrics,
        reasoning_metrics,
        world_model_metrics,
    )

    runners: dict[str, Callable[..., Any]] = {
        "reasoning": reasoning_metrics.run_reasoning_benchmark,
        "execution": execution_metrics.run_execution_benchmark,
        "memory": memory_metrics.run_memory_benchmark,
        "world_model": world_model_metrics.run_world_benchmark,
        "agent": agent_metrics.run_agent_benchmark,
        "multimodal": multimodal_metrics.run_retrieval_benchmark,
    }
    results: dict[str, dict] = {}
    for suite in args.suites:
        runner = runners.get(suite)
        if runner is None:
            print(f"[warn] unknown suite {suite!r}; skipping")
            continue
        try:
            results[suite] = runner(checkpoint=checkpoint, seed=args.seed)
        except Exception as exc:
            print(f"[error] suite {suite} failed: {exc}")
            results[suite] = {"error": str(exc)}
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        for suite, data in results.items():
            print(f"== {suite} ==")
            if isinstance(data, dict):
                for key, value in data.items():
                    print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
