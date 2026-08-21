"""
cadgenesis.cli.benchmark
========================
``python -m cadgenesis.cli.benchmark`` — run the platform benchmark suites.

Invokes the standalone ``benchmarks/*.py`` runners (foundation, tokenizer,
attention, reasoning, execution, memory, world-model, multimodal, agent) and
writes a JSON summary.  Complements ``python -m cadgenesis.research`` for
full experiment tracking.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

BENCHMARKS = [
    "attention_benchmarks",
    "tokenizer_benchmarks",
    "foundation_benchmarks",
    "reasoning_benchmarks",
    "execution_benchmarks",
    "memory_benchmarks",
    "world_model_benchmarks",
    "multimodal_benchmarks",
    "agent_benchmarks",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cadgenesis benchmark", description="Run benchmark suites"
    )
    parser.add_argument(
        "--suites", nargs="+", default=None, help=f"one or more of: {', '.join(BENCHMARKS)}"
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--root", default="benchmarks", help="benchmarks directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    suites = args.suites or BENCHMARKS
    results: dict[str, dict] = {}
    exit_code = 0
    for suite in suites:
        script = os.path.join(args.root, f"{suite}.py")
        if not os.path.exists(script):
            print(f"[error] missing benchmark script: {script}")
            exit_code = 2
            continue
        print(f"== running {suite} (reps={args.reps}) ==")
        try:
            proc = subprocess.run(
                [sys.executable, script, "--reps", str(args.reps)],
                capture_output=True,
                text=True,
                timeout=3600,
            )
            results[suite] = {
                "returncode": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-2000:],
            }
            if proc.returncode != 0:
                exit_code = 1
        except subprocess.TimeoutExpired:
            results[suite] = {"returncode": -1, "error": "timeout"}
            exit_code = 1
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"summary written: {args.output}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
