"""
scripts/run_benchmarks.py
=========================
Runs the CAD benchmark suite over the held-out eval set and writes
`reports/BENCHMARK_REPORT.md` (baselines + model when available).

This is part of the pre-training gate's Phase 10 (benchmark baseline);
it does NOT start any training.

Usage:
    python scripts/run_benchmarks.py [--eval data/benchmarks/eval_set.jsonl] [--max-len 64]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from cadgenesis.evaluation.cad_bench import (
    CADBenchItem,
    CADBenchmark,
    FrequencyBaseline,
    ProgramOracle,
    RandomBaseline,
    write_benchmark_report,
)
from cadgenesis.tokenizer.cad_tokenizer import AutonomousCADTokenizer

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _load_items(path: Path, tokenizer: AutonomousCADTokenizer) -> list[CADBenchItem]:
    items: list[CADBenchItem] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            cad = record.get("cad") or ""
            tokens = cad.split() if isinstance(cad, str) else list(cad)
            reference_ids = tokenizer.encode_cad_sequence(tokens).cad_ids if tokens else None
            items.append(
                CADBenchItem(
                    prompt=record["text"],
                    reference_ids=reference_ids,
                )
            )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=Path("data/benchmarks/eval_set.jsonl"))
    parser.add_argument("--max-len", type=int, default=64)
    args = parser.parse_args()

    if not args.eval.exists():
        raise SystemExit(f"eval set not found: {args.eval} (run scripts/build_benchmark_eval_set.py first)")

    tokenizer = AutonomousCADTokenizer.build_mini()
    items = _load_items(args.eval, tokenizer)
    oracle = ProgramOracle(tokenizer=tokenizer)
    bench = CADBenchmark(items=items, oracle=oracle)

    # Random baseline: the same vocabulary scale as the tokenizer.
    random_base = RandomBaseline(vocab_size=tokenizer.vocab_size, seed=1234)
    # Frequency baseline: cycles the most common DSL tokens.
    freq_base = FrequencyBaseline(
        token_ids=list(range(4, 12)),  # head of the canonical vocab
        top_k=4,
    )

    results = [
        ("random", bench.evaluate_baseline(random_base, max_len=args.max_len)),
        ("frequency", bench.evaluate_baseline(freq_base, max_len=args.max_len)),
    ]

    try:
        from cadgenesis.inference.engine import CADInferenceEngine

        model = CADInferenceEngine()
        results.append(("model", bench.evaluate(model, max_len=args.max_len)))
    except Exception as exc:  # no model checkpoint yet — baselines only
        print(f"model eval skipped: {exc}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "BENCHMARK_REPORT.md"
    write_benchmark_report(str(out), results, title="CAD Benchmark Report (held-out eval set)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()