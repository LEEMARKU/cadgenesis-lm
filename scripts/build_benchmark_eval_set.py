"""
scripts/build_benchmark_eval_set.py
===================================
Builds the held-out benchmark evaluation set for Phase 10 (H. Benchmarks):
`data/benchmarks/eval_set.jsonl`.

The set is generated with a seed that is DISJOINT from the training-data
seed (curriculum used seed 0), so no program from `data/curriculum/` leaks
into the eval set. Records follow the manifest schema (`id`, `type`,
`text`, `cad`, `dsl`).

Usage:
    python scripts/build_benchmark_eval_set.py [--n 100] [--seed 999]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from cadgenesis.datasets.cad_program_synth import build_synthetic_records

EVAL_SEED = 999
EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "benchmarks"

HAND_AUTHORED = [
    {
        "id": "bench_manual_001",
        "type": "manual",
        "dsl": "legacy",
        "text": "Bracket with two vertical plate flanges.",
        "cad": "SKETCH_RECT NUM_60 NUM_40 EXTRUDE NUM_10 BOX",
    },
    {
        "id": "bench_manual_002",
        "type": "manual",
        "dsl": "legacy",
        "text": "Base plate, ten millimeters thick.",
        "cad": "SKETCH_RECT NUM_100 NUM_60 EXTRUDE NUM_10 BOX",
    },
    {
        "id": "bench_manual_003",
        "type": "manual",
        "dsl": "legacy",
        "text": "Circular pedestal of 80 millimeters diameter.",
        "cad": "SKETCH_CIRCLE NUM_40 EXTRUDE NUM_50 CYLINDER",
    },
]


def _records_as_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        text = record.get("text") or record.get("description") or ""
        rows.append(
            {
                "id": record.get("id", f"eval_{len(rows)}"),
                "type": record.get("type", "other"),
                "dsl": record.get("dsl", "synth"),
                "text": text,
                "cad": record["cad"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="number of synthetic records")
    parser.add_argument("--seed", type=int, default=EVAL_SEED, help="RNG seed (must differ from curriculum seed 0)")
    args = parser.parse_args()

    if args.seed == 0:
        raise SystemExit("refusing seed 0: it matches the curriculum training seed (leakage risk)")

    synthetic = build_synthetic_records(args.n, seed=args.seed)
    rows = _records_as_rows(synthetic) + HAND_AUTHORED
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "eval_set.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with (EVAL_DIR / "eval_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "n": len(rows),
                "synthetic": len(synthetic),
                "manual": len(HAND_AUTHORED),
                "seed": args.seed,
                "leakage_policy": "seed disjoint from curriculum seed 0",
            },
            fh,
            indent=2,
        )
    print(f"wrote {out} ({len(rows)} records)")


if __name__ == "__main__":
    main()