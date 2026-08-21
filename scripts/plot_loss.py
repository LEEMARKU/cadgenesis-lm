"""
scripts/plot_loss.py
====================
Renders training loss curves from a `metrics.jsonl` file produced by
`MetricsJsonlCallback` (pre-training gate: loss curves must be persisted
and inspectable — no fake curves).

Outputs:
  - an ASCII line chart to stdout,
  - `reports/loss_curve_<run>.md` markdown table.

Usage:
    python scripts/plot_loss.py --file checkpoints/run1/metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))


def load_curve(path: Path) -> dict[str, list[dict[str, float]]]:
    """Group JSONL rows by event kind -> list of {epoch, step, loss}."""
    curves: dict[str, list[dict[str, float]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("event", "")
            metrics = row.get("metrics") or {}
            loss = metrics.get("loss")
            if loss is None:
                continue
            curves.setdefault(kind, []).append(
                {
                    "epoch": float(row.get("epoch", 0)),
                    "step": float(row.get("step", 0)),
                    "loss": float(loss),
                }
            )
    return curves


def ascii_chart(points: list[float], width: int = 60, height: int = 12) -> list[str]:
    """Minimal dependency-free ASCII line chart."""
    if not points:
        return ["(no points)"]
    lo, hi = min(points), max(points)
    span = (hi - lo) or 1.0
    grid: list[list[str]] = [[" "] * width for _ in range(height)]
    n = len(points)
    for i, value in enumerate(points):
        col = int(i * (width - 1) / max(1, n - 1))
        row = height - 1 - int((value - lo) / span * (height - 1))
        row = max(0, min(height - 1, row))
        grid[row][col] = "*"
    lines = [f"{hi:8.4f} |" + "".join(row) for row in grid]
    lines.append(f"{lo:8.4f} |" + "-" * width)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True, help="path to metrics.jsonl")
    parser.add_argument("--out", type=Path, default=None, help="markdown report path")
    args = parser.parse_args()

    if not args.file.exists():
        raise SystemExit(f"metrics file not found: {args.file}")

    curves = load_curve(args.file)
    if not curves:
        raise SystemExit(f"no loss-bearing events in {args.file}")

    for kind, points in sorted(curves.items()):
        losses = [p["loss"] for p in points]
        print(f"== {kind} ({len(points)} events, last loss {losses[-1]:.6f}) ==")
        for line in ascii_chart(losses):
            print(line)
        print()

    out = args.out or (_ROOT / "reports" / f"loss_curve_{args.file.stem}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write(f"# Loss curve from {args.file.name}\n\n")
        for kind, points in sorted(curves.items()):
            handle.write(f"## {kind}\n\n")
            handle.write("| epoch | step | loss |\n| --- | --- | --- |\n")
            for point in points:
                handle.write(
                    f"| {point['epoch']:.0f} | {point['step']:.0f} | {point['loss']:.6f} |\n"
                )
            handle.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()