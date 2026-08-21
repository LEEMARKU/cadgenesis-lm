"""
cadgenesis.smoke.runner
=======================
Runs the four G15 CPU smoke stages end-to-end and writes
`reports/SMOKE_TEST_RESULTS.md`.

Usage (from repo root):
    python scripts/smoke/run_all.py [--out reports/SMOKE_TEST_RESULTS.md]
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from cadgenesis.smoke.stages import (
    SMOKE_OUTPUTS,
    stage1_forward_backward,
    stage2_tiny_dataset,
    stage3_overfit,
    stage4_dev_run,
)

REPORTS_DIR = Path("reports")


def _render_markdown(results: list[dict[str, Any]]) -> str:
    lines = ["# SMOKE TEST RESULTS (Phase 12, CPU)", ""]
    lines.append("| stage | status | duration_s | key metric |")
    lines.append("| --- | --- | --- | --- |")
    keys = {
        "stage1_forward_backward": "loss",
        "stage2_tiny_dataset": "final_val_loss",
        "stage3_overfit": "final_loss",
        "stage4_dev_run": "best_val_loss",
    }
    for result in results:
        name = result["stage"]
        key = keys.get(name, "loss")
        value = result.get("result", {}).get(key, "n/a")
        lines.append(
            f"| {name} | {result['status']} | {result['duration_s']:.1f} | {value} |"
        )
    lines.append("")
    for result in results:
        name = result["stage"]
        lines.append(f"## {name}")
        lines.append("")
        for k, v in sorted(result["result"].items()):
            lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if all(r["status"] == "PASS" for r in results):
        lines.append("**ALL STAGES PASS** - training pipeline is ready for the PRE-TRAINING READINESS REVIEW.")
    else:
        lines.append("**AT LEAST ONE STAGE FAILED** - do NOT proceed to the readiness review.")
    return "\n".join(lines)


def run_all(out: str | Path | None = None) -> dict[str, Any]:
    stages = {
        "stage1_forward_backward": lambda: stage1_forward_backward(),
        "stage2_tiny_dataset": lambda: stage2_tiny_dataset(),
        "stage3_overfit": lambda: stage3_overfit(),
        "stage4_dev_run": lambda: stage4_dev_run(out_dir=SMOKE_OUTPUTS / "stage4"),
    }
    results: list[dict[str, Any]] = []
    for name, fn in stages.items():
        started = time.time()
        try:
            result = fn()
            status = result.get("status", "FAIL")
        except Exception as exc:  # stage crash -> FAIL, keep going
            result = {"error": f"{type(exc).__name__}: {exc}"}
            status = "FAIL"
        results.append(
            {
                "stage": name,
                "status": status,
                "duration_s": round(time.time() - started, 2),
                "result": result,
            }
        )
        print(f"[{status}] {name} ({results[-1]['duration_s']:.1f}s)")

    target = Path(out or REPORTS_DIR / "SMOKE_TEST_RESULTS.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_render_markdown(results), encoding="utf-8")
    print(f"wrote {target}")
    return {"results": results, "all_pass": all(r["status"] == "PASS" for r in results)}


def main() -> None:
    summary = run_all()
    print("ALL STAGES PASS" if summary["all_pass"] else "SOME STAGES FAILED")


if __name__ == "__main__":
    main()