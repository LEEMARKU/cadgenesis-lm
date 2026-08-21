"""
tools/profile_foundation.py
============================
Per-block profiling for the Pillar 1 (v6.0) hierarchical transformer.

Wraps every planner/geometry/constraint/execution/validation block with
wall-clock pre/post hooks, accumulates per-block times across the whole forward
pass, and dumps a per-stage, per-layer breakdown plus the dynamic-routing
telemetry (layers executed / early-exit reason / savings).

Useful for verifying that computation budgeting actually skips layers and for
spotting pathological blocks.

Run with::

    python tools/profile_foundation.py
    python tools/profile_foundation.py --budget 0.4 --early-exit 0.9
"""

from __future__ import annotations

import argparse
import time

import torch

from cadgenesis.config import CADConfig
from cadgenesis.transformer.hierarchical_transformer import STAGE_NAMES, HierarchicalCADTransformer


def profile(
    budget: float,
    early_exit: float,
    batch: int,
    src_len: int,
    tgt_len: int,
    seed: int,
) -> None:
    torch.manual_seed(seed)

    cfg = CADConfig.mini()
    cfg.model.use_hierarchical_transformer = True
    cfg.model.computation_budget = budget
    cfg.model.early_exit_threshold = early_exit
    model = HierarchicalCADTransformer(cfg)
    model.eval()

    src = torch.randint(0, 64, (batch, src_len))
    tgt = torch.randint(0, 64, (batch, tgt_len))
    tgt_type = torch.randint(0, 3, (batch, tgt_len))

    # Per-stage accumulator: {stage: [layer_ms, ...]}.
    stage_ms: dict[str, list[float]] = {s: [] for s in STAGE_NAMES}
    for stage in STAGE_NAMES:
        blocks = getattr(model, f"{stage}_blocks")
        for _ in blocks:
            stage_ms[stage].append(0.0)

    handles = []
    for stage in STAGE_NAMES:
        blocks = getattr(model, f"{stage}_blocks")
        for i, block in enumerate(blocks):
            state: dict[str, float] = {"t0": 0.0}

            def pre_hook(module, args, state=state) -> None:
                state["t0"] = time.perf_counter()

            def post_hook(module, args, output, state=state, stage=stage, i=i) -> None:
                stage_ms[stage][i] += time.perf_counter() - state["t0"]

            handles.append(block.register_forward_pre_hook(pre_hook))
            handles.append(block.register_forward_hook(post_hook))

    with torch.no_grad():
        model(src, tgt, tgt_type)

    for h in handles:
        h.remove()

    print(f"\nHierarchical profile  (budget={budget}, early_exit={early_exit})")
    print(f"{'stage':>12} | {'layer':>4} | {'ms':>10} | {'% of run':>9}")
    print("-" * 42)
    total = sum(t for stage in stage_ms.values() for t in stage)
    for stage in STAGE_NAMES:
        for i, ms in enumerate(stage_ms[stage]):
            pct = 100.0 * ms / total if total > 0 else 0.0
            print(f"{stage:>12} | {i:>4} | {ms * 1e3:>10.3f} | {pct:>8.1f}%")
    print(f"{'TOTAL':>12} | {'':>4} | {total * 1e3:>10.3f}")

    report = model.routing.report()
    print("\nRouting telemetry:")
    for k, v in report.items():
        print(f"  {k:<18}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile hierarchical transformer blocks")
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--early-exit", type=float, default=0.0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--src-len", type=int, default=24)
    parser.add_argument("--tgt-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    profile(args.budget, args.early_exit, args.batch, args.src_len, args.tgt_len, args.seed)


if __name__ == "__main__":
    main()
