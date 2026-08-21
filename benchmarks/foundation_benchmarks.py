"""
benchmarks/foundation_benchmarks.py
===================================
Micro-benchmarks for the Pillar 1 (v6.0) Foundation-Model additions.

Measures wall-clock forward time for:
* sparse attention (local / sliding-window / block-sparse / mixed) vs the
  legacy quadratic ``math`` backend across sequence lengths,
* multi-scale attention,
* specialised MoE vs dense SwiGLU inside one transformer block,
* the five-stage hierarchical model with and without a computation budget
  (early exit), showing the expected latency win from dynamic routing.

Run with::

    python benchmarks/foundation_benchmarks.py                    # all
    python benchmarks/foundation_benchmarks.py --max-len 1024
    python benchmarks/foundation_benchmarks.py --section sparse
"""

from __future__ import annotations

import argparse
import time

import torch

from cadgenesis.config import CADConfig
from cadgenesis.transformer import build_sparse_attention
from cadgenesis.transformer.hierarchical_transformer import HierarchicalCADTransformer
from cadgenesis.transformer.multi_scale_attention import MultiScaleAttention
from cadgenesis.transformer.sparse_attention import SPARSE_PATTERNS
from cadgenesis.transformer.specialized_moe import SpecializedMoEFFN
from cadgenesis.transformer.transformer_block import SwiGLU

torch.manual_seed(0)

SECTIONS = ("sparse", "multi_scale", "moe", "hierarchical")


def time_forward(fn, reps: int) -> float:
    """Mean wall-clock seconds over ``reps`` forward-only runs."""
    fn()  # warm-up (allocations / cache fills)
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def bench_sparse(d_model: int, heads: int, max_len: int, reps: int) -> None:
    print("\n== sparse attention (forward, no-grad) ==")
    cols = ["quadratic", *list(SPARSE_PATTERNS)]
    header = f"{'seq_len':>8} | " + " | ".join(f"{c:>11}" for c in cols)
    print(header)
    print("-" * len(header))

    @torch.no_grad()
    def make(pattern):
        return build_sparse_attention(
            pattern,
            d_model=d_model,
            num_heads=heads,
            window_size=128,
            num_global_tokens=16,
            block_size=32,
            dropout=0.0,
        )

    from cadgenesis.transformer.efficient_attention import build_self_attention

    for L in dict.fromkeys((64, 256, 1024, max_len)):  # dedupe when max_len repeats
        x = torch.randn(1, L, d_model)
        cells = []
        quadratic = build_self_attention("math", d_model, heads, dropout=0.0)
        cells.append(f"{time_forward(lambda q=quadratic, xx=x: q(xx), reps) * 1e3:>11.3f}ms")
        for pattern in SPARSE_PATTERNS:
            attn = make(pattern)
            cells.append(f"{time_forward(lambda a=attn, xx=x: a(xx), reps) * 1e3:>11.3f}ms")
        print(f"{L:>8} | " + " | ".join(cells))
    print("\nNote: patterns restrict *which* (query, key) pairs may attend, not the")
    print("score-matrix size; this mask-based implementation keeps wall-clock time")
    print("comparable across patterns. The sub-quadratic win needs block-sparse")
    print("kernels (Triton/xformers) at long sequences.")


def bench_multi_scale(d_model: int, heads: int, max_len: int, reps: int) -> None:
    print("\n== multi-scale attention (local + medium + global heads) ==")
    attn = MultiScaleAttention(d_model=d_model, num_heads=heads, dropout=0.0)
    print(f"head split: {attn.scale_report}")
    header = f"{'seq_len':>8} | {'multi_scale':>14}"
    print(header)
    print("-" * len(header))
    for L in dict.fromkeys((64, 256, 1024, max_len)):
        x = torch.randn(1, L, d_model)
        ms = time_forward(lambda xx=x: attn(xx), reps) * 1e3
        print(f"{L:>8} | {ms:>14.3f}ms")


def bench_moe(d_model: int, reps: int) -> None:
    print("\n== FFN: dense SwiGLU vs specialised MoE (5 domains x 2 experts) ==")
    dense = SwiGLU(d_model, 4 * d_model)
    moe = SpecializedMoEFFN(d_model=d_model, experts_per_domain=2, top_k=2)
    x = torch.randn(1, 1024, d_model)

    def run_dense():
        dense(x).sum().backward()

    def run_moe():
        out = moe(x)
        (out.sum() + moe.get_aux_loss()).backward()

    for name, fn in (("dense swiglu", run_dense), ("specialized moe", run_moe)):
        t = time_forward(fn, reps) * 1e3
        print(f"  {name:<18} {t:>10.3f}ms  (fwd+bwd)")
    print(f"  moe experts: {moe.num_experts}, top_k: {moe.top_k}")


def bench_hierarchical(reps: int) -> None:
    print("\n== hierarchical model: full vs budgeted (dynamic routing) ==")
    src = torch.randint(0, 64, (2, 24))
    tgt = torch.randint(0, 64, (2, 16))
    tgt_type = torch.randint(0, 3, (2, 16))

    def build_model(budget: float) -> HierarchicalCADTransformer:
        cfg = CADConfig.mini()
        cfg.model.use_hierarchical_transformer = True
        cfg.model.computation_budget = budget
        model = HierarchicalCADTransformer(cfg)
        model.eval()
        return model

    @torch.no_grad()
    def fwd(model) -> None:
        model(src, tgt, tgt_type)

    for full in (True, False):
        model = build_model(1.0 if full else 0.5)
        fwd(model)  # warm-up (caches / allocations)
        t = time_forward(lambda m=model: fwd(m), reps) * 1e3
        report = model.routing.report()
        label = "full   " if full else "budget "
        print(
            f"  [{label}] {t:>10.3f}ms  layers={report['layers_executed']:>2}/"
            f"{report['total_layers']}  savings={report['savings_fraction']:.0%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pillar 1 foundation benchmarks")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--section", type=str, default=None, choices=SECTIONS, help="Restrict to one section."
    )
    args = parser.parse_args()

    print(
        f"device={'cuda' if torch.cuda.is_available() else 'cpu'}  "
        f"d_model={args.d_model}  heads={args.heads}  reps={args.reps}"
    )
    sections = [args.section] if args.section else list(SECTIONS)
    if "sparse" in sections:
        bench_sparse(args.d_model, args.heads, args.max_len, args.reps)
    if "multi_scale" in sections:
        bench_multi_scale(args.d_model, args.heads, args.max_len, args.reps)
    if "moe" in sections:
        bench_moe(args.d_model, args.reps)
    if "hierarchical" in sections:
        bench_hierarchical(args.reps)


if __name__ == "__main__":
    main()
