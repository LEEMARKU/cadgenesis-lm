"""
benchmarks/attention_benchmarks.py
==================================
Micro-benchmark for the efficient attention backends of CADGenesis-LM v2.0.

Compares ``math`` (legacy quadratic), ``sdpa`` (torch fused kernel) and
``linear`` (Performer-style random features) self-attention across sequence
lengths.  On CPU the SDPA backend uses the math kernel internally, so the most
meaningful comparison on this machine is quadratic (math/sdpa) vs linear.

Run with::

    python benchmarks/attention_benchmarks.py            # all backends
    python benchmarks/attention_benchmarks.py --max-len 512
    python benchmarks/attention_benchmarks.py --backend linear
"""

from __future__ import annotations

import argparse
import time

import torch

from cadgenesis.transformer.efficient_attention import BACKENDS, build_self_attention

torch.manual_seed(0)


def bench_one(backend: str, d_model: int, seq_len: int, num_heads: int, reps: int) -> float:
    """Return mean forward+backward wall-clock seconds over `reps` runs."""
    attn = build_self_attention(backend, d_model, num_heads, dropout=0.0)
    attn.train()
    x = torch.randn(1, seq_len, d_model)
    # Warm-up (allocates caches / JITs kernels).
    attn(x).sum().backward()

    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = attn(x)
        out.sum().backward()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def main() -> None:
    parser = argparse.ArgumentParser(description="Attention backend micro-benchmark")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Restrict to one backend (default: all).",
    )
    args = parser.parse_args()

    seq_lens = [64, 128, 256, args.max_len]
    backends = [args.backend] if args.backend else list(BACKENDS)
    if args.backend and args.backend not in BACKENDS:
        raise SystemExit(f"Unknown backend {args.backend!r}; choose from {BACKENDS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  d_model={args.d_model}  heads={args.heads}  reps={args.reps}\n")
    header = f"{'seq_len':>8} | " + " | ".join(f"{b:>10}" for b in backends)
    print(header)
    print("-" * len(header))

    for L in seq_lens:
        cells = []
        for b in backends:
            try:
                t = bench_one(b, args.d_model, L, args.heads, args.reps)
                cells.append(f"{t * 1e3:>9.3f}ms")
            except torch.OutOfMemoryError:  # noqa: PERF203 - OOM probing per backend is the benchmark
                cells.append(f"{'OOM':>10}")
        print(f"{L:>8} | " + " | ".join(cells))

    print("\nNote: 'math' and 'sdpa' are both quadratic; 'sdpa' engages fused")
    print("kernels on CUDA. 'linear' is O(seq_len) and shines at long sequences.")


if __name__ == "__main__":
    main()
