"""
A/B harness: lean architecture vs the full experimental stack.

Trains two identically-shaped mini models on the same toy pairs, one with
every experimental subsystem enabled (multi-agent, memory pools, neuro-symbolic
rules, RLAIF reward model + exotic attention heads) and one with only standard
self-attention + encoder-decoder cross-attention.  Reports parameter count,
loss trajectory, and wall-clock so the "de-over-engineering" decision can be
made with numbers instead of opinions.

Run:  PYTHONPATH=src python examples/ab_lean_vs_full.py
"""

from __future__ import annotations

import time

import torch

from cadgenesis.config import CADConfig
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


def _train(model, steps: int = 60, lr: float = 3e-3) -> list[float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        opt.zero_grad()
        src = torch.randint(0, 50, (4, 12))
        tgt_in = torch.randint(0, 30, (4, 6))
        tgt_type = torch.randint(0, 3, (4, 6))
        logits, _ = model(src, tgt_in, tgt_type)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tgt_in.reshape(-1),
            ignore_index=0,
        )
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def _n_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def main() -> None:
    torch.manual_seed(0)

    # Lean: standard heads only, no subsystems.
    lean_cfg = CADConfig.mini()
    lean_cfg.model.constraint_attn_heads = 0
    lean_cfg.model.memory_attn_heads = 0
    lean_cfg.model.agent_attn_heads = 0
    lean_cfg.model.uncertainty_attn_heads = 0
    lean_cfg.model.geometry_attn_heads = 2
    lean_cfg.model.use_multi_agent_system = False
    lean_cfg.model.use_memory_system = False
    lean_cfg.model.use_neuro_symbolic_reasoning = False
    lean_cfg.model.use_rlaf_reward_model = False
    lean = GeometryAwareTransformer(lean_cfg)

    # Full research stack (all subsystems + exotic heads).
    full = GeometryAwareTransformer(CADConfig.mini())

    print(f"params  lean={_n_params(lean):.2f}M  full={_n_params(full):.2f}M")

    t0 = time.perf_counter()
    lean_losses = _train(lean)
    lean_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    full_losses = _train(full)
    full_time = time.perf_counter() - t0

    print(f"time    lean={lean_time:.1f}s  full={full_time:.1f}s")
    print("step   lean_loss   full_loss")
    for i, (a, b) in enumerate(zip(lean_losses, full_losses, strict=True)):
        print(f"{i:4d}   {a:.4f}      {b:.4f}")
    print(
        f"final   lean={lean_losses[-1]:.4f}  full={full_losses[-1]:.4f}"
        f"  (lean-to-full ratio {full_losses[-1] / max(lean_losses[-1], 1e-9):.2f}x)"
    )


if __name__ == "__main__":
    main()
