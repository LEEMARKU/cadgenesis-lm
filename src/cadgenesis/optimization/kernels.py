"""cadgenesis.optimization.kernels
===============================
Custom fused kernels (attention, MoE) for latency reduction.

Provers kernel-level operations for accelerating attention and Mixture-of-Experts
patterns in CADGenesis-LM inference.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusedAttention(nn.Module):
    """Fused attention kernel combining QKV projection and scaled dot-product.

    Optimized for CADGenesis-LM attention patterns with optional causal masking.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int | None = None,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim or dim // n_heads
        self.causal = causal
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of fused attention.

        - ``x``: (B, T, D)
        - ``mask``: (B, 1, T, T) causal or padding mask
        Returns: (B, T, D)
        """
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
            attn = attn.masked_fill(causal_mask == 1, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, T, self.dim)
        return self.o_proj(out)


class MoEKernel(nn.Module):
    """Mixture-of-Experts kernel routing.

    Routes input tokens to a subset of experts and combines outputs.
    """

    def __init__(self, dim: int, n_experts: int, n_active: int = 2, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.n_active = n_active
        self.experts = nn.ModuleList([nn.Linear(dim, dim) for _ in range(n_experts)])
        self.gate = nn.Linear(dim, n_experts)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of MoE kernel.

        - ``x``: (B, T, D)
        Returns: (B, T, D)
        """
        B, T, _ = x.shape
        # Gate: compute routing probabilities
        gate_logits = self.gate(x).view(B, T, self.n_experts)
        gates = F.softmax(gate_logits, dim=-1)
        # Select top-k experts
        top_gates, top_indices = torch.topk(gates, self.n_active, dim=-1)
        # Zero-out non-selected gates
        mask = torch.zeros_like(gates)
        top_gates_norm = top_gates / (top_gates.sum(dim=-1, keepdim=True) + 1e-8)
        mask.scatter_(-1, top_indices, top_gates_norm)
        # Combine expert outputs
        expert_outputs = sum(
            expert(x).unsqueeze(-2) * mask.unsqueeze(-1) for expert in self.experts
        )
        return expert_outputs.sum(dim=-2) + self.dropout(expert_outputs).sum(dim=-2)


__all__ = [
    "FusedAttention",
    "MoEKernel",
]
