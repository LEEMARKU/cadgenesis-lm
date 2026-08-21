"""
cadgenesis.transformer.ssm
==========================
Hybrid state-space (linear-time) layer: Gated DeltaNet.

2025-2026 frontier: pure softmax attention is quadratic in context length, so
modern architectures (Mamba-2, Griffin, Samba, Gated DeltaNet) interleave a
*linear-time recurrent* layer with full attention every few blocks.  This gives
linear-time long-context handling and O(1) recurrent memory during decoding,
which is especially valuable on CPU.

This module implements the Gated DeltaNet recurrence (Yang et al., NeurIPS
2024):

    h_t = (1 - delta_t) * h_{t-1} + delta_t * (k_t (x) v_t)      (outer product)
    o_t = q_t (.) h_t

with a *data-dependent* decay ``delta_t = sigmoid(beta_h + k_t . g_h)``, giving
the model the ability to copy/erase information in state — the property that
makes DeltaNet a practical replacement for attention at long range.

``forward`` runs the exact recurrence for training; ``forward_cached`` runs a
single step with an explicit recurrent state for KV-cache-style decoding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedDeltaNet(nn.Module):
    """
    Multi-head Gated DeltaNet layer.

    Parameters
    ----------
    d_model : int
        Model dimension (must be divisible by ``heads``).
    heads : int
        Number of parallel state heads.
    dropout : float
        Dropout applied to the output.

    Shapes
    ------
    forward(x: (B, T, d_model)) -> (B, T, d_model)
    forward_cached(x: (B, 1, d_model), state: (B, heads, d_model//heads))
        -> (out, state')   single recurrent step.
    """

    def __init__(self, d_model: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by heads={heads}.")
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        # Input projections (query / key / value).
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        # Data-dependent decay gate: beta (per-head) + key-gating weights.
        self.beta = nn.Parameter(torch.randn(heads) * 0.5)
        self.w_gate = nn.Parameter(torch.randn(heads, self.head_dim) * 0.01)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        return q, k, v

    def _decay(self, k: torch.Tensor) -> torch.Tensor:
        """
        Data-dependent decay per head: delta = sigmoid(beta + k.gate).
        k: (B, T, heads, head_dim) -> (B, T, heads, 1)
        """
        gate = torch.einsum("bthd,hd->bth", k, self.w_gate).unsqueeze(-1)
        return torch.sigmoid(self.beta.view(1, 1, self.heads, 1) + gate)

    def _initial_state(self, batch: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch, self.heads, self.head_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Exact sequential recurrence (training / full-sequence).
        x: (B, T, d_model) -> (B, T, d_model)
        """
        B, T, _ = x.shape
        q, k, v = self._qkv(x)
        q = q.view(B, T, self.heads, self.head_dim)
        k = k.view(B, T, self.heads, self.head_dim)
        v = v.view(B, T, self.heads, self.head_dim)
        decay = self._decay(k)  # (B, T, heads, 1)

        h = self._initial_state(B, x.device)  # (B, heads, head_dim)
        outs = []
        for t in range(T):
            kt, vt, dt = k[:, t], v[:, t], decay[:, t]
            h = h * (1.0 - dt) + dt * (kt * vt)
            outs.append((q[:, t] * h).reshape(B, self.d_model))
        out = torch.stack(outs, dim=1)  # (B, T, d_model)
        return self.dropout(self.out_proj(out))

    def forward_cached(
        self,
        x: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step recurrence for incremental decoding.
        x: (B, 1, d_model); state: (B, heads, head_dim).
        Returns (out: (B, 1, d_model), state': (B, heads, head_dim)).

        Identical to :meth:`forward` on the last step (same projections,
        decay, dropout, and output projection), so a full forward equals the
        step-by-step cached replay in evaluation mode.  Gradients flow
        through the recurrence; callers that decode in inference wrap the
        whole loop in ``torch.no_grad()`` (as ``decode_step`` does).
        """
        B = x.shape[0]
        q, k, v = self._qkv(x)
        q = q.view(B, self.heads, self.head_dim)
        k = k.view(B, self.heads, self.head_dim)
        v = v.view(B, self.heads, self.head_dim)
        decay = self._decay(k.unsqueeze(1))[:, 0]  # (B, heads, 1)
        state = state * (1.0 - decay) + decay * (k * v)
        out = self.out_proj((q * state).reshape(B, self.d_model))
        return self.dropout(out).unsqueeze(1), state


def add_ssm_blocks(
    block_count: int,
    d_model: int,
    every_n: int,
    heads: int = 4,
    dropout: float = 0.1,
) -> list[GatedDeltaNet | None]:
    """
    Build an SSM interleave plan: returns a list of length ``block_count`` where
    position ``i`` holds a :class:`GatedDeltaNet` when ``(i + 1) % every_n == 0``
    else ``None``.  E.g. ``every_n=3`` places an SSM layer after blocks 2, 5, 8…
    """
    if every_n < 1:
        raise ValueError("every_n must be >= 1.")
    return [
        GatedDeltaNet(d_model, heads=heads, dropout=dropout) if (i + 1) % every_n == 0 else None
        for i in range(block_count)
    ]
