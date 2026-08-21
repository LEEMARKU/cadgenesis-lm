"""
cadgenesis.transformer.attention
==========================
Six Specialized Attention Mechanisms & Multi-Head Mixture for CADGenesis-LM v2.0:
1. Self Attention — standard multi-head self-attention
2. Geometry Attention — specialized attention over B-Rep topology & continuous parameters
3. Constraint Attention — graph-aware attention enforcing geometric/dimensional constraints
4. Memory Attention — cross-attention over layer-integrated memory pools
5. Agent Attention — inter-agent context routing
6. Uncertainty Attention — entropy & confidence estimation head
7. MultiHeadAttentionMixture — adaptive mixture of expert attention heads with gating
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadgenesis.transformer.efficient_attention import build_self_attention
from cadgenesis.transformer.positional import RotaryEmbedding


def safe_softmax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax that never emits NaN for fully-masked rows.

    Rows whose scores are entirely ``-inf`` (e.g. fully-padding query rows in
    packed sequences) would otherwise produce ``0/0 = NaN`` that then pollutes
    every downstream layer.  Their outputs are excluded from the loss by the
    padding mask, so zeroing them is numerically safe and exactly equivalent
    to the idealised "attend to nothing" row.
    """
    probs = F.softmax(scores, dim=dim)
    fully_masked = torch.isneginf(scores).all(dim=dim, keepdim=True)
    if fully_masked.any():
        probs = probs.masked_fill(fully_masked, 0.0)
    return probs


def repair_fully_masked_rows(mask: torch.Tensor | None) -> torch.Tensor | None:
    """Guarantee every query row of an additive attention mask has a key.

    A query row whose scores are entirely ``-inf`` makes softmax return NaN
    (``0/0``).  ``pack_batch`` repairs dead rows at packing time, but merging a
    padding mask into a packed block mask (``encode``/``decode``) can re-kill
    them.  This helper re-applies the same repair after such merges:

    * self-attention masks ``(B, 1, T, T)``: the dead row's diagonal slot is
      opened (a padded query row attends itself; its output is masked out of
      the loss anyway).
    * cross-attention masks ``(B, 1, T, S)``: the whole dead row is opened to
      the first key column.

    Returns the input unchanged when no row needs repair.
    """
    if mask is None:
        return mask
    visible = mask.max(dim=-1).values > float("-inf")
    if visible.all():
        return mask
    m = mask.clone()
    dead = ~visible
    if m.shape[-2] == m.shape[-1]:
        b_idx, t_idx = dead.squeeze(1).nonzero(as_tuple=True)
        if b_idx.numel():
            m[b_idx, 0, t_idx, t_idx] = 0.0
    else:
        m[dead] = 0.0
    return m


class SelfAttention(nn.Module):
    """Standard Multi-Head Self-Attention with RoPE support."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_rope: bool = True,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if use_rope:
            q, k = self.rope(q, k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask

        probs = safe_softmax(scores)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

    def forward_cached(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        use_rope: bool = True,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Incremental KV-cache forward for one new token: x is (B, 1, C).

        ``past_kv``: (k, v) from previous steps (B, H, T_prev, D).
        Returns ``(out (B, 1, C), new_kv)`` with ``new_kv`` = concatenated
        history — exactly equivalent to running the full-sequence forward.
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if use_rope:
            q, k = self.rope(q, k, position_offset=position_offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask[:, :, -T:, :]

        probs = safe_softmax(scores)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), (k.detach(), v.detach())


class GeometryAttention(nn.Module):
    """
    Geometry Attention: Cross-attends CAD geometry/topology tokens against
    language & spatial features with adaptive geometric scale factors.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # Fixed (non-learnable) scale: a learnable per-head scale lets the
        # model push scores toward 0 and collapse cross-attention to a
        # src-invariant mean-pool (observed: trained ``geom_scale`` -> ~0.03
        # and the decoder stopped conditioning on the encoder).  Frozen at 1.0
        # so cross-attention must learn informative scores.
        self.geom_scale: torch.Tensor
        self.register_buffer("geom_scale", torch.ones(1, num_heads, 1, 1))
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T_q, C = query.shape
        T_k = key_value.shape[1]

        q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)) * self.geom_scale
        if attn_mask is not None:
            scores = scores + attn_mask

        probs = safe_softmax(scores)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T_q, C)
        return self.out_proj(out)

    def forward_cached(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Incremental cross-attention forward: ``query`` is one token (B, 1, C);
        ``past_kv`` (optional) holds the already-projected encoder K/V so the
        projection is only computed once.
        """
        B, T_q, C = query.shape
        if past_kv is None:
            T_k = key_value.shape[1]
            k = self.k_proj(key_value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(key_value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k, v = past_kv

        q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)) * self.geom_scale
        if attn_mask is not None:
            scores = scores + attn_mask[:, :, -T_q:, :]

        probs = safe_softmax(scores)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T_q, C)
        return self.out_proj(out), (k.detach(), v.detach())


class ConstraintAttention(nn.Module):
    """
    Constraint Attention: Graph-aware attention enforcing topological & parametric constraints
    using an adjacency/bias graph constraint matrix.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.constraint_bias_proj = nn.Linear(d_model, num_heads)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        constraint_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if constraint_mask is not None:
            # constraint_mask: (B, 1 or H, T, T)
            scores = scores + constraint_mask
        else:
            # Learned per-query constraint bias (activates constraint_bias_proj):
            # each token contributes a scalar head bias toward every key, which
            # the model can shape into hard "must attend" / "must ignore" priors.
            bias = self.constraint_bias_proj(x)  # (B, T, H)
            scores = scores + bias.transpose(1, 2).unsqueeze(-1)  # (B, H, T, 1)

        probs = safe_softmax(scores)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

    def forward_cached(
        self,
        x: torch.Tensor,
        constraint_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Incremental forward for one new token (B, 1, C); see SelfAttention."""
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if past_kv is None:
            k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k, v = past_kv
            k = torch.cat(
                [k, self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)], dim=2
            )
            v = torch.cat(
                [v, self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)], dim=2
            )

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if constraint_mask is not None:
            scores = scores + constraint_mask[:, :, -T:, :]
        else:
            bias = self.constraint_bias_proj(x)  # (B, 1, H)
            scores = scores + bias.transpose(1, 2).unsqueeze(-1)

        probs = safe_softmax(scores)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), (k.detach(), v.detach())


class MemoryAttention(nn.Module):
    """
    Memory Attention: Cross-attention querying short-term working memory
    and long-term vector memory slots.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory_bank: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, T, C)
        memory_bank: (B, M, C) or (1, M, C)
        """
        B, T, C = x.shape
        M = memory_bank.shape[1]
        if memory_bank.dim() == 2:
            memory_bank = memory_bank.unsqueeze(0).expand(B, -1, -1)

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory_bank).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory_bank).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask

        probs = safe_softmax(scores)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

    def forward_cached(
        self,
        x: torch.Tensor,
        memory_bank: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Incremental cross-attention over a (frozen) memory bank."""
        B, T, C = x.shape
        if past_kv is None:
            M = memory_bank.shape[1]
            if memory_bank.dim() == 2:
                memory_bank = memory_bank.unsqueeze(0).expand(B, -1, -1)
            k = self.k_proj(memory_bank).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(memory_bank).view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k, v = past_kv

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask[:, :, -T:, :]

        probs = safe_softmax(scores)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), (k.detach(), v.detach())


class AgentAttention(nn.Module):
    """
    Agent Attention: Multi-agent communication routing enabling 8 internal
    agent roles (Planner, Geometry, Constraint, etc.) to exchange state.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.agent_gate = nn.Linear(d_model, num_heads)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        agent_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kv_input = x if agent_states is None else agent_states
        B, T, C = x.shape
        T_kv = kv_input.shape[1]

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, T_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        gate_scale = torch.sigmoid(self.agent_gate(x)).transpose(1, 2).unsqueeze(-1)  # (B, H, T, 1)
        scores = scores * gate_scale
        probs = safe_softmax(scores)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

    def forward_cached(
        self,
        x: torch.Tensor,
        agent_states: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Incremental forward for one new token (B, 1, C); see SelfAttention."""
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if agent_states is None:
            agent_states = x
        if past_kv is None:
            k = self.k_proj(agent_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(agent_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k, v = past_kv
            k = torch.cat(
                [
                    k,
                    self.k_proj(agent_states)
                    .view(B, T, self.num_heads, self.head_dim)
                    .transpose(1, 2),
                ],
                dim=2,
            )
            v = torch.cat(
                [
                    v,
                    self.v_proj(agent_states)
                    .view(B, T, self.num_heads, self.head_dim)
                    .transpose(1, 2),
                ],
                dim=2,
            )

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        gate_scale = torch.sigmoid(self.agent_gate(x))
        scores = scores * gate_scale.transpose(1, 2).unsqueeze(-1)

        if attn_mask is not None:
            scores = scores + attn_mask[:, :, -T:, :]

        probs = safe_softmax(scores)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), (k.detach(), v.detach())


class UncertaintyAttention(nn.Module):
    """
    Uncertainty Attention: Computes entropy-weighted self-attention and
    outputs both representation and token-wise confidence logits.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.confidence_head = nn.Linear(d_model, 1)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (output_features: (B, T, C), confidence_logits: (B, T, 1))
        """
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask

        probs = safe_softmax(scores)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, C)
        out_features = self.out_proj(out)
        confidence_logits = self.confidence_head(out_features)
        return out_features, confidence_logits

    def forward_cached(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Incremental forward for one new token (B, 1, C); returns
        (out, confidence_logits, new_kv)."""
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if past_kv is None:
            k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k, v = past_kv
            k = torch.cat(
                [k, self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)], dim=2
            )
            v = torch.cat(
                [v, self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)], dim=2
            )

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask[:, :, -T:, :]

        probs = safe_softmax(scores)
        probs = self.dropout(probs)

        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(B, T, C)
        out_features = self.out_proj(out)
        confidence_logits = self.confidence_head(out_features)
        return out_features, confidence_logits, (k.detach(), v.detach())


class MultiHeadAttentionMixture(nn.Module):
    """
    Adaptive Mixture of Attention Heads: Combines all 6 specialized attention heads
    (Self, Geometry, Constraint, Memory, Agent, Uncertainty) via learned gating.
    """

    def __init__(
        self,
        d_model: int,
        self_heads: int = 4,
        geometry_heads: int = 4,
        constraint_heads: int = 2,
        memory_heads: int = 2,
        agent_heads: int = 2,
        uncertainty_heads: int = 2,
        dropout: float = 0.1,
        self_attn_backend: str = "math",
        num_kv_heads: int | None = None,
        kv_lora_rank: int = 64,
        qk_rope_head_dim: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.self_attn_backend = self_attn_backend

        self.self_attn = (
            cast(
                SelfAttention,
                build_self_attention(
                    self_attn_backend,
                    d_model,
                    self_heads,
                    dropout,
                    num_kv_heads=num_kv_heads,
                    kv_lora_rank=kv_lora_rank,
                    qk_rope_head_dim=qk_rope_head_dim,
                ),
            )
            if self_heads > 0
            else None
        )
        self.geometry_attn = (
            GeometryAttention(d_model, geometry_heads, dropout) if geometry_heads > 0 else None
        )
        self.constraint_attn = (
            ConstraintAttention(d_model, constraint_heads, dropout)
            if constraint_heads > 0
            else None
        )
        self.memory_attn = (
            MemoryAttention(d_model, memory_heads, dropout) if memory_heads > 0 else None
        )
        self.agent_attn = AgentAttention(d_model, agent_heads, dropout) if agent_heads > 0 else None
        self.uncertainty_attn = (
            UncertaintyAttention(d_model, uncertainty_heads, dropout)
            if uncertainty_heads > 0
            else None
        )

        # Gating router to compute adaptive weights over active heads
        num_active = sum(
            [
                self.self_attn is not None,
                self.geometry_attn is not None,
                self.constraint_attn is not None,
                self.memory_attn is not None,
                self.agent_attn is not None,
                self.uncertainty_attn is not None,
            ]
        )
        self.num_active = num_active
        self.gate = nn.Linear(d_model, num_active)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        memory_bank: torch.Tensor | None = None,
        agent_states: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
        constraint_mask: torch.Tensor | None = None,
        head_weights: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        x: (B, T, C)
        head_weights: (B, T, num_active), optional — adaptive attention head
            gating.  Multiplied into the softmax gate weights after the
            learned ``self.gate`` so head activation can be modulated
            (or forced to zero) on a per-token basis.
        cross_attn_mask: (B, 1, T, S), optional — block-diagonal mask for
            geometry cross-attention into ``encoder_hidden_states`` (used by
            sequence packing to block cross-sample attention).
        Returns (combined_output: (B, T, C), confidence_logits: Optional[(B, T, 1)])
        """
        head_outputs = []
        confidence_logits = None

        if self.self_attn is not None:
            head_outputs.append(self.self_attn(x, attn_mask=causal_mask))

        if self.geometry_attn is not None:
            if encoder_hidden_states is not None:
                head_outputs.append(
                    self.geometry_attn(
                        x, key_value=encoder_hidden_states, attn_mask=cross_attn_mask
                    )
                )
            else:
                head_outputs.append(self.geometry_attn(x, key_value=x, attn_mask=causal_mask))

        if self.constraint_attn is not None:
            head_outputs.append(self.constraint_attn(x, constraint_mask=constraint_mask))

        if self.memory_attn is not None:
            mem = memory_bank if memory_bank is not None else x
            head_outputs.append(self.memory_attn(x, memory_bank=mem))

        if self.agent_attn is not None:
            head_outputs.append(self.agent_attn(x, agent_states=agent_states))

        if self.uncertainty_attn is not None:
            unc_out, conf_logits = self.uncertainty_attn(x, attn_mask=causal_mask)
            head_outputs.append(unc_out)
            confidence_logits = conf_logits

        # Gating
        weights = F.softmax(self.gate(x), dim=-1)  # (B, T, num_active)

        # Adaptive attention head modulation (self-designing extension)
        if head_weights is not None:
            weights = weights * head_weights

        combined = torch.zeros_like(x)
        for idx, h_out in enumerate(head_outputs):
            w = weights[..., idx : idx + 1]  # (B, T, 1)
            combined = combined + (h_out * w)

        return self.out_proj(combined), confidence_logits

    def forward_cached(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        memory_bank: torch.Tensor | None = None,
        agent_states: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
        constraint_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
        head_weights: torch.Tensor | None = None,
        past_kv: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """
        Incremental forward for a single new token (B, 1, C).

        ``past_kv``: dict keyed by head name ("self", "geometry",
        "constraint", "memory", "agent", "uncertainty") holding that head's
        cached (k, v). Returns ``(combined, confidence_logits, new_kv)``;
        the new cache is exactly equivalent to running the full-sequence
        forward with the same history.
        """
        kv: dict[str, tuple[torch.Tensor, torch.Tensor]] = dict(past_kv or {})
        head_outputs = []
        confidence_logits = None

        if self.self_attn is not None:
            out, kv["self"] = self.self_attn.forward_cached(
                x,
                attn_mask=causal_mask,
                past_kv=kv.get("self"),
                position_offset=position_offset,
            )
            head_outputs.append(out)

        if self.geometry_attn is not None:
            kvs = encoder_hidden_states if encoder_hidden_states is not None else x
            out, kv["geometry"] = self.geometry_attn.forward_cached(
                x,
                key_value=kvs,
                attn_mask=cross_attn_mask,
                past_kv=kv.get("geometry"),
            )
            head_outputs.append(out)

        if self.constraint_attn is not None:
            out, kv["constraint"] = self.constraint_attn.forward_cached(
                x,
                constraint_mask=constraint_mask,
                past_kv=kv.get("constraint"),
            )
            head_outputs.append(out)

        if self.memory_attn is not None:
            mem = memory_bank if memory_bank is not None else x
            out, kv["memory"] = self.memory_attn.forward_cached(
                x,
                memory_bank=mem,
                past_kv=kv.get("memory"),
            )
            head_outputs.append(out)

        if self.agent_attn is not None:
            out, kv["agent"] = self.agent_attn.forward_cached(
                x,
                agent_states=agent_states,
                attn_mask=causal_mask,
                past_kv=kv.get("agent"),
            )
            head_outputs.append(out)

        if self.uncertainty_attn is not None:
            unc_out, conf_logits, kv["uncertainty"] = self.uncertainty_attn.forward_cached(
                x,
                attn_mask=causal_mask,
                past_kv=kv.get("uncertainty"),
            )
            head_outputs.append(unc_out)
            confidence_logits = conf_logits

        weights = F.softmax(self.gate(x), dim=-1)  # (B, 1, num_active)
        if head_weights is not None:
            weights = weights * head_weights

        combined = torch.zeros_like(x)
        for idx, h_out in enumerate(head_outputs):
            w = weights[..., idx : idx + 1]
            combined = combined + (h_out * w)

        return self.out_proj(combined), confidence_logits, kv
