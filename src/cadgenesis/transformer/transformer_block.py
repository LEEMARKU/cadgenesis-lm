"""
cadgenesis.transformer.transformer_block
==================================
Adaptive CAD Transformer Block for CADGenesis-LM v2.0:
- RMSNorm for pre-normalization
- SwiGLU Gated Feed-Forward Network
- CADTransformerBlock integrating MultiHeadAttentionMixture, RMSNorm,
  SwiGLU, and residual connections
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadgenesis.transformer.attention import MultiHeadAttentionMixture
from cadgenesis.transformer.interaction import FeatureInteractionLayer

if TYPE_CHECKING:  # pragma: no cover
    from cadgenesis.transformer.moe import SparseMoEFFN
    from cadgenesis.transformer.specialized_moe import SpecializedMoEFFN


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU Gated Linear Unit Feed-Forward Network."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.w2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.w3 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Swish(W1 x) * W3 x -> W2
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CADTransformerBlock(nn.Module):
    """
    Adaptive CAD Transformer Block integrating MultiHeadAttentionMixture,
    RMSNorm, SwiGLU (or Sparse MoE) FeedForward, and optional gradient
    checkpointing.

    Self-Designing extensions (all optional, None by default → identical to
    the original dense block):

    * ``use_moe=True`` swaps the SwiGLU FFN for a growable ``SparseMoEFFN``.
    * ``layer_gate`` (B, T, 1) scales the block's *delta* contribution per
      token → dynamic layer routing (``gate=0`` exactly reproduces the input).
    * ``head_weights`` (B, T, num_active_heads) multiplies the attention
      mixture's learned gate → adaptive attention heads.
    """

    def __init__(
        self,
        d_model: int = 1024,
        self_heads: int = 4,
        geometry_heads: int = 4,
        constraint_heads: int = 2,
        memory_heads: int = 2,
        agent_heads: int = 2,
        uncertainty_heads: int = 2,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
        use_moe: bool = False,
        num_experts: int = 4,
        top_k_experts: int = 2,
        expert_dim: int | None = None,
        expert_router_jitter: float = 0.02,
        self_attn_backend: str = "math",
        use_feature_interaction: bool = False,
        interaction_heads: int = 2,
        num_kv_heads: int | None = None,
        kv_lora_rank: int = 64,
        qk_rope_head_dim: int = 64,
        moe_aux_free_balancing: bool = False,
        moe_balance_speed: float = 0.001,
        moe_z_loss_weight: float = 1e-3,
        moe_capacity_factor: float | None = None,
        moe_drop_tokens: bool = False,
        num_shared_experts: int = 0,
        shared_expert_dim: int | None = None,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.use_feature_interaction = use_feature_interaction
        self.norm1 = RMSNorm(d_model)
        self.attn_mixture = MultiHeadAttentionMixture(
            d_model=d_model,
            self_heads=self_heads,
            geometry_heads=geometry_heads,
            constraint_heads=constraint_heads,
            memory_heads=memory_heads,
            agent_heads=agent_heads,
            uncertainty_heads=uncertainty_heads,
            dropout=dropout,
            self_attn_backend=self_attn_backend,
            num_kv_heads=num_kv_heads,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )
        self.feature_interaction: FeatureInteractionLayer | None
        if use_feature_interaction:
            self.norm_interact = RMSNorm(d_model)
            self.feature_interaction = FeatureInteractionLayer(
                d_model=d_model,
                num_heads=interaction_heads,
                dropout=dropout,
            )
        else:
            self.feature_interaction = None
        self.norm2 = RMSNorm(d_model)
        self.ffn: SwiGLU | SparseMoEFFN | SpecializedMoEFFN
        if use_moe:
            # Lazy import avoids a circular dependency at module load time.
            from cadgenesis.transformer.moe import SparseMoEFFN

            self.ffn = SparseMoEFFN(
                d_model=d_model,
                num_experts=num_experts,
                top_k=top_k_experts,
                expert_dim=expert_dim,
                dropout=dropout,
                router_jitter=expert_router_jitter,
                use_aux_free_balancing=moe_aux_free_balancing,
                balance_speed=moe_balance_speed,
                z_loss_weight=moe_z_loss_weight,
                capacity_factor=moe_capacity_factor,
                drop_tokens=moe_drop_tokens,
                num_shared_experts=num_shared_experts,
                shared_expert_dim=shared_expert_dim,
            )
        else:
            self.ffn = SwiGLU(d_model, dim_feedforward, dropout)

    @property
    def is_moe(self) -> bool:
        return self.use_moe

    def moe_layer(self) -> SparseMoEFFN | None:
        """Return the SparseMoEFFN if this block is MoE, else None."""
        if not self.use_moe:
            return None
        from cadgenesis.transformer.moe import SparseMoEFFN

        return cast(SparseMoEFFN, self.ffn)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        memory_bank: torch.Tensor | None = None,
        agent_states: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
        constraint_mask: torch.Tensor | None = None,
        layer_gate: torch.Tensor | None = None,
        head_weights: torch.Tensor | None = None,
        feature_type_ids: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        layer_gate : (B, T, 1), optional
            Per-token gate in [0, 1].  ``x_out = x0 + gate * (block(x0) - x0)``
            so ``gate == 0`` skips the whole block for that token.
        head_weights : (B, T, num_active_heads), optional
            Multiplier applied to the attention mixture's head gate.
        feature_type_ids : (B, T), optional
            Token family ids consumed by the optional feature-interaction
            sub-layer (only used when ``use_feature_interaction=True``).
        cross_attn_mask : (B, 1, T, S), optional
            Block-diagonal mask for geometry cross-attention into
            ``encoder_hidden_states`` (sequence packing).

        Returns
        -------
        (output_hidden_states: (B, T, C), confidence_logits: Optional[(B, T, 1)])
        """
        x0 = x
        # Attention sub-layer with residual connection
        norm_x = self.norm1(x)
        attn_out, conf_logits = self.attn_mixture(
            norm_x,
            encoder_hidden_states=encoder_hidden_states,
            memory_bank=memory_bank,
            agent_states=agent_states,
            causal_mask=causal_mask,
            constraint_mask=constraint_mask,
            head_weights=head_weights,
            cross_attn_mask=cross_attn_mask,
        )
        x = x0 + attn_out

        # Optional gated cross-feature interaction sub-layer.
        if self.feature_interaction is not None:
            x = self.feature_interaction(
                self.norm_interact(x),
                feature_type_ids=feature_type_ids,
                causal_mask=causal_mask,
            )

        # FFN sub-layer with residual connection
        x = x + self.ffn(self.norm2(x))

        # Dynamic layer routing gate (0 → skip this layer for that token)
        if layer_gate is not None:
            x = x0 + layer_gate * (x - x0)

        return x, conf_logits

    def forward_cached(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        memory_bank: torch.Tensor | None = None,
        agent_states: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
        constraint_mask: torch.Tensor | None = None,
        layer_gate: torch.Tensor | None = None,
        head_weights: torch.Tensor | None = None,
        feature_type_ids: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
        past_kv: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """
        Incremental forward for a single new token (B, 1, C) using the KV
        cache from previous steps. ``past_kv`` is the block's per-head cache
        dict; returns ``(x, confidence_logits, new_kv)``. Exactly equivalent
        to the full-sequence ``forward`` when the history is replayed.
        """
        x0 = x
        norm_x = self.norm1(x)
        attn_out, conf_logits, new_kv = self.attn_mixture.forward_cached(
            norm_x,
            encoder_hidden_states=encoder_hidden_states,
            memory_bank=memory_bank,
            agent_states=agent_states,
            causal_mask=causal_mask,
            constraint_mask=constraint_mask,
            head_weights=head_weights,
            cross_attn_mask=cross_attn_mask,
            past_kv=past_kv,
            position_offset=position_offset,
        )
        x = x0 + attn_out

        if self.feature_interaction is not None:
            x = self.feature_interaction(
                self.norm_interact(x),
                feature_type_ids=feature_type_ids,
                causal_mask=causal_mask,
            )

        x = x + self.ffn(self.norm2(x))

        if layer_gate is not None:
            x = x0 + layer_gate * (x - x0)

        return x, conf_logits, new_kv
