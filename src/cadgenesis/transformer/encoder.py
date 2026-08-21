"""cadgenesis.transformer.encoder
==============================
Reusable encoder stack built from :class:`CADTransformerBlock` blocks, with
optional layer-integrated memory refinement and self-designing routing hooks.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from cadgenesis.config import ModelConfig
from cadgenesis.transformer.transformer_block import CADTransformerBlock, RMSNorm


class EncoderStack(nn.Module):
    """A stack of CAD transformer encoder blocks with a final RMS norm.

    Mirrors the encoder behaviour of :class:`GeometryAwareTransformer`: each
    block optionally conditions on the layer-integrated memory bank and on
    self-designing ``layer_gate`` / ``head_weights`` masks.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.num_encoder_layers < 1:
            raise ValueError("num_encoder_layers must be >= 1")
        self.config = config
        self.d_model = config.d_model
        self.blocks = nn.ModuleList(
            [
                CADTransformerBlock(
                    d_model=config.d_model,
                    self_heads=config.self_attn_heads,
                    geometry_heads=config.geometry_attn_heads,
                    constraint_heads=config.constraint_attn_heads,
                    memory_heads=config.memory_attn_heads,
                    agent_heads=config.agent_attn_heads,
                    uncertainty_heads=config.uncertainty_attn_heads,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    use_moe=config.use_moe,
                    num_experts=config.num_experts,
                    top_k_experts=config.top_k_experts,
                    expert_dim=config.expert_dim,
                    expert_router_jitter=config.expert_router_jitter,
                    self_attn_backend=config.attention_backend,
                    use_feature_interaction=config.feature_interaction,
                    interaction_heads=config.interaction_heads,
                )
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.norm = RMSNorm(config.d_model)

    @property
    def num_layers(self) -> int:
        return len(self.blocks)

    def forward(
        self,
        x: torch.Tensor,
        memory_bank: torch.Tensor | None = None,
        layer_gate: Callable[[int, torch.Tensor, str], torch.Tensor | None] | None = None,
        head_weights: Callable[[int, torch.Tensor, str], torch.Tensor | None] | None = None,
        refine_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Encode ``x`` (B, S, d_model) through the stack.

        Args:
            x: Input embeddings.
            memory_bank: Shared memory bank passed to every block.
            layer_gate: Optional callable ``(block_idx, x, "encoder") -> mask``.
            head_weights: Optional callable ``(block_idx, x, "encoder") -> mask``.
            refine_fn: Optional callable ``(memory_bank, x) -> memory_bank``
                applied after each block (layer-integrated memory write-back).

        Returns:
            Encoder hidden states (B, S, d_model).
        """
        for i, block in enumerate(self.blocks):
            gate = layer_gate(i, x, "encoder") if layer_gate is not None else None
            heads = head_weights(i, x, "encoder") if head_weights is not None else None
            x, _ = block(
                x,
                memory_bank=memory_bank,
                layer_gate=gate,
                head_weights=heads,
            )
            if refine_fn is not None and memory_bank is not None:
                memory_bank = refine_fn(memory_bank, x)
        return self.norm(x)
