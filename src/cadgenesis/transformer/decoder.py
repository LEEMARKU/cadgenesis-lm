"""cadgenesis.transformer.decoder
==============================
Reusable decoder stack built from :class:`CADTransformerBlock` blocks, with
causal masking, cross-attention to encoder states, layer-integrated memory,
and an internal agent-bus hook.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from cadgenesis.config import ModelConfig
from cadgenesis.transformer.heads import ConfidenceHead
from cadgenesis.transformer.transformer_block import CADTransformerBlock, RMSNorm


class DecoderStack(nn.Module):
    """A stack of CAD transformer decoder blocks with a final RMS norm.

    Mirrors the decoder behaviour of :class:`GeometryAwareTransformer`:
    a causal attention mask is applied, blocks attend to encoder hidden states,
    and an internal multi-agent bus can be re-derived per block.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.num_decoder_layers < 1:
            raise ValueError("num_decoder_layers must be >= 1")
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
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.norm = RMSNorm(config.d_model)

    @property
    def num_layers(self) -> int:
        return len(self.blocks)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        memory_bank: torch.Tensor | None = None,
        agent_states: torch.Tensor | None = None,
        agent_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor | None] | None = None,
        constraint_mask: torch.Tensor | None = None,
        layer_gate: Callable[[int, torch.Tensor, str], torch.Tensor | None] | None = None,
        head_weights: Callable[[int, torch.Tensor, str], torch.Tensor | None] | None = None,
        feature_type_ids: torch.Tensor | None = None,
        refine_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode ``x`` (B, T, d_model) through the stack.

        Args:
            x: Target embeddings.
            encoder_hidden_states: Encoder states for cross-attention.
            memory_bank: Shared memory bank passed to every block.
            agent_states: Precomputed agent bus; when None and ``agent_fn`` is
                given, the bus is derived per block via ``agent_fn(x, memory_bank)``.
            agent_fn: Callable producing agent states from ``(x, memory_bank)``.
            constraint_mask: Optional constraint attention mask.
            layer_gate / head_weights: Self-designing callables.
            feature_type_ids: Token family ids for feature interaction.
            refine_fn: Optional memory write-back after each block.

        Returns:
            ``(hidden_states, confidence_logits_or_None)``.
        """
        _, T, _ = x.shape
        causal_mask = (
            torch.triu(
                torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        conf_logits_last = None
        for i, block in enumerate(self.blocks):
            if agent_states is None and agent_fn is not None:
                block_agent = agent_fn(x, memory_bank)
            else:
                block_agent = agent_states
            gate = layer_gate(i, x, "decoder") if layer_gate is not None else None
            heads = head_weights(i, x, "decoder") if head_weights is not None else None
            x, conf_logits = block(
                x,
                encoder_hidden_states=encoder_hidden_states,
                memory_bank=memory_bank,
                agent_states=block_agent,
                causal_mask=causal_mask,
                constraint_mask=constraint_mask,
                layer_gate=gate,
                head_weights=heads,
                feature_type_ids=feature_type_ids,
            )
            if conf_logits is not None:
                conf_logits_last = conf_logits
            if refine_fn is not None and memory_bank is not None:
                memory_bank = refine_fn(memory_bank, x)

        out = self.norm(x)
        if conf_logits_last is None:
            # No uncertainty attention head in the block config — still expose
            # a confidence logit per token so the decoder API is uniform.
            if not hasattr(self, "_conf_head"):
                self._conf_head = ConfidenceHead(self.d_model)
            conf_logits_last = self._conf_head(out)
        return out, conf_logits_last
