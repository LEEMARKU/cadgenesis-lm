"""
cadgenesis.transformer.geometry_transformer
======================================
GeometryAwareTransformer: Production Foundation Model for Generative Parametric CAD.

Integrated Subsystems:
- Autonomous CAD Tokenizer Space (10 families)
- Adaptive CAD Transformer Blocks (6 Specialized Attention Heads)
- Layer-Integrated Memory Pools (8 Pools)
- Internal Multi-Agent System (8 Roles)
- Neuro-Symbolic Reasoning Engine
- RLAIF & Confidence Heads
"""

from __future__ import annotations

import math
from typing import Protocol, cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _checkpoint

from cadgenesis.agents.multi_agent_system import MultiAgentSystem
from cadgenesis.alignment.constitutional_ai import RLAIFRewardModel
from cadgenesis.config import CADConfig, ModelConfig
from cadgenesis.memory.memory_pools import LayerIntegratedMemorySystem
from cadgenesis.reasoning.neuro_symbolic import NeuroSymbolicReasoningEngine
from cadgenesis.transformer.attention import repair_fully_masked_rows
from cadgenesis.transformer.positional import (
    GeometryPositionalEncoding,
    RotaryEmbedding,
    SinusoidalPositionalEncoding,
)
from cadgenesis.transformer.ssm import GatedDeltaNet, add_ssm_blocks
from cadgenesis.transformer.transformer_block import CADTransformerBlock, RMSNorm


class AdaptiveController(Protocol):
    """Duck-typed self-designing controller contract consumed by the backbone.

    Implemented by ``SelfDesigningTransformer`` (and any test double exposing
    the same two methods).
    """

    def layer_gate(
        self,
        layer_idx: int,
        x: torch.Tensor,
        layer_type: str = "encoder",
    ) -> torch.Tensor | None: ...

    def head_weights(
        self,
        layer_idx: int,
        x: torch.Tensor,
        layer_type: str = "encoder",
    ) -> torch.Tensor | None: ...


class CrossAttentionSource(Protocol):
    """Per-head cross-attention projection surface used by cache preparation.

    Structurally satisfied by ``GeometryAttention`` and ``MemoryAttention``.
    """

    num_heads: int
    head_dim: int

    def k_proj(self, x: torch.Tensor) -> torch.Tensor: ...

    def v_proj(self, x: torch.Tensor) -> torch.Tensor: ...


class GeometryAwareTransformer(nn.Module):
    """
    Complete GeometryAwareTransformer Core Model for CADGenesis-LM v2.0.
    """

    def __init__(self, config: CADConfig):
        super().__init__()
        self.config = config
        m_cfg: ModelConfig = config.model
        t_cfg = config.tokenizer

        self.d_model = m_cfg.d_model
        self.lang_vocab_size = t_cfg.lang_vocab_size
        self.cad_vocab_size = (
            t_cfg.geometry_token_slots
            + t_cfg.feature_token_slots
            + t_cfg.constraint_token_slots
            + t_cfg.material_token_slots
            + t_cfg.assembly_token_slots
            + t_cfg.manufacturing_token_slots
            + t_cfg.simulation_token_slots
            + t_cfg.numeric_token_slots
            + t_cfg.special_token_slots
        )

        # --- Multi-Modal Embedding Layers ---
        self.lang_embed = nn.Embedding(self.lang_vocab_size, self.d_model, padding_idx=0)
        self.cad_embed = nn.Embedding(self.cad_vocab_size, self.d_model, padding_idx=0)
        self.type_embed = nn.Embedding(10, self.d_model)  # 10 token family types (0..9)

        self.pos_enc = SinusoidalPositionalEncoding(self.d_model, max_len=m_cfg.max_seq_len)

        # Optional geometry positional encoding (X/Y/Z coordinate aware).
        self.geometry_pos_enc = (
            GeometryPositionalEncoding(d_model=self.d_model)
            if m_cfg.geometry_pos_encoding
            else None
        )

        # --- Integrated Subsystems (all optional; lean default = OFF) ---
        # Each is only instantiated when enabled, so the lean architecture is a
        # standard transformer with no wasted parameters or compute.
        self.memory_system = (
            LayerIntegratedMemorySystem(d_model=self.d_model) if m_cfg.use_memory_system else None
        )
        self.multi_agent_system = (
            MultiAgentSystem(d_model=self.d_model) if m_cfg.use_multi_agent_system else None
        )
        self.neuro_symbolic_engine = (
            NeuroSymbolicReasoningEngine(d_model=self.d_model)
            if m_cfg.use_neuro_symbolic_reasoning
            else None
        )
        self.reward_model = (
            RLAIFRewardModel(d_model=self.d_model) if m_cfg.use_rlaf_reward_model else None
        )

        # Specialized heads are only meaningful when their subsystem exists;
        # otherwise the block would compute them against None states.
        agent_heads = m_cfg.agent_attn_heads if m_cfg.use_multi_agent_system else 0
        memory_heads = m_cfg.memory_attn_heads if m_cfg.use_memory_system else 0

        # Long-context RoPE scaling.  Reset the module-level defaults to the
        # canonical values first so earlier models' scaling settings cannot
        # leak into this one (instances capture these at construction time,
        # so previously built models are unaffected), then apply this model's
        # configuration before any attention module is constructed.  The
        # precomputed RoPE table always covers at least ``max_seq_len``
        # (v6.1 §4.7), so the configured context is usable out of the box.
        RotaryEmbedding.configure_defaults(
            max_position_embeddings=4096,
            base=10000.0,
            scaling_factor=1.0,
            scaling_type="none",
        )
        if (
            m_cfg.rope_scaling_type != "none"
            or m_cfg.rope_scaling_factor != 1.0
            or m_cfg.max_position_embeddings != m_cfg.max_seq_len
        ):
            RotaryEmbedding.configure_defaults(
                max_position_embeddings=max(
                    m_cfg.max_position_embeddings, m_cfg.max_seq_len
                ),
                base=m_cfg.rope_theta,
                scaling_factor=m_cfg.rope_scaling_factor,
                scaling_type=m_cfg.rope_scaling_type,
            )

        # Hybrid SSM interleave plans (Gated DeltaNet after every Nth block).
        no_encoder_ssm: list[GatedDeltaNet | None] = [None] * m_cfg.num_encoder_layers
        no_decoder_ssm: list[GatedDeltaNet | None] = [None] * m_cfg.num_decoder_layers
        self.encoder_ssm = nn.ModuleList(
            cast(
                list[nn.Module],
                add_ssm_blocks(
                    m_cfg.num_encoder_layers,
                    self.d_model,
                    m_cfg.ssm_every_n_blocks,
                    heads=m_cfg.ssm_heads,
                    dropout=m_cfg.dropout,
                )
                if m_cfg.use_ssm
                else no_encoder_ssm,
            )
        )
        self.decoder_ssm = nn.ModuleList(
            cast(
                list[nn.Module],
                add_ssm_blocks(
                    m_cfg.num_decoder_layers,
                    self.d_model,
                    m_cfg.ssm_every_n_blocks,
                    heads=m_cfg.ssm_heads,
                    dropout=m_cfg.dropout,
                )
                if m_cfg.use_ssm
                else no_decoder_ssm,
            )
        )

        # --- Encoder Layers ---
        self.encoder_blocks = nn.ModuleList(
            [
                CADTransformerBlock(
                    d_model=self.d_model,
                    self_heads=m_cfg.self_attn_heads,
                    geometry_heads=m_cfg.geometry_attn_heads,
                    constraint_heads=m_cfg.constraint_attn_heads,
                    memory_heads=memory_heads,
                    agent_heads=agent_heads,
                    uncertainty_heads=m_cfg.uncertainty_attn_heads,
                    dim_feedforward=m_cfg.dim_feedforward,
                    dropout=m_cfg.dropout,
                    use_moe=m_cfg.use_moe,
                    num_experts=m_cfg.num_experts,
                    top_k_experts=m_cfg.top_k_experts,
                    expert_dim=m_cfg.expert_dim,
                    expert_router_jitter=m_cfg.expert_router_jitter,
                    self_attn_backend=m_cfg.attention_backend,
                    use_feature_interaction=m_cfg.feature_interaction,
                    interaction_heads=m_cfg.interaction_heads,
                    num_shared_experts=m_cfg.num_shared_experts,
                    shared_expert_dim=m_cfg.shared_expert_dim,
                    **self._block_kwargs(),
                )
                for _ in range(m_cfg.num_encoder_layers)
            ]
        )
        self.encoder_norm = RMSNorm(self.d_model)

        # --- Decoder Layers ---
        self.decoder_blocks = nn.ModuleList(
            [
                CADTransformerBlock(
                    d_model=self.d_model,
                    self_heads=m_cfg.self_attn_heads,
                    geometry_heads=m_cfg.geometry_attn_heads,
                    constraint_heads=m_cfg.constraint_attn_heads,
                    memory_heads=memory_heads,
                    agent_heads=agent_heads,
                    uncertainty_heads=m_cfg.uncertainty_attn_heads,
                    dim_feedforward=m_cfg.dim_feedforward,
                    dropout=m_cfg.dropout,
                    use_moe=m_cfg.use_moe,
                    num_experts=m_cfg.num_experts,
                    top_k_experts=m_cfg.top_k_experts,
                    expert_dim=m_cfg.expert_dim,
                    expert_router_jitter=m_cfg.expert_router_jitter,
                    self_attn_backend=m_cfg.attention_backend,
                    use_feature_interaction=m_cfg.feature_interaction,
                    interaction_heads=m_cfg.interaction_heads,
                    num_shared_experts=m_cfg.num_shared_experts,
                    shared_expert_dim=m_cfg.shared_expert_dim,
                    **self._block_kwargs(),
                )
                for _ in range(m_cfg.num_decoder_layers)
            ]
        )
        self.decoder_norm = RMSNorm(self.d_model)

        # --- Output Projection Heads ---
        self.out_proj = nn.Linear(self.d_model, self.cad_vocab_size, bias=False)
        # Standard GPT-style weight tying: the LM head shares the CAD token
        # embedding (also required by the MTP head's weight-tied logits).
        self.out_proj.weight = self.cad_embed.weight
        self.confidence_head = nn.Linear(self.d_model, 1)
        self.use_confidence_head = m_cfg.use_confidence_head

        # --- Multi-Token Prediction auxiliary head (DeepSeek-V3 style) ---
        self.mtp_head = None
        if getattr(m_cfg, "mtp_depth", 0) and m_cfg.mtp_depth > 0:
            from cadgenesis.transformer.mtp import MultiTokenPredictionHead

            self.mtp_head = MultiTokenPredictionHead(
                d_model=self.d_model,
                vocab_size=self.cad_vocab_size,
                mtp_depth=int(m_cfg.mtp_depth),
                dropout=m_cfg.dropout,
            )

        self._init_weights()

        # BitNet b1.58: ternary weights + int8 activations (straight-through
        # training).  Applied last so weight tying and initialisation run on
        # full-precision tensors first.
        if m_cfg.use_bitnet:
            from cadgenesis.quantization.bitnet import apply_bitnet

            apply_bitnet(self)

    def _block_kwargs(self) -> dict:
        """Common kwargs for every transformer block (P0/P1 modernization)."""
        m_cfg = self.config.model
        qk_rope = getattr(m_cfg, "qk_rope_head_dim", 64)
        if m_cfg.attention_backend == "mla":
            # MLA requires qk_rope_head_dim < head_dim (and an even rope dim);
            # auto-clamp to the largest even value below head_dim for small
            # models, regardless of when the config was mutated (v6.1 §4.6).
            head_dim = m_cfg.d_model // m_cfg.self_attn_heads
            max_even = head_dim - 1 - ((head_dim - 1) % 2)
            qk_rope = min(qk_rope, max_even)
        return {
            "num_kv_heads": getattr(m_cfg, "num_kv_heads", None),
            "kv_lora_rank": getattr(m_cfg, "kv_lora_rank", 64),
            "qk_rope_head_dim": qk_rope,
            "moe_aux_free_balancing": getattr(m_cfg, "moe_aux_free_balancing", False),
            "moe_balance_speed": getattr(m_cfg, "moe_balance_speed", 0.001),
            "moe_z_loss_weight": getattr(m_cfg, "moe_z_loss_weight", 1e-3),
            "moe_capacity_factor": getattr(m_cfg, "moe_capacity_factor", None),
            "moe_drop_tokens": getattr(m_cfg, "moe_drop_tokens", False),
        }

    def _maybe_checkpoint(self, fn, *args, **kwargs):
        """
        Wrap ``fn(*args, **kwargs)`` in activation checkpointing when enabled
        (``TrainingConfig.gradient_checkpointing``) *and* the model is in
        training mode.  In evaluation / inference the call is passed through
        untouched.  Non-reentrant checkpointing keeps the numerics identical
        to a plain forward while trading recompute for memory.
        """
        if self.config.training.gradient_checkpointing and self.training:
            return _checkpoint(fn, *args, use_reentrant=False, **kwargs)
        return fn(*args, **kwargs)

    def aux_loss(self) -> torch.Tensor:
        """
        Sum of the MoE auxiliary losses (load balancing / router z-loss) over
        every encoder and decoder block.  Returns a zero tensor when no MoE
        block is present; the trainer mixes it into the total loss scaled by
        ``TrainingConfig.moe_aux_scale``.
        """
        device = next(self.parameters()).device
        total = torch.zeros((), device=device)
        for block in (*self.encoder_blocks, *self.decoder_blocks):
            moe = block.moe_layer() if hasattr(block, "moe_layer") else None
            if moe is not None:
                total = total + moe.get_aux_loss()
        return total

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    def encode(
        self,
        src_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        adaptive: AdaptiveController | None = None,
        geometry_coords: torch.Tensor | None = None,
        src_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        src_ids: (B, S)
        adaptive: optional object exposing
            ``layer_gate(block_idx, x, layer_type)`` and
            ``head_weights(block_idx, x, layer_type)`` (duck-typed;
            None → standard path).
        geometry_coords: (B, S, 3) or (S, 3), optional X/Y/Z coordinates used
            only when ``model.geometry_pos_encoding`` is enabled.
        src_attn_mask: (B, 1, S, S), optional block-diagonal self-attention
            mask (sequence packing blocks cross-sample attention).
        Returns: encoder_hidden_states (B, S, C)
        """
        B, _S = src_ids.shape
        src = self.pos_enc(self.lang_embed(src_ids) * math.sqrt(self.d_model))
        if self.geometry_pos_enc is not None:
            src = self.geometry_pos_enc(src, geometry_coords)

        # Padding tokens must not be attended to: merge the (B, S) padding
        # mask into the additive self-attention mask (keys of pad tokens are
        # masked to -inf; pad rows are never scored by the loss anyway).
        # ``repair_fully_masked_rows`` re-opens a self slot for any query row
        # that would otherwise see an all-(-inf) score row (e.g. fully-padding
        # tail blocks of a packed row) — all-(-inf) rows make softmax emit NaN.
        if src_key_padding_mask is not None:
            pad = src_key_padding_mask.to(src.device)
            if pad.dtype != torch.bool:
                pad = pad.bool()
            pad_attn = torch.zeros(B, _S, _S, device=src.device)
            pad_attn = pad_attn.masked_fill(pad[:, None, :], float("-inf"))
            pad_attn = pad_attn.unsqueeze(1)  # (B, 1, S, S)
            src_attn_mask = (
                pad_attn if src_attn_mask is None else src_attn_mask + pad_attn
            )
            src_attn_mask = repair_fully_masked_rows(src_attn_mask)

        # Layer-integrated memory: every encoder layer reads + refines memory.
        memory_bank = (
            self.memory_system.get_combined_memory_bank(batch_size=B)
            if self.memory_system is not None
            else None
        )

        x = src
        for i, block in enumerate(self.encoder_blocks):
            gate = adaptive.layer_gate(i, x, "encoder") if adaptive is not None else None
            heads = adaptive.head_weights(i, x, "encoder") if adaptive is not None else None
            x, _ = self._maybe_checkpoint(
                block,
                x,
                memory_bank=memory_bank,
                layer_gate=gate,
                head_weights=heads,
                causal_mask=src_attn_mask,
            )
            if self.memory_system is not None and memory_bank is not None:
                memory_bank = self.memory_system.refine(memory_bank, x)
            # Hybrid SSM interleave: a Gated DeltaNet layer after the block.
            ssm = self.encoder_ssm[i]
            if ssm is not None:
                x = self._maybe_checkpoint(ssm, x)

        return self.encoder_norm(x)

    def decode(
        self,
        tgt_in_ids: torch.Tensor,
        tgt_type_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        memory_bank: torch.Tensor | None = None,
        agent_states: torch.Tensor | None = None,
        constraint_mask: torch.Tensor | None = None,
        adaptive: AdaptiveController | None = None,
        geometry_coords: torch.Tensor | None = None,
        tgt_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        tgt_in_ids: (B, T)
        tgt_type_ids: (B, T)
        encoder_hidden_states: (B, S, C)
        adaptive: optional object exposing ``layer_gate``/``head_weights``
            with ``(block_idx, x, layer_type)`` signature.
        geometry_coords: (B, T, 3) or (T, 3), optional X/Y/Z coordinates used
            only when ``model.geometry_pos_encoding`` is enabled.
        tgt_attn_mask: (B, 1, T, T), optional — block-diagonal causal mask
            (sequence packing); falls back to a plain triangular causal mask.
        cross_attn_mask: (B, 1, T, S), optional — block-diagonal mask for
            geometry cross-attention (sequence packing).

        Returns: (logits: (B, T, cad_vocab_size), confidence: (B, T, 1))
        """
        B, T = tgt_in_ids.shape
        tgt = self.cad_embed(tgt_in_ids) + self.type_embed(tgt_type_ids)
        tgt = self.pos_enc(tgt * math.sqrt(self.d_model))
        if self.geometry_pos_enc is not None:
            tgt = self.geometry_pos_enc(tgt, geometry_coords)

        # Retrieve memory bank if not provided
        if memory_bank is None and self.memory_system is not None:
            memory_bank = self.memory_system.get_combined_memory_bank(batch_size=B)

        # Causal mask for autoregressive target decoding
        if tgt_attn_mask is None:
            tgt_attn_mask = (
                torch.triu(torch.full((T, T), float("-inf"), device=tgt.device), diagonal=1)
                .unsqueeze(0)
                .unsqueeze(0)
            )  # (1, 1, T, T)

        # Padding tokens must not be attended to in either self-attention or
        # cross-attention (encoder padding also masks the geometry heads).
        # As in ``encode``, dead query rows (all-(-inf) scores) are re-opened
        # so softmax never produces NaN.
        if tgt_key_padding_mask is not None:
            pad = tgt_key_padding_mask.to(tgt.device)
            if pad.dtype != torch.bool:
                pad = pad.bool()
            pad_attn = torch.zeros(B, T, T, device=tgt.device)
            pad_attn = pad_attn.masked_fill(pad[:, None, :], float("-inf"))
            tgt_attn_mask = tgt_attn_mask + pad_attn.unsqueeze(1)
            tgt_attn_mask = repair_fully_masked_rows(tgt_attn_mask)
        if src_key_padding_mask is not None and encoder_hidden_states is not None:
            pad = src_key_padding_mask.to(encoder_hidden_states.device)
            if pad.dtype != torch.bool:
                pad = pad.bool()
            _S = encoder_hidden_states.shape[1]
            pad_cross = torch.zeros(B, T, _S, device=encoder_hidden_states.device)
            pad_cross = pad_cross.masked_fill(pad[:, None, :], float("-inf"))
            pad_cross = pad_cross.unsqueeze(1)  # (B, 1, T, S)
            cross_attn_mask = (
                pad_cross if cross_attn_mask is None else cross_attn_mask + pad_cross
            )
            cross_attn_mask = repair_fully_masked_rows(cross_attn_mask)

        x = tgt
        conf_logits_last = None
        for i, block in enumerate(self.decoder_blocks):
            # Agents re-derive their communication bus from the latest state,
            # conditioned on the shared memory bank (layer-integrated).
            block_agent_states = (
                agent_states
                if agent_states is not None
                else (
                    self.multi_agent_system(x, memory_bank=memory_bank)
                    if self.multi_agent_system is not None
                    else None
                )
            )
            gate = adaptive.layer_gate(i, x, "decoder") if adaptive is not None else None
            heads = adaptive.head_weights(i, x, "decoder") if adaptive is not None else None
            x, conf_logits = self._maybe_checkpoint(
                block,
                x,
                encoder_hidden_states=encoder_hidden_states,
                memory_bank=memory_bank,
                agent_states=block_agent_states,
                causal_mask=tgt_attn_mask,
                constraint_mask=constraint_mask,
                layer_gate=gate,
                head_weights=heads,
                feature_type_ids=tgt_type_ids,
                cross_attn_mask=cross_attn_mask,
            )
            if conf_logits is not None:
                conf_logits_last = conf_logits
            # Layer-integrated memory refinement: each layer updates working memory.
            if self.memory_system is not None and memory_bank is not None:
                memory_bank = self.memory_system.refine(memory_bank, x)
            # Hybrid SSM interleave after the decoder block.
            ssm = self.decoder_ssm[i]
            if ssm is not None:
                x = self._maybe_checkpoint(ssm, x)

        # Neuro-symbolic constraint refinement
        if self.neuro_symbolic_engine is not None:
            _symbolic_scores, x = self.neuro_symbolic_engine.evaluate_constraints(x)

        x = self.decoder_norm(x)
        self.decode_hidden_states = x
        self.last_reward = self.reward_model(x) if self.reward_model is not None else None
        logits = self.out_proj(x)

        if conf_logits_last is None and self.use_confidence_head:
            conf_logits_last = self.confidence_head(x)

        return logits, conf_logits_last

    @staticmethod
    def _project_cross_kv(
        attn: CrossAttentionSource,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project an invariant cross-attention source (encoder states or a
        frozen memory bank) into per-head (k, v) tensors (B, H, M, D)."""
        B, M, _C = source.shape
        k = attn.k_proj(source).view(B, M, attn.num_heads, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(source).view(B, M, attn.num_heads, attn.head_dim).transpose(1, 2)
        return k.detach(), v.detach()

    @torch.no_grad()
    def prepare_decoder_cache(
        self,
        src_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        adaptive: AdaptiveController | None = None,
    ) -> dict:
        """
        Precompute everything invariant across autoregressive steps:

        * ``encoder_hidden_states`` (B, S, C),
        * per-decoder-layer geometry cross-attention K/V (projected encoder),
        * per-decoder-layer memory-attention K/V (projected memory bank),
        * the frozen ``memory_bank``, and
        * an empty per-block KV slot dict.

        Returns a cache dict consumable by :meth:`decode_step`. The memory
        bank is deliberately frozen for the whole generation (no per-step
        refinement) so every step stays exactly equivalent to the uncached
        full-sequence forward; refinement is a soft mechanism only.
        """
        encoder_hidden_states = self.encode(
            src_ids,
            src_key_padding_mask=src_key_padding_mask,
            adaptive=adaptive,
        )
        B, _S, _C = encoder_hidden_states.shape
        memory_bank = (
            self.memory_system.get_combined_memory_bank(batch_size=B)
            if self.memory_system is not None
            else None
        )

        geometry_kv: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        memory_kv: list[tuple[torch.Tensor, torch.Tensor] | None] = []
        decoder_blocks: list[CADTransformerBlock] = cast(
            list[CADTransformerBlock], list(self.decoder_blocks)
        )
        for block in decoder_blocks:
            mixture = block.attn_mixture
            if mixture.geometry_attn is not None:
                geometry_kv.append(
                    self._project_cross_kv(mixture.geometry_attn, encoder_hidden_states)
                )
            else:
                geometry_kv.append(None)
            if mixture.memory_attn is not None and memory_bank is not None:
                memory_kv.append(self._project_cross_kv(mixture.memory_attn, memory_bank))
            else:
                memory_kv.append(None)

        return {
            "encoder_hidden_states": encoder_hidden_states,
            "memory_bank": memory_bank,
            "geometry_kv": geometry_kv,
            "memory_kv": memory_kv,
            "blocks": [None] * len(decoder_blocks),
            "position_offset": 0,
            "ssm_states": [
                (
                    ssm._initial_state(B, encoder_hidden_states.device)
                    if isinstance(ssm, GatedDeltaNet)
                    else None
                )
                for ssm in self.decoder_ssm
            ],
        }

    @torch.no_grad()
    def decode_step(
        self,
        tgt_in_ids: torch.Tensor,
        tgt_type_ids: torch.Tensor,
        cache: dict,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        One autoregressive step for a single new token: ``tgt_in_ids`` /
        ``tgt_type_ids`` are (B, 1).

        ``cache`` is the dict produced by :meth:`prepare_decoder_cache`; it is
        mutated in place (per-block K/V grown, ``position_offset`` advanced).
        Returns ``(logits (B, 1, V), confidence (B, 1, 1))`` for the new token.

        When the decoder contains only per-token subsystems (self/geometry/
        constraint/agent/uncertainty attention, MLP agents, neuro-symbolic
        rules), this is *exactly* equivalent to the full-sequence
        :meth:`decode` (verified by tests).
        """
        _B, T = tgt_in_ids.shape
        position_offset = cache["position_offset"]
        tgt = self.cad_embed(tgt_in_ids) + self.type_embed(tgt_type_ids)
        tgt = self.pos_enc(tgt * math.sqrt(self.d_model), position_offset=position_offset)

        memory_bank = cache["memory_bank"]
        x = tgt
        conf_logits_last = None
        decoder_blocks: list[CADTransformerBlock] = cast(
            list[CADTransformerBlock], list(self.decoder_blocks)
        )
        for i, block in enumerate(decoder_blocks):
            block_agent_states = (
                self.multi_agent_system(x, memory_bank=memory_bank)
                if self.multi_agent_system is not None
                else None
            )
            past_kv = cache["blocks"][i]
            if past_kv is None:
                # First step: seed the precomputed cross-attention K/V so the
                # projection of the (frozen) encoder states / memory bank is
                # computed once per sequence instead of once per token.
                past_kv = {}
                geometry_kv = cache["geometry_kv"][i] if cache.get("geometry_kv") else None
                if geometry_kv is not None:
                    past_kv["geometry"] = geometry_kv
                memory_kv = cache["memory_kv"][i] if cache.get("memory_kv") else None
                if memory_kv is not None:
                    past_kv["memory"] = memory_kv
            x, conf_logits, new_kv = block.forward_cached(
                x,
                encoder_hidden_states=cache["encoder_hidden_states"],
                memory_bank=memory_bank,
                agent_states=block_agent_states,
                causal_mask=None,
                constraint_mask=None,
                feature_type_ids=tgt_type_ids,
                cross_attn_mask=None,
                past_kv=past_kv,
                position_offset=position_offset,
            )
            cache["blocks"][i] = new_kv
            if conf_logits is not None:
                conf_logits_last = conf_logits
            # Hybrid SSM recurrent step, updating the cached state.
            ssm = self.decoder_ssm[i]
            if isinstance(ssm, GatedDeltaNet):
                x, cache["ssm_states"][i] = ssm.forward_cached(x, cache["ssm_states"][i])

        if self.neuro_symbolic_engine is not None:
            _, x = self.neuro_symbolic_engine.evaluate_constraints(x)
        x = self.decoder_norm(x)
        self.decode_hidden_states = x
        self.last_reward = self.reward_model(x) if self.reward_model is not None else None
        logits = self.out_proj(x)

        if conf_logits_last is None and self.use_confidence_head:
            conf_logits_last = self.confidence_head(x)

        cache["position_offset"] = position_offset + T
        return logits, conf_logits_last

    def mtp_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor | None, dict]:
        """
        Multi-token prediction auxiliary loss (DeepSeek-V3 style MTP head).

        ``hidden``: final decoder hidden states (B, T, d_model);
        ``targets``: teacher-forced target ids (B, T).

        Returns ``(mtp_loss, breakdown)``; ``(None, {})`` when the MTP head is
        disabled (``config.model.mtp_depth == 0``).
        """
        if self.mtp_head is None:
            return None, {}
        from cadgenesis.transformer.mtp import mtp_loss as _mtp_loss

        logits_list = self.mtp_head(hidden, targets, self.cad_embed)
        loss, breakdown = _mtp_loss(
            logits_list,
            targets,
            self.mtp_head.mtp_depth,
            pad_id=0,
        )
        return loss, breakdown

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_in_ids: torch.Tensor,
        tgt_type_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_bank: torch.Tensor | None = None,
        agent_states: torch.Tensor | None = None,
        constraint_mask: torch.Tensor | None = None,
        adaptive: AdaptiveController | None = None,
        geometry_coords: torch.Tensor | None = None,
        src_attn_mask: torch.Tensor | None = None,
        tgt_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
    ):
        """
        Full Forward Pass:
        src_ids: (B, S) - Language request tokens
        tgt_in_ids: (B, T) - CAD sequence inputs (shifted right)
        tgt_type_ids: (B, T) - Token type family ids
        adaptive: optional self-designing controller (layer routing / head gating)
        geometry_coords: (B, T, 3) or (T, 3), optional — used when
            ``model.geometry_pos_encoding`` is enabled (applied to the target).
        src_attn_mask / tgt_attn_mask / cross_attn_mask: optional
            block-diagonal masks for packed sequences.
        return_hidden: when True, returns ``(logits, confidence, hidden)`` with
            ``hidden`` = decoder norm output (B, T, d_model) — required by the
            MTP auxiliary head.

        Returns: (cad_logits: (B, T, cad_vocab_size), confidence_scores: (B, T, 1))
        """
        encoder_states = self.encode(
            src_ids,
            src_key_padding_mask=src_key_padding_mask,
            adaptive=adaptive,
            src_attn_mask=src_attn_mask,
        )
        logits, confidence = self.decode(
            tgt_in_ids=tgt_in_ids,
            tgt_type_ids=tgt_type_ids,
            encoder_hidden_states=encoder_states,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_key_padding_mask=src_key_padding_mask,
            memory_bank=memory_bank,
            agent_states=agent_states,
            constraint_mask=constraint_mask,
            adaptive=adaptive,
            geometry_coords=geometry_coords,
            tgt_attn_mask=tgt_attn_mask,
            cross_attn_mask=cross_attn_mask,
        )
        if return_hidden:
            hidden = self.decode_hidden_states
            return logits, confidence, hidden
        return logits, confidence
