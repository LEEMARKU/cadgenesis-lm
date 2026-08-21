"""
cadgenesis.transformer.hierarchical_transformer
===============================================
Hierarchical CAD Transformer for CADGenesis-LM v6.0 (Pillar 1).

CAD generation is naturally a *staged* process.  This module implements the
five-stage hierarchical pipeline from the Foundation Model specification::

    Planner Layer  (language / intent -> plan representation)
        ↓
    Geometry Layer (plan -> geometric primitives & B-Rep structure)
        ↓
    Constraint Layer (geometry -> dimension / geometric / assembly / dependency constraints)
        ↓
    Execution Layer (constrained model -> executable feature operations)
        ↓
    Validation Layer (executable sequence -> validity checks & final output)

Each stage is an independent group of :class:`CADTransformerBlock` layers with
its own RMSNorm and a learnable stage-scale, so the model can modulate the
contribution of every discipline.  The pipeline is *dynamic*: a
:class:`cadgenesis.transformer.dynamic_routing.DynamicRoutingController`
enforces the computation budget and can early-exit at any stage boundary once
confidence is high enough.

API contract
------------
The forward signature mirrors
:class:`cadgenesis.transformer.geometry_transformer.GeometryAwareTransformer`
exactly — ``forward(src_ids, tgt_in_ids, tgt_type_ids, ...) -> (logits, confidence)``
— so the existing trainer (:mod:`cadgenesis.training.trainer`), inference engine
(:mod:`cadgenesis.inference.engine`) and LoRA/distillation tooling work without
changes.

Specialized MoE
---------------
When ``config.model.use_specialized_moe`` is enabled, every stage FFN is a
:class:`cadgenesis.transformer.specialized_moe.SpecializedMoEFFN` (geometry /
manufacturing / reasoning / simulation / optimization experts) instead of the
dense SwiGLU.  The block construction is untouched — the specialised FFN is
injected via the block's existing ``ffn``/``use_moe`` contract, so backward
compatibility is preserved.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

from cadgenesis.config import CADConfig, ModelConfig
from cadgenesis.transformer.dynamic_routing import DynamicRoutingController
from cadgenesis.transformer.geometry_transformer import AdaptiveController
from cadgenesis.transformer.positional import (
    GeometryPositionalEncoding,
    SinusoidalPositionalEncoding,
)
from cadgenesis.transformer.specialized_moe import SpecializedMoEFFN
from cadgenesis.transformer.transformer_block import CADTransformerBlock, RMSNorm

logger = logging.getLogger(__name__)

STAGE_NAMES = ("planner", "geometry", "constraint", "execution", "validation")


def _stage_depth(config: ModelConfig, stage: str) -> int:
    """Per-stage depth from the configuration (planner/geometry/.../validation)."""
    return int(getattr(config, f"{stage}_layers"))


def _cad_vocab_size(t_cfg) -> int:
    return (
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


def _build_stage_block(
    m_cfg: ModelConfig,
    *,
    use_specialized_moe: bool,
) -> CADTransformerBlock:
    """
    Build one CADTransformerBlock, optionally injecting a specialised MoE FFN.

    The block is constructed with ``use_moe=False`` (dense SwiGLU) and, when
    requested, its FFN is replaced by a :class:`SpecializedMoEFFN` through the
    block's existing ``ffn`` / ``use_moe`` interface — the block itself is not
    modified.
    """
    block = CADTransformerBlock(
        d_model=m_cfg.d_model,
        self_heads=m_cfg.self_attn_heads,
        geometry_heads=m_cfg.geometry_attn_heads,
        constraint_heads=m_cfg.constraint_attn_heads,
        memory_heads=m_cfg.memory_attn_heads,
        agent_heads=m_cfg.agent_attn_heads,
        uncertainty_heads=m_cfg.uncertainty_attn_heads,
        dim_feedforward=m_cfg.dim_feedforward,
        dropout=m_cfg.dropout,
        use_moe=False,
        self_attn_backend=m_cfg.attention_backend,
        use_feature_interaction=m_cfg.feature_interaction,
        interaction_heads=m_cfg.interaction_heads,
    )
    if use_specialized_moe:
        block.ffn = SpecializedMoEFFN(
            d_model=m_cfg.d_model,
            experts_per_domain=m_cfg.experts_per_domain,
            top_k=m_cfg.top_k_domain_experts,
            dropout=m_cfg.dropout,
        )
        block.use_moe = True
    return block


class HierarchicalCADTransformer(nn.Module):
    """
    Five-stage hierarchical encoder-decoder for generative parametric CAD.

    Parameters
    ----------
    config : CADConfig
        Master configuration.  ``config.model.use_hierarchical_transformer`` is
        not required here (the class is explicitly chosen by the caller) but the
        per-stage depths and dynamic-routing knobs are read from it.
    """

    STAGES = STAGE_NAMES

    def __init__(self, config: CADConfig):
        super().__init__()
        self.config = config
        m_cfg: ModelConfig = config.model
        t_cfg = config.tokenizer

        if any(_stage_depth(m_cfg, s) < 1 for s in STAGE_NAMES):
            raise ValueError("every hierarchical stage must have >= 1 layer.")

        self.d_model = m_cfg.d_model
        self.cad_vocab_size = _cad_vocab_size(t_cfg)
        self.use_specialized_moe = m_cfg.use_specialized_moe

        # --- embeddings ----------------------------------------------------
        self.lang_embed = nn.Embedding(t_cfg.lang_vocab_size, self.d_model, padding_idx=0)
        self.cad_embed = nn.Embedding(self.cad_vocab_size, self.d_model, padding_idx=0)
        self.type_embed = nn.Embedding(10, self.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(self.d_model, max_len=m_cfg.max_seq_len)
        self.geometry_pos_enc = (
            GeometryPositionalEncoding(d_model=self.d_model)
            if m_cfg.geometry_pos_encoding
            else None
        )

        # --- Planner stage (encoder) ----------------------------------------
        self.planner_blocks = nn.ModuleList(
            [
                _build_stage_block(m_cfg, use_specialized_moe=self.use_specialized_moe)
                for _ in range(_stage_depth(m_cfg, "planner"))
            ]
        )
        self.planner_norm = RMSNorm(self.d_model)

        # --- Geometry / Constraint / Execution / Validation stages (decoder)
        for stage in ("geometry", "constraint", "execution", "validation"):
            setattr(
                self,
                f"{stage}_blocks",
                nn.ModuleList(
                    [
                        _build_stage_block(m_cfg, use_specialized_moe=self.use_specialized_moe)
                        for _ in range(_stage_depth(m_cfg, stage))
                    ]
                ),
            )
            setattr(self, f"{stage}_norm", RMSNorm(self.d_model))
            setattr(self, f"{stage}_scale", nn.Parameter(torch.ones(1)))

        # --- output heads ----------------------------------------------------
        self.out_proj = nn.Linear(self.d_model, self.cad_vocab_size, bias=False)
        self.confidence_head = nn.Linear(self.d_model, 1)

        # --- dynamic computation routing -------------------------------------
        decoder_layers = sum(
            _stage_depth(m_cfg, s) for s in ("geometry", "constraint", "execution", "validation")
        )
        self.routing = DynamicRoutingController(
            total_layers=decoder_layers,
            budget=m_cfg.computation_budget,
            early_exit_threshold=m_cfg.early_exit_threshold,
            min_steps=1,
        )

        self._init_weights()

    # --------------------------------------------------------------- helpers

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    @property
    def total_decoder_layers(self) -> int:
        return sum(
            len(getattr(self, f"{stage}_blocks"))
            for stage in ("geometry", "constraint", "execution", "validation")
        )

    # --------------------------------------------------------------- planner

    def encode(
        self,
        src_ids: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        adaptive: AdaptiveController | None = None,
        geometry_coords: torch.Tensor | None = None,
        memory_bank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the Planner stage over a language/plan prompt.

        ``src_ids``: (B, S).  Returns the plan representation (B, S, d_model).
        """
        src = self.pos_enc(self.lang_embed(src_ids) * math.sqrt(self.d_model))
        if self.geometry_pos_enc is not None:
            src = self.geometry_pos_enc(src, geometry_coords)

        x = src
        for i, block in enumerate(self.planner_blocks):
            gate = adaptive.layer_gate(i, x, "planner") if adaptive is not None else None
            heads = adaptive.head_weights(i, x, "planner") if adaptive is not None else None
            x, _ = block(x, memory_bank=memory_bank, layer_gate=gate, head_weights=heads)
        return self.planner_norm(x)

    # ------------------------------------------------------------------ forward

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full five-stage hierarchical forward pass.

        Returns ``(cad_logits (B, T, cad_vocab_size), confidence (B, T, 1))`` —
        the identical contract of :class:`GeometryAwareTransformer`.
        """
        encoder_states = self.encode(
            src_ids,
            src_key_padding_mask=src_key_padding_mask,
            adaptive=adaptive,
            memory_bank=memory_bank,
        )

        _, T = tgt_in_ids.shape
        tgt = self.cad_embed(tgt_in_ids) + self.type_embed(tgt_type_ids)
        tgt = self.pos_enc(tgt * math.sqrt(self.d_model))
        if self.geometry_pos_enc is not None:
            tgt = self.geometry_pos_enc(tgt, geometry_coords)

        causal_mask = (
            torch.triu(
                torch.full((T, T), float("-inf"), device=tgt.device, dtype=tgt.dtype),
                diagonal=1,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        x = tgt
        conf_last: torch.Tensor | None = None
        stage_offset = _stage_depth(self.config.model, "planner")
        self.routing.reset()
        exited = False

        for stage in ("geometry", "constraint", "execution", "validation"):
            blocks: nn.ModuleList = getattr(self, f"{stage}_blocks")
            stage_mask = constraint_mask if stage == "constraint" else None
            stage_norm: nn.Module = getattr(self, f"{stage}_norm")
            stage_scale = getattr(self, f"{stage}_scale")
            global_start = stage_offset
            stage_offset += len(blocks)

            for i, block in enumerate(blocks):
                gate = (
                    adaptive.layer_gate(global_start + i, x, stage)
                    if adaptive is not None
                    else None
                )
                heads = (
                    adaptive.head_weights(global_start + i, x, stage)
                    if adaptive is not None
                    else None
                )
                x, conf = block(
                    x,
                    encoder_hidden_states=encoder_states,
                    memory_bank=memory_bank,
                    agent_states=agent_states,
                    causal_mask=causal_mask,
                    constraint_mask=stage_mask,
                    layer_gate=gate,
                    head_weights=heads,
                    feature_type_ids=tgt_type_ids,
                )
                if conf is not None:
                    conf_last = conf
                if self.routing.should_stop(
                    i + (global_start - _stage_depth(self.config.model, "planner")),
                    confidence=float(torch.sigmoid(conf).mean()) if conf is not None else None,
                ):
                    exited = True
                    break
            # Stage boundary: normalise + learnable stage scale.
            x = stage_norm(x) * stage_scale
            if exited:
                break

        if conf_last is None:
            conf_last = self.confidence_head(x)
        logits = self.out_proj(x)
        return logits, conf_last

    # ------------------------------------------------------------- diagnostics

    def aux_loss(self) -> torch.Tensor:
        """Sum of auxiliary load-balancing losses from specialised MoE blocks."""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for stage in STAGE_NAMES:
            for block in getattr(self, f"{stage}_blocks"):
                moe = block.moe_layer()
                if moe is not None and hasattr(moe, "get_aux_loss"):
                    total = total + moe.get_aux_loss()
        return total

    def stage_report(self) -> dict:
        """Per-stage layer counts and routing telemetry."""
        stages = {stage: len(getattr(self, f"{stage}_blocks")) for stage in STAGE_NAMES}
        report = {
            "stages": stages,
            "total_decoder_layers": self.total_decoder_layers,
            "routing": self.routing.report(),
        }
        if self.use_specialized_moe:
            report["expert_load"] = {
                f"{stage}:{i}": block.moe_layer().domain_load()
                for stage in STAGE_NAMES
                for i, block in enumerate(getattr(self, f"{stage}_blocks"))
                if block.moe_layer() is not None
            }
        return report
