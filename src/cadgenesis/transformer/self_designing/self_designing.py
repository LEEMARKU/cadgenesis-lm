"""
cadgenesis.transformer.self_designing.self_designing
==============================================
SelfDesigningTransformer — the self-designing orchestrator for CADGenesis-LM v2.0.

This module upgrades the previous thin wrapper into a complete
*self-designing* controller that reuses the entire existing
``GeometryAwareTransformer`` backbone without rebuilding it.  It adds:

    1. Neural Architecture Search       — ``search_architecture()``
    2. Dynamic Layer Routing            — ``DynamicLayerRouter`` (per-token skip)
    3. Adaptive Attention Heads         — ``AdaptiveAttentionHeadSelector``
    4. Sparse Expert Growth             — growable ``SparseMoEFFN`` experts
    5. Layer Pruning                    — ``LayerPruningController`` (reversible)
    6. Architecture Evaluation          — ``ArchitectureEvaluator``
    7. Automatic Rollback               — ``AutomaticRollback`` (versioned)

The controller implements the backbone's duck-typed *adaptive* interface::

    layer_gate(layer_idx, x, layer_type) -> (B, T, 1) | None
    head_weights(layer_idx, x, layer_type) -> (B, T, H) | None

so the unchanged forward contract of ``GeometryAwareTransformer`` is preserved
and existing training code keeps working.

Complexity
----------
    Forward:   O(Σ_l (B·T·C²·heads + B·T·FFN))  — identical to the backbone;
               routing/head gating add O(B·T·L·C) and O(B·T·L·H·C).
    NAS:       O(I · eval_cost)  for I iterations.
    Growth:    O(E·d)  per new expert (amortised).
"""

from __future__ import annotations

import contextlib
from typing import Any, cast

import torch
import torch.nn as nn

from cadgenesis.config import CADConfig
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer
from cadgenesis.transformer.self_designing.adaptive_heads import AdaptiveAttentionHeadSelector
from cadgenesis.transformer.self_designing.architecture import (
    ArchitectureSpec,
    NeuralArchitectureSearch,
)
from cadgenesis.transformer.self_designing.evaluation import ArchitectureEvaluator
from cadgenesis.transformer.self_designing.pruning import LayerPruningController
from cadgenesis.transformer.self_designing.rollback import AutomaticRollback
from cadgenesis.transformer.self_designing.routing import DynamicLayerRouter
from cadgenesis.transformer.transformer_block import CADTransformerBlock


class SelfDesigningTransformer(nn.Module):
    """
    Self-Designing & Dynamic Neural Architecture wrapper around the
    GeometryAwareTransformer backbone.

    Parameters
    ----------
    config : CADConfig
        Master configuration.
    arch_spec : ArchitectureSpec, optional
        Explicit architecture to start from.  Defaults to ``config.model``.
    """

    def __init__(self, config: CADConfig, arch_spec: ArchitectureSpec | None = None):
        super().__init__()
        self.config = config
        self.arch_spec = arch_spec or ArchitectureSpec.from_model_config(config.model)
        self.arch_spec.validate()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = GeometryAwareTransformer(self.arch_spec.to_config(config))

        self._init_controllers()
        self.complexity_evaluator = nn.Linear(config.model.d_model, 1)

        # Per-forward caches for routing/head masks (reset each forward pass).
        self._route_cache: dict[int, torch.Tensor] = {}
        self._head_cache: dict[int, torch.Tensor] = {}

    # --------------------------------------------------------------- helpers

    @property
    def num_encoder_layers(self) -> int:
        return self.arch_spec.num_encoder_layers

    @property
    def num_decoder_layers(self) -> int:
        return self.arch_spec.num_decoder_layers

    @property
    def total_layers(self) -> int:
        return self.num_encoder_layers + self.num_decoder_layers

    def _num_active_heads(self) -> int:
        heads = (
            self.arch_spec.self_attn_heads,
            self.arch_spec.geometry_attn_heads,
            self.arch_spec.constraint_attn_heads,
            self.arch_spec.memory_attn_heads,
            self.arch_spec.agent_attn_heads,
            self.arch_spec.uncertainty_attn_heads,
        )
        return sum(1 for h in heads if h > 0)

    def _init_controllers(self) -> None:
        d_model = self.arch_spec.d_model
        self.router = DynamicLayerRouter(
            d_model=d_model,
            num_layers=self.total_layers,
        )
        self.head_selector = AdaptiveAttentionHeadSelector(
            d_model=d_model,
            num_layers=self.total_layers,
            num_active_heads=self._num_active_heads(),
        )
        self.pruning = LayerPruningController(
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
        )
        self.rollback = AutomaticRollback(self.backbone)

    # ------------------------------------------- backbone adaptive interface

    def layer_gate(
        self,
        layer_idx: int,
        x: torch.Tensor,
        layer_type: str = "encoder",
    ) -> torch.Tensor:
        """
        Per-token layer gate (B, T, 1).  Pruned layers are hard-forced to 0.
        """
        global_idx = layer_idx if layer_type == "encoder" else self.num_encoder_layers + layer_idx
        mask = self._route_mask(x)
        gate = mask[:, :, global_idx : global_idx + 1]
        if self.pruning.is_pruned(layer_type, layer_idx):
            gate = torch.zeros_like(gate)
        return gate

    def head_weights(
        self,
        layer_idx: int,
        x: torch.Tensor,
        layer_type: str = "encoder",
    ) -> torch.Tensor:
        """Per-token adaptive head modulation (B, T, num_active_heads)."""
        global_idx = layer_idx if layer_type == "encoder" else self.num_encoder_layers + layer_idx
        masks = self._head_mask(x)  # (B, T, total_layers, H)
        return masks[:, :, global_idx, :]

    def _route_mask(self, x: torch.Tensor) -> torch.Tensor:
        # Key by tensor identity (not sequence length!): encoder and decoder
        # tensors of the same length must not share masks, and the cache is
        # cleared at the start of every forward pass anyway.
        key = id(x)
        if key not in self._route_cache:
            self._route_cache[key] = self.router(x)
        return self._route_cache[key]

    def _head_mask(self, x: torch.Tensor) -> torch.Tensor:
        key = id(x)
        if key not in self._head_cache:
            self._head_cache[key] = self.head_selector(x)
        return self._head_cache[key]

    def _reset_caches(self) -> None:
        self._route_cache = {}
        self._head_cache = {}

    # -------------------------------------------------------------- forward

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full adaptive forward — identical signature to GeometryAwareTransformer.

        Returns (cad_logits, confidence_scores).
        """
        self._reset_caches()
        return self.backbone(
            src_ids=src_ids,
            tgt_in_ids=tgt_in_ids,
            tgt_type_ids=tgt_type_ids,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_bank=memory_bank,
            agent_states=agent_states,
            constraint_mask=constraint_mask,
            adaptive=self,
        )

    def evaluate_complexity(self, src_ids: torch.Tensor) -> torch.Tensor:
        """Prompt-complexity score in [0, 1] (backward-compatible API)."""
        self._reset_caches()
        encoder_states = self.backbone.encode(src_ids, adaptive=self)
        pooled = encoder_states.mean(dim=1)
        return torch.sigmoid(self.complexity_evaluator(pooled))

    # ------------------------------------------------------ architecture NAS

    def search_architecture(
        self,
        dataset,
        iterations: int = 6,
        mode: str = "random",
        generations: int = 4,
        population_size: int = 4,
    ) -> tuple[ArchitectureSpec, float, dict]:
        """
        Run NAS over the search space, apply the best architecture to this
        controller, and return ``(best_spec, best_score, summary)``.

        dataset: iterable of (src_ids, tgt_ids) pairs used by the evaluator.
        """
        evaluator = ArchitectureEvaluator(
            device=str(self.device),
            train_steps=max(2, min(20, iterations * 2)),
        )
        nas = NeuralArchitectureSearch(
            evaluator=lambda spec: evaluator.score(spec, dataset),
            seed=self.config.seed,
        )
        best_spec, best_score = nas.run(
            iterations=iterations,
            generations=generations,
            population_size=population_size,
            mode=mode,
        )
        self.apply_architecture(best_spec)
        return best_spec, best_score, nas.summary()

    def apply_architecture(self, spec: ArchitectureSpec) -> ArchitectureSpec:
        """
        Adopt a new architecture.  Compatible layers are copied over from the
        current backbone; incompatible layers are freshly initialised.
        """
        spec.validate()
        old_state = self.backbone.state_dict() if hasattr(self, "backbone") else None
        self.arch_spec = spec
        self.config.model = spec.to_model_config()
        new_backbone = GeometryAwareTransformer(self.arch_spec.to_config(self.config))
        if old_state is not None:
            with contextlib.suppress(RuntimeError):
                new_backbone.load_state_dict(old_state, strict=False)
        self.backbone = new_backbone
        self._init_controllers()
        self.complexity_evaluator = nn.Linear(self.arch_spec.d_model, 1)
        return spec

    # -------------------------------------------------------- expert growth

    def grow_experts(self, count: int = 1) -> dict[str, int]:
        """
        Add ``count`` experts to every MoE block.  Returns a mapping of
        ``"{encoder|decoder}:{layer}" -> new expert count``.
        """
        grown: dict[str, int] = {}
        for key, blocks in (
            ("encoder", self.backbone.encoder_blocks),
            ("decoder", self.backbone.decoder_blocks),
        ):
            for i, block in enumerate(blocks):
                moe = cast(CADTransformerBlock, block).moe_layer()
                if moe is not None:
                    for _ in range(count):
                        moe.add_expert()
                    grown[f"{key}:{i}"] = moe.num_experts
        return grown

    def retire_expert(self, layer_type: str, layer_idx: int, expert_idx: int) -> int:
        """Remove one expert from a specific MoE block."""
        blocks = (
            self.backbone.encoder_blocks
            if layer_type == "encoder"
            else self.backbone.decoder_blocks
        )
        moe = cast(CADTransformerBlock, blocks[layer_idx]).moe_layer()
        if moe is None:
            raise ValueError(f"{layer_type}:{layer_idx} is not an MoE block.")
        return moe.remove_expert(expert_idx)

    def expert_load(self) -> dict[str, list[int]]:
        """Per-expert token loads from the last forward pass (diagnostics)."""
        loads: dict[str, list[int]] = {}
        for key, blocks in (
            ("encoder", self.backbone.encoder_blocks),
            ("decoder", self.backbone.decoder_blocks),
        ):
            for i, block in enumerate(blocks):
                moe = cast(CADTransformerBlock, block).moe_layer()
                if moe is not None:
                    loads[f"{key}:{i}"] = moe.expert_load()
        return loads

    # ------------------------------------------------------------ layer prune

    def prune_layers(self, fraction: float = 0.25) -> list[tuple[str, int]]:
        """Compute importance and reversibly prune the weakest layers."""
        self.pruning.compute_importance(self.backbone)
        return self.pruning.prune_layers(fraction=fraction)

    def unprune_layers(self) -> None:
        self.pruning.unprune_all()

    # ------------------------------------------------------------- rollback

    def snapshot(self, metric: float, metadata: dict | None = None) -> str:
        return self.rollback.snapshot(float(metric), self.arch_spec.signature(), metadata)

    def check_performance(self, metric: float, reason: str = "metric deterioration") -> str | None:
        return self.rollback.check_and_rollback(float(metric), reason)

    def rollback_to(self, snapshot_id: str) -> str:
        return self.rollback.restore(snapshot_id)

    # ------------------------------------------------------------- reporting

    def routing_stats(self, x: torch.Tensor | None = None) -> dict:
        if x is None:
            return {"note": "provide x to measure routing sparsity"}
        return {
            "keep_ratio": self.router.keep_ratio(x),
            "active_head_ratio": self.head_selector.active_head_ratio(x),
        }

    def architecture_report(self) -> dict[str, Any]:
        return {
            "arch_spec": self.arch_spec.signature(),
            "repr": repr(self.arch_spec),
            "encoder_layers": self.num_encoder_layers,
            "decoder_layers": self.num_decoder_layers,
            "d_model": self.arch_spec.d_model,
            "use_moe": self.arch_spec.use_moe,
            "pruning": self.pruning.report(),
            "rollback": self.rollback.report(),
            "params": sum(p.numel() for p in self.backbone.parameters()),
        }

    @classmethod
    def from_config(cls, config: CADConfig) -> SelfDesigningTransformer:
        """Factory: build from a CADConfig (uses config.model as the spec)."""
        return cls(config)
