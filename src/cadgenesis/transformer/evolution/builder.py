"""
cadgenesis.transformer.evolution.builder
========================================
Configuration-driven architecture builder.

This is the "no core edits" entry point of the Configurable Transformer
Evolution Framework: an architecture is described as a plain dict/JSON and
materialised into a runnable module.  Layer kinds referenced by the spec are
resolved through a :class:`~cadgenesis.transformer.evolution.registry.LayerRegistry`
(by default :data:`~cadgenesis.transformer.evolution.registry.global_registry`),
so a researcher can introduce a brand-new attention or FFN layer, register it,
and point the builder at a spec that uses it — without touching the backbone.

Supported architecture ``type`` values:

* ``"standard"`` — the classic
  :class:`~cadgenesis.transformer.geometry_transformer.GeometryAwareTransformer`.
* ``"hierarchical"`` — the five-stage
  :class:`~cadgenesis.transformer.hierarchical_transformer.HierarchicalCADTransformer`.
* ``"stack"``     — a raw sequential stack of registry-resolved layers
  (:class:`RegistryStack`), for component-level experiments.

Example spec::

    {
        "type": "hierarchical",
        "name": "cadgenesis-hier-v1",
        "d_model": 128,
        "nhead": 4,
        "heads": {"self": 2, "geometry": 1, "agent": 1},
        "encoder_layers": 1,
        "stages": {"geometry": 2, "constraint": 1, "execution": 1, "validation": 1},
        "ffn": "specialized_moe",
        "attention": "sparse",          # resolved via the layer registry
        "computation_budget": 0.8,
    }
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from cadgenesis.config import CADConfig, ModelConfig
from cadgenesis.transformer.evolution.registry import LayerRegistry, global_registry

logger = logging.getLogger(__name__)

_SUPPORTED_TYPES = ("standard", "hierarchical", "stack")
_HEAD_FIELDS = ("self", "geometry", "constraint", "memory", "agent", "uncertainty")

# Shorthand names from architecture specs -> registry layer kinds.
_FFN_KINDS = {
    "moe": "moe_ffn",
    "specialized_moe": "specialized_moe_ffn",
}
_ATTENTION_KINDS = {
    "sparse": "sparse_attention",
    "multi_scale": "multi_scale_attention",
}


class RegistryStack(nn.Module):
    """
    Sequential stack of registry-resolved layers.

    Each layer must accept ``forward(x: (B, T, C)) -> (B, T, C)`` (e.g. an
    attention or FFN module).  Built by
    :meth:`ConfigurationDrivenBuilder.build_stack`.
    """

    def __init__(self, layers: list[nn.Module], name: str = "registry_stack") -> None:
        super().__init__()
        self.name = name
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ConfigurationDrivenBuilder:
    """
    Turn architecture specs (dicts/JSON) into runnable transformer models.

    Parameters
    ----------
    registry : LayerRegistry | None
        Layer registry used to resolve layer kinds (defaults to the global one).
    config : CADConfig | None
        Base configuration whose non-model groups (tokenizer, training, ...) are
        reused when materialising a model.
    """

    def __init__(
        self,
        registry: LayerRegistry | None = None,
        config: CADConfig | None = None,
    ) -> None:
        self.registry = registry or global_registry
        self.config = config or CADConfig.mini()

    # ------------------------------------------------------------ validation

    def validate(self, architecture: dict) -> None:
        """Fast-fail on malformed or unresolvable architecture specs."""
        if not isinstance(architecture, dict):
            raise TypeError("architecture must be a dict.")
        arch_type = architecture.get("type", "standard")
        if arch_type not in _SUPPORTED_TYPES:
            raise ValueError(
                f"unknown architecture type {arch_type!r}; choose from {_SUPPORTED_TYPES}."
            )
        if arch_type == "stack":
            layers = architecture.get("layers", [])
            if not isinstance(layers, list) or not layers:
                raise ValueError("a 'stack' architecture requires a non-empty 'layers' list.")
            for i, layer in enumerate(layers):
                if not isinstance(layer, dict) or "kind" not in layer:
                    raise ValueError(f"stack layer {i} must be a dict with a 'kind'.")
                if not self.registry.has(layer["kind"]):
                    raise KeyError(
                        f"stack layer {i}: unknown kind {layer['kind']!r}; "
                        f"registered: {self.registry.kinds()}."
                    )
        for key, mapping in (("ffn", _FFN_KINDS), ("attention", _ATTENTION_KINDS)):
            if key in architecture and architecture[key] not in ("default", None):
                kind = mapping.get(architecture[key], architecture[key])
                if not self.registry.has(kind):
                    raise KeyError(
                        f"unregistered {key} kind {architecture[key]!r}; "
                        f"registered: {self.registry.kinds()}."
                    )

    # -------------------------------------------------------------- building

    def build_stack(self, layer_specs: list[dict], name: str = "registry_stack") -> RegistryStack:
        """Build a :class:`RegistryStack` from a list of ``{kind, params}`` specs."""
        if not layer_specs:
            raise ValueError("layer_specs must be non-empty.")
        layers = []
        for spec in layer_specs:
            kind = spec["kind"]
            if not self.registry.has(kind):
                raise KeyError(f"unknown layer kind {kind!r}.")
            layers.append(self.registry.build(kind, **spec.get("params", {})))
        return RegistryStack(layers, name=name)

    def _config_from_architecture(self, architecture: dict) -> CADConfig:
        """Materialise a ModelConfig from a spec, reusing ``self.config`` for the rest."""
        cfg = CADConfig(
            tokenizer=self.config.tokenizer,
            training=self.config.training,
            lora=self.config.lora,
            memory=self.config.memory,
            observability=self.config.observability,
            seed=self.config.seed,
        )
        m: ModelConfig = cfg.model
        if "d_model" in architecture:
            m.d_model = int(architecture["d_model"])
        if "nhead" in architecture:
            m.nhead = int(architecture["nhead"])
        if "encoder_layers" in architecture:
            m.num_encoder_layers = int(architecture["encoder_layers"])
        if "decoder_layers" in architecture:
            m.num_decoder_layers = int(architecture["decoder_layers"])
        if "dropout" in architecture:
            m.dropout = float(architecture["dropout"])
        if "computation_budget" in architecture:
            m.computation_budget = float(architecture["computation_budget"])
        if "early_exit_threshold" in architecture:
            m.early_exit_threshold = float(architecture["early_exit_threshold"])

        heads = architecture.get("heads", {})
        if heads:
            # A head layout fully specifies the mixture: zero the defaults that
            # are not listed so the layout is interpreted exactly.
            for field in _HEAD_FIELDS:
                setattr(m, f"{field}_attn_heads", 0)
            m.self_attn_heads = int(heads.get("self", 0))
            m.geometry_attn_heads = int(heads.get("geometry", 0))
            m.constraint_attn_heads = int(heads.get("constraint", 0))
            m.memory_attn_heads = int(heads.get("memory", 0))
            m.agent_attn_heads = int(heads.get("agent", 0))
            m.uncertainty_attn_heads = int(heads.get("uncertainty", 0))
            # Agent/memory heads are only valid when their producing subsystem
            # is enabled (ModelConfig validation enforces this).
            m.use_multi_agent_system = m.agent_attn_heads > 0
            m.use_memory_system = m.memory_attn_heads > 0

        stages = architecture.get("stages", {})
        if stages:
            for stage, depth in stages.items():
                if stage not in ("planner", "geometry", "constraint", "execution", "validation"):
                    raise ValueError(f"unknown hierarchical stage {stage!r}.")
                setattr(m, f"{stage}_layers", int(depth))

        ffn = architecture.get("ffn", "default")
        if ffn == "moe":
            m.use_moe = True
        elif ffn == "specialized_moe":
            m.use_specialized_moe = True
        attention = architecture.get("attention", "default")
        if attention == "sparse":
            m.sparse_attention = True
        elif attention == "multi_scale":
            m.use_multi_scale_attention = True
        if "sliding_window_size" in architecture:
            m.sliding_window_size = int(architecture["sliding_window_size"])
        if "experts_per_domain" in architecture:
            m.experts_per_domain = int(architecture["experts_per_domain"])
        if "top_k_domain_experts" in architecture:
            m.top_k_domain_experts = int(architecture["top_k_domain_experts"])

        # Rescale head counts if nhead changed and no explicit layout given.
        if (
            heads
            and sum(
                (
                    m.self_attn_heads,
                    m.geometry_attn_heads,
                    m.constraint_attn_heads,
                    m.memory_attn_heads,
                    m.agent_attn_heads,
                    m.uncertainty_attn_heads,
                )
            )
            != m.nhead
        ):
            raise ValueError("heads must sum to nhead.")
        cfg._validate()
        return cfg

    def build_model(self, architecture: dict) -> nn.Module:
        """
        Build a runnable model from an architecture spec (see module docstring).
        """
        self.validate(architecture)
        arch_type = architecture.get("type", "standard")
        if arch_type == "stack":
            return self.build_stack(architecture.get("layers", []))
        if arch_type == "hierarchical":
            cfg = self._config_from_architecture(architecture)
            from cadgenesis.transformer.hierarchical_transformer import (
                HierarchicalCADTransformer,
            )

            return HierarchicalCADTransformer(cfg)
        # standard
        cfg = self._config_from_architecture(architecture)
        from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer

        return GeometryAwareTransformer(cfg)

    def build_from_config(self, config: CADConfig | None = None) -> nn.Module:
        """Build the canonical model for a :class:`CADConfig`."""
        cfg = config or self.config
        if cfg.model.use_hierarchical_transformer:
            from cadgenesis.transformer.hierarchical_transformer import (
                HierarchicalCADTransformer,
            )

            return HierarchicalCADTransformer(cfg)
        from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer

        return GeometryAwareTransformer(cfg)

    # -------------------------------------------------------------- describe

    def describe(self, architecture: dict) -> dict:
        """Human-readable resolution plan for a spec (no model is built)."""
        self.validate(architecture)
        arch_type = architecture.get("type", "standard")
        plan: dict[str, Any] = {
            "type": arch_type,
            "name": architecture.get("name", "unnamed"),
        }
        if arch_type == "stack":
            plan["layers"] = [
                {"kind": layer["kind"], "params": list(layer.get("params", {}))}
                for layer in architecture["layers"]
            ]
        else:
            plan["d_model"] = architecture.get("d_model", "config-default")
            plan["ffn"] = architecture.get("ffn", "default")
            plan["attention"] = architecture.get("attention", "default")
        return plan
