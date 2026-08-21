"""
cadgenesis.transformer.evolution.registry
==========================================
Layer Registry for the Configurable Transformer Evolution Framework.

The registry maps **layer-kind names** to *factory callables*::

    factory(kind_params: dict) -> torch.nn.Module

Architecture specs (dicts / JSON) reference layers by name; the registry
resolves them into concrete modules.  Researchers add new transformer
components by registering a factory — no core code is touched.  Every built-in
layer is pre-registered on :data:`global_registry`.

Layer kinds
-----------
===============  =============================================================
kind             factory
===============  =============================================================
``rms_norm``     :class:`cadgenesis.transformer.transformer_block.RMSNorm`
``swiglu_ffn``   :class:`cadgenesis.transformer.transformer_block.SwiGLU`
``self_attention`` :class:`cadgenesis.transformer.attention.SelfAttention`
``geometry_attention`` :class:`cadgenesis.transformer.attention.GeometryAttention`
``constraint_attention`` :class:`cadgenesis.transformer.attention.ConstraintAttention`
``memory_attention`` :class:`cadgenesis.transformer.attention.MemoryAttention`
``uncertainty_attention`` :class:`cadgenesis.transformer.attention.UncertaintyAttention`
``sparse_attention`` :class:`cadgenesis.transformer.sparse_attention.SparseSelfAttention`
``multi_scale_attention`` :class:`cadgenesis.transformer.multi_scale_attention.MultiScaleAttention`
``moe_ffn``     :class:`cadgenesis.transformer.moe.SparseMoEFFN`
``specialized_moe_ffn`` :class:`cadgenesis.transformer.specialized_moe.SpecializedMoEFFN`
``cad_block``   :class:`cadgenesis.transformer.transformer_block.CADTransformerBlock`
===============  =============================================================
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)

#: Callable signature: ``(params: dict) -> nn.Module``.
LayerFactory = Callable[[dict], nn.Module]


class LayerRegistry:
    """Name → factory registry with validation and plugin hooks."""

    def __init__(self) -> None:
        self._factories: dict[str, LayerFactory] = {}

    def register(self, kind: str, factory: LayerFactory) -> None:
        """Register (or replace) the factory for a layer kind."""
        kind = kind.strip()
        if not kind:
            raise ValueError("layer kind must be non-empty.")
        if not callable(factory):
            raise TypeError("factory must be callable.")
        self._factories[kind] = factory
        logger.debug("registered layer kind %r", kind)

    def unregister(self, kind: str) -> None:
        """Remove a layer kind (raises ``KeyError`` if unknown)."""
        del self._factories[kind]

    def has(self, kind: str) -> bool:
        return kind in self._factories

    def kinds(self) -> list[str]:
        """Sorted list of registered layer kind names."""
        return sorted(self._factories)

    def build(self, kind: str, **params: Any) -> nn.Module:
        """
        Resolve ``kind`` into a module, passing ``params`` to its factory.
        """
        factory = self._factories.get(kind)
        if factory is None:
            raise KeyError(f"Unknown layer kind {kind!r}; registered kinds: {self.kinds()}.")
        module = factory(params)
        if not isinstance(module, nn.Module):
            raise TypeError(f"factory for {kind!r} must return nn.Module, got {type(module)}.")
        return module

    def register_from_module(
        self,
        kind: str,
        module_class: type[nn.Module],
        required_params: tuple[str, ...] = (),
    ) -> None:
        """Register a module *class*: params are passed as constructor kwargs."""

        def factory(params: dict) -> nn.Module:
            missing = [p for p in required_params if p not in params]
            if missing:
                raise KeyError(f"{kind!r} requires params {missing}.")
            return module_class(**params)

        self.register(kind, factory)

    def snapshot(self) -> dict[str, str]:
        """Diagnostic snapshot of registered kinds and their factories."""
        return {k: getattr(v, "__name__", repr(v)) for k, v in self._factories.items()}


def _builtin_factories() -> dict[str, LayerFactory]:
    """Lazy registration of the built-in layers (deferred imports avoid cycles)."""

    def _rms_norm(p: dict) -> nn.Module:
        from cadgenesis.transformer.transformer_block import RMSNorm

        return RMSNorm(**p)

    def _swiglu(p: dict) -> nn.Module:
        from cadgenesis.transformer.transformer_block import SwiGLU

        return SwiGLU(**p)

    def _self_attn(p: dict) -> nn.Module:
        from cadgenesis.transformer.attention import SelfAttention

        return SelfAttention(**p)

    def _geometry_attn(p: dict) -> nn.Module:
        from cadgenesis.transformer.attention import GeometryAttention

        return GeometryAttention(**p)

    def _constraint_attn(p: dict) -> nn.Module:
        from cadgenesis.transformer.attention import ConstraintAttention

        return ConstraintAttention(**p)

    def _memory_attn(p: dict) -> nn.Module:
        from cadgenesis.transformer.attention import MemoryAttention

        return MemoryAttention(**p)

    def _uncertainty_attn(p: dict) -> nn.Module:
        from cadgenesis.transformer.attention import UncertaintyAttention

        return UncertaintyAttention(**p)

    def _sparse_attn(p: dict) -> nn.Module:
        from cadgenesis.transformer.sparse_attention import build_sparse_attention

        pattern = p.pop("pattern", "sliding_window")
        return build_sparse_attention(pattern, **p)

    def _multi_scale_attn(p: dict) -> nn.Module:
        from cadgenesis.transformer.multi_scale_attention import MultiScaleAttention

        return MultiScaleAttention(**p)

    def _moe_ffn(p: dict) -> nn.Module:
        from cadgenesis.transformer.moe import SparseMoEFFN

        return SparseMoEFFN(**p)

    def _specialized_moe_ffn(p: dict) -> nn.Module:
        from cadgenesis.transformer.specialized_moe import SpecializedMoEFFN

        return SpecializedMoEFFN(**p)

    def _cad_block(p: dict) -> nn.Module:
        from cadgenesis.transformer.transformer_block import CADTransformerBlock

        return CADTransformerBlock(**p)

    return {
        "rms_norm": _rms_norm,
        "swiglu_ffn": _swiglu,
        "self_attention": _self_attn,
        "geometry_attention": _geometry_attn,
        "constraint_attention": _constraint_attn,
        "memory_attention": _memory_attn,
        "uncertainty_attention": _uncertainty_attn,
        "sparse_attention": _sparse_attn,
        "multi_scale_attention": _multi_scale_attn,
        "moe_ffn": _moe_ffn,
        "specialized_moe_ffn": _specialized_moe_ffn,
        "cad_block": _cad_block,
    }


def _make_global_registry() -> LayerRegistry:
    registry = LayerRegistry()
    for kind, factory in _builtin_factories().items():
        registry.register(kind, factory)
    return registry


#: Process-wide registry pre-populated with every built-in layer.
global_registry = _make_global_registry()
