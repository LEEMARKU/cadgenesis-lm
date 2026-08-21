"""tests/transformer/evolution/test_registry.py
=================================================
Unit tests for the Configurable Transformer Evolution layer registry.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from cadgenesis.transformer.evolution.registry import LayerRegistry, global_registry


class TestLayerRegistry:
    def test_builtin_kinds_present(self):
        for kind in (
            "rms_norm",
            "swiglu_ffn",
            "self_attention",
            "geometry_attention",
            "constraint_attention",
            "memory_attention",
            "uncertainty_attention",
            "sparse_attention",
            "multi_scale_attention",
            "moe_ffn",
            "specialized_moe_ffn",
            "cad_block",
        ):
            assert global_registry.has(kind), kind

    def test_build_resolves(self):
        norm = global_registry.build("rms_norm", dim=64)
        assert isinstance(norm, nn.Module)

    def test_build_unknown_raises(self):
        with pytest.raises(KeyError):
            global_registry.build("no_such_layer")

    def test_register_and_unregister(self):
        reg = LayerRegistry()
        reg.register("identity", lambda p: nn.Identity())
        assert reg.has("identity")
        assert isinstance(reg.build("identity"), nn.Identity)
        reg.unregister("identity")
        assert not reg.has("identity")

    def test_register_invalid(self):
        reg = LayerRegistry()
        with pytest.raises(ValueError):
            reg.register("", lambda p: nn.Identity())
        with pytest.raises(TypeError):
            reg.register("bad", "not callable")

    def test_register_from_module(self):
        reg = LayerRegistry()
        reg.register_from_module("my_linear", nn.Linear)
        layer = reg.build("my_linear", in_features=4, out_features=4)
        assert isinstance(layer, nn.Linear)

    def test_snapshot(self):
        assert "rms_norm" in global_registry.snapshot()
