"""tests/transformer/evolution/test_plugins.py
================================================
Unit tests for the Configurable Transformer Evolution plugin system.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cadgenesis.transformer.evolution.plugins import Plugin, PluginManager, register_layer
from cadgenesis.transformer.evolution.registry import LayerRegistry, global_registry


class SprocketPlugin(Plugin):
    name = "sprocket_attention"
    version = "1.2.0"

    def register(self, registry):
        def factory(p):
            return nn.Linear(p["d_model"], p["d_model"])

        registry.register("sprocket_attention", factory)


class TestPluginManager:
    def test_activate(self):
        manager = PluginManager()
        manager.register_plugin(SprocketPlugin())
        assert manager.list_plugins() == [{"name": "sprocket_attention", "version": "1.2.0"}]
        activated = manager.activate_all()
        assert activated == ["sprocket_attention"]
        assert manager.is_active("sprocket_attention")
        assert manager.registry.has("sprocket_attention")
        # Second activation is a no-op.
        assert manager.activate_all() == []

    def test_duplicate_plugin_rejected(self):
        manager = PluginManager()

        class P(Plugin):
            name = "p"

            def register(self, registry):
                pass

        manager.register_plugin(P())
        with pytest.raises(ValueError):
            manager.register_plugin(P())

    def test_unregister_plugin(self):
        manager = PluginManager()
        manager.register_plugin(SprocketPlugin())
        manager.activate_all()
        manager.unregister_plugin("sprocket_attention")
        assert manager.list_plugins() == []


class TestRegisterLayerDecorator:
    def test_decorates_class(self):
        @register_layer("test_linear_layer")
        class TestLinear(nn.Module):
            def __init__(self, d_model: int):
                super().__init__()
                self.linear = nn.Linear(d_model, d_model)

            def forward(self, x):
                return self.linear(x)

        assert global_registry.has("test_linear_layer")
        layer = global_registry.build("test_linear_layer", d_model=8)
        assert isinstance(layer, TestLinear)
        assert layer(torch.randn(2, 8)).shape == (2, 8)

    def test_decorates_custom_registry(self):
        reg = LayerRegistry()

        @register_layer("custom_kind", registry=reg)
        def factory(p):
            return nn.Identity()

        assert reg.has("custom_kind")
