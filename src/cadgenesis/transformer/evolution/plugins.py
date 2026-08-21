"""
cadgenesis.transformer.evolution.plugins
========================================
Plugin architecture for the Configurable Transformer Evolution Framework.

A :class:`Plugin` is a self-contained package of transformer components.  When
activated it registers its layers with a :class:`LayerRegistry` (defaulting to
:data:`~cadgenesis.transformer.evolution.registry.global_registry`).  This is the
"no core edits" path for research: new attention / FFN / block layers ship as
plugins and are resolved by the configuration-driven builder.

Usage::

    from cadgenesis.transformer.evolution.plugins import Plugin, PluginManager

    class MyAttentionPlugin(Plugin):
        name = "my_labs_attention"
        version = "0.1.0"
        def register(self, registry):
            registry.register("my_attention", lambda p: MyAttention(**p))

    manager = PluginManager()
    manager.register_plugin(MyAttentionPlugin())
    manager.activate_all()
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from cadgenesis.transformer.evolution.registry import LayerRegistry

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """Base class for transformer evolution plugins."""

    #: Human-readable plugin name (overridden by subclasses).
    name: str = "unnamed_plugin"
    #: Semantic version string.
    version: str = "0.0.0"

    @abstractmethod
    def register(self, registry: LayerRegistry) -> None:
        """Register this plugin's layers with ``registry``."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "version": self.version}


class PluginManager:
    """
    Registry of plugins.  Activation registers all plugin layers once.
    """

    def __init__(self, registry: LayerRegistry | None = None) -> None:
        if registry is None:
            from cadgenesis.transformer.evolution.registry import global_registry

            registry = global_registry
        self.registry = registry
        self._plugins: dict[str, Plugin] = {}
        self._activated: set[str] = set()

    def register_plugin(self, plugin: Plugin) -> None:
        """Register a plugin instance (does not activate it yet)."""
        if not isinstance(plugin, Plugin):
            raise TypeError("plugin must be a Plugin instance.")
        if plugin.name in self._plugins:
            raise ValueError(f"plugin {plugin.name!r} already registered.")
        self._plugins[plugin.name] = plugin
        logger.info("registered plugin %s v%s", plugin.name, plugin.version)

    def activate_all(self) -> list[str]:
        """Activate every registered plugin; returns newly-activated names."""
        activated: list[str] = []
        for name, plugin in self._plugins.items():
            if name not in self._activated:
                plugin.register(self.registry)
                self._activated.add(name)
                activated.append(name)
                logger.info("activated plugin %s", name)
        return activated

    def is_active(self, name: str) -> bool:
        return name in self._activated

    def list_plugins(self) -> list[dict]:
        return [p.describe() for p in self._plugins.values()]

    def unregister_plugin(self, name: str) -> None:
        """Remove a plugin (its registered layers remain unless unregistered)."""
        del self._plugins[name]
        self._activated.discard(name)


def register_layer(
    kind: str,
    registry: LayerRegistry | None = None,
) -> Callable[[type | Callable[..., Any]], type | Callable[..., Any]]:
    """
    Decorator that registers a module *class* (or factory) under ``kind``.

    When decorating an ``nn.Module`` subclass the constructor kwargs are passed
    through automatically.  When decorating a plain callable the callable is
    used verbatim (it receives the params dict).

    Example::

        from cadgenesis.transformer.evolution.plugins import register_layer

        @register_layer("my_attention")
        class MyAttention(nn.Module):
            def __init__(self, d_model, num_heads, **kw): ...

    Calling the decorator with a registry is also supported:
    ``@register_layer("x", registry=custom_registry)``.
    """

    def decorator(target: type | Callable[..., Any]) -> type | Callable[..., Any]:
        from cadgenesis.transformer.evolution.registry import global_registry

        reg = registry if registry is not None else global_registry
        if isinstance(target, type):
            reg.register_from_module(kind, target)
        else:
            reg.register(kind, target)
        return target

    return decorator
