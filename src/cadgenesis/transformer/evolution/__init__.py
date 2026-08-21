"""
cadgenesis.transformer.evolution
================================
Configurable Transformer Evolution Framework (Pillar 1).

A research-friendly framework for *evolving* the transformer architecture
without touching the core.  Researchers register new layers, describe
architectures declaratively, version them, and track experiments:

* :mod:`cadgenesis.transformer.evolution.registry` — the layer registry: a
  name → factory map that resolves architecture specs into real modules.
* :mod:`cadgenesis.transformer.evolution.plugins` — plugin architecture:
  third-party layer packages can register themselves through plugins or the
  ``@register_layer`` decorator.
* :mod:`cadgenesis.transformer.evolution.versioning` — architecture versioning:
  every architecture carries a semantic version and a content hash.
* :mod:`cadgenesis.transformer.evolution.experiments` — experiment registry:
  reproducible experiment records (config hash + arch version + metric).
* :mod:`cadgenesis.transformer.evolution.builder` — configuration-driven
  architecture builder that turns a JSON/dict spec into a runnable model.
"""

from cadgenesis.transformer.evolution.builder import (
    ConfigurationDrivenBuilder,
    RegistryStack,
)
from cadgenesis.transformer.evolution.experiments import ExperimentRecord, ExperimentRegistry
from cadgenesis.transformer.evolution.plugins import Plugin, PluginManager, register_layer
from cadgenesis.transformer.evolution.registry import LayerRegistry, global_registry
from cadgenesis.transformer.evolution.versioning import (
    ArchitectureVersion,
    VersionedArchitecture,
    hash_architecture,
)

__all__ = [
    "ArchitectureVersion",
    "ConfigurationDrivenBuilder",
    "ExperimentRecord",
    "ExperimentRegistry",
    "LayerRegistry",
    "Plugin",
    "PluginManager",
    "RegistryStack",
    "VersionedArchitecture",
    "global_registry",
    "hash_architecture",
    "register_layer",
]
