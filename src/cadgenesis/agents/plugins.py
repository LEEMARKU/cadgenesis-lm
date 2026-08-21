"""cadgenesis.agents.plugins
==========================
Plugin interface for extending the agent fleet.

A plugin is a versioned bundle that can contribute agents, register event
handlers and participate in the platform lifecycle.  The plugin interface is
deliberately small so third parties can implement it without importing heavy
infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.agents.base import Agent
from cadgenesis.agents.versioning import AgentVersion


@dataclass
class PluginManifest:
    """Declared metadata for a plugin bundle."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    requires: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def version_obj(self) -> AgentVersion:
        return AgentVersion.parse(self.version)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "requires": list(self.requires),
            "extra": dict(self.extra),
        }


class AgentPlugin:
    """Base class for MAS plugins.

    Subclasses implement :meth:`create_agents` to contribute agents and may
    override the lifecycle hooks.  All hooks are no-ops by default.
    """

    manifest: PluginManifest = PluginManifest(name="base")

    # ------------------------------------------------------------- contribution

    def create_agents(self) -> list[Agent]:
        """Return the agents contributed by this plugin."""
        return []

    # ------------------------------------------------------------- lifecycle

    def on_install(self, platform: Any = None) -> None:
        """Called when the plugin is installed into a platform (no-op)."""

    def on_uninstall(self, platform: Any = None) -> None:
        """Called when the plugin is removed (no-op)."""

    def on_enable(self, platform: Any = None) -> None:
        """Called when the plugin is enabled (no-op)."""

    def on_disable(self, platform: Any = None) -> None:
        """Called when the plugin is disabled (no-op)."""

    def describe(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "agents": [agent.role for agent in self.create_agents()],
        }
