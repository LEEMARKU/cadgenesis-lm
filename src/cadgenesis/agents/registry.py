"""cadgenesis.agents.registry
==========================
Central agent registry with capability discovery.

A single :class:`AgentRegistry` is the source of truth for the agent fleet.
It stores agents by role, supports capability / action queries, tracks agent
versions and can produce a serializable snapshot for health dashboards.
"""

from __future__ import annotations

import threading
from typing import Any

from cadgenesis.agents.base import Agent
from cadgenesis.agents.infrastructure import AgentBase


class RegistryError(ValueError):
    """Raised for invalid registration or lookup operations."""


class AgentRegistry:
    """Thread-safe registry mapping ``role -> Agent``."""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._lock = threading.RLock()

    # ---------------------------------------------------------------- mutation

    def register(self, agent: Agent) -> AgentRegistry:
        """Register an agent under its role name.

        Raises :class:`RegistryError` when the role is already taken.
        """
        if not getattr(agent, "role", ""):
            raise RegistryError("cannot register an agent without a role")
        with self._lock:
            if agent.role in self._agents:
                raise RegistryError(f"agent role {agent.role!r} already registered")
            self._agents[agent.role] = agent
        if isinstance(agent, AgentBase):
            agent.on_register(self)
        return self

    def register_many(self, agents: list[Agent]) -> AgentRegistry:
        for agent in agents:
            self.register(agent)
        return self

    def unregister(self, role: str) -> Agent | None:
        """Remove and return the agent for ``role`` (None if absent)."""
        with self._lock:
            agent = self._agents.pop(role, None)
        if agent is not None and isinstance(agent, AgentBase):
            agent.on_unregister(self)
        return agent

    def clear(self) -> None:
        with self._lock:
            self._agents.clear()

    # ----------------------------------------------------------------- lookups

    def get(self, role: str) -> Agent | None:
        with self._lock:
            return self._agents.get(role)

    def require(self, role: str) -> Agent:
        """Return the agent for ``role`` or raise :class:`RegistryError`."""
        agent = self.get(role)
        if agent is None:
            raise RegistryError(f"no agent registered for role {role!r}")
        return agent

    def __contains__(self, role: object) -> bool:
        with self._lock:
            return role in self._agents

    def __len__(self) -> int:
        with self._lock:
            return len(self._agents)

    @property
    def agents(self) -> list[Agent]:
        with self._lock:
            return list(self._agents.values())

    @property
    def roles(self) -> list[str]:
        with self._lock:
            return list(self._agents.keys())

    # ------------------------------------------------------ capability discovery

    def find_by_action(self, action: str) -> list[Agent]:
        """All agents able to handle ``action``."""
        return [a for a in self.agents if a.can_handle(action)]

    def find_by_capability(self, capability: str) -> list[Agent]:
        """All agents exposing a structured capability named ``capability``."""
        return [
            agent
            for agent in self.agents
            if isinstance(agent, AgentBase) and agent.has_capability(capability)
        ]

    def capabilities(self) -> list[dict[str, Any]]:
        """Flattened capability manifest across the whole fleet."""
        return [
            {"role": agent.role, **cap.to_dict()}
            for agent in self.agents
            if isinstance(agent, AgentBase)
            for cap in agent.structured_capabilities()
        ]

    def discover(self, query: str | None = None) -> list[str]:
        """Roles matching ``query`` by role, action or capability name.

        ``query=None`` returns every registered role.
        """
        roles: list[str] = []
        for agent in self.agents:
            if query is None or query in agent.role or query in agent.actions:
                roles.append(agent.role)
                continue
            if isinstance(agent, AgentBase) and any(
                query == cap.name for cap in agent.structured_capabilities()
            ):
                roles.append(agent.role)
        return roles

    # --------------------------------------------------------------- reporting

    def snapshot(self) -> list[dict[str, Any]]:
        """Serializable fleet snapshot (version, state, actions, capabilities)."""
        out: list[dict[str, Any]] = []
        for agent in self.agents:
            entry: dict[str, Any] = {
                "role": agent.role,
                "actions": list(agent.actions),
                "version": agent.version if isinstance(agent, AgentBase) else "1.0.0",
            }
            if isinstance(agent, AgentBase):
                entry["state"] = agent.state.value
                entry["capabilities"] = [c.to_dict() for c in agent.structured_capabilities()]
            out.append(entry)
        return out

    def summary(self) -> dict[str, Any]:
        return {"count": len(self), "roles": self.roles}
