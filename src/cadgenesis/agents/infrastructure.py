"""cadgenesis.agents.infrastructure
================================
Pillar 5 agent infrastructure: lifecycle-aware agent base class, structured
capability discovery, agent metadata and the lifecycle manager.

This is **additive** to ``cadgenesis.agents.base.Agent``: the existing
``Agent`` protocol (``role`` / ``actions`` / ``handle`` / ``process`` /
``describe``) is preserved unchanged, while :class:`AgentBase` layers on
versioning, lifecycle state, capabilities and health metadata for the new
production fleet.
"""

from __future__ import annotations

import threading
import time
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cadgenesis.agents.base import Agent, AgentRequest, AgentResult
from cadgenesis.agents.versioning import AgentVersion


class AgentState(str, Enum):
    """Lifecycle states of a managed agent."""

    CREATED = "created"
    REGISTERED = "registered"
    STARTED = "started"
    PAUSED = "paused"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class Capability:
    """A structured, discoverable capability offered by an agent."""

    name: str
    description: str = ""
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    async_supported: bool = False
    cost: float = 1.0
    tags: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "async_supported": self.async_supported,
            "cost": self.cost,
            "tags": sorted(self.tags),
        }


@dataclass
class AgentMetadata:
    """Descriptive metadata attached to an :class:`AgentBase`."""

    description: str = ""
    author: str = ""
    license: str = ""
    dependencies: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "author": self.author,
            "license": self.license,
            "dependencies": list(self.dependencies),
            "extra": dict(self.extra),
        }


class AgentBase(Agent, ABC):
    """Lifecycle-aware, versioned agent base.

    Subclasses declare ``role``, ``actions``, ``version``, and either
    ``capabilities`` (structured) or plain action names, and implement
    :meth:`process` as before.

    Lifecycle: ``created -> registered -> started <-> paused -> stopped``.
    Override the ``on_*`` hooks to react; the defaults are no-ops so existing
    simple agents remain compatible.
    """

    version: str = "1.0.0"
    capabilities: tuple[Capability | str, ...] = ()
    metadata: AgentMetadata = AgentMetadata()

    def __init__(self) -> None:
        super().__init__()
        self._state = AgentState.CREATED
        self._started_at: float | None = None
        self._last_heartbeat: float | None = None
        self._error_count = 0
        self._request_count = 0
        self._lifecycle_lock = threading.Lock()
        self._parsed_version = AgentVersion.parse(self.version)

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def version_obj(self) -> AgentVersion:
        return self._parsed_version

    @property
    def started_at(self) -> float | None:
        return self._started_at

    @property
    def last_heartbeat(self) -> float | None:
        return self._last_heartbeat

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def request_count(self) -> int:
        return self._request_count

    def heartbeat(self) -> None:
        """Called by the health monitor; records a liveness tick."""
        self._last_heartbeat = time.time()

    def is_alive(self, timeout: float = 30.0) -> bool:
        """True while heartbeats keep arriving within ``timeout`` seconds."""
        if self._last_heartbeat is None:
            return False
        return (time.time() - self._last_heartbeat) <= timeout

    # ---------------------------------------------------------- lifecycle hooks

    def on_register(self, registry: Any = None) -> None:
        """Called when the agent is added to a registry (no-op by default)."""

    def on_unregister(self, registry: Any = None) -> None:
        """Called when the agent is removed from a registry (no-op)."""

    def on_start(self) -> None:
        """Startup hook (no-op)."""

    def on_pause(self) -> None:
        """Pause hook (no-op)."""

    def on_resume(self) -> None:
        """Resume hook (no-op)."""

    def on_stop(self) -> None:
        """Shutdown hook (no-op)."""

    def on_error(self, error: Exception) -> None:
        """Error hook; records the failure (no-op beyond bookkeeping)."""
        self._error_count += 1

    # ---------------------------------------------------------- lifecycle API

    def start(self) -> AgentBase:
        with self._lifecycle_lock:
            self.on_start()
            self._state = AgentState.STARTED
            self._started_at = time.time()
            self._last_heartbeat = time.time()
        return self

    def pause(self) -> AgentBase:
        with self._lifecycle_lock:
            self.on_pause()
            self._state = AgentState.PAUSED
        return self

    def resume(self) -> AgentBase:
        with self._lifecycle_lock:
            self.on_resume()
            self._state = AgentState.STARTED
            self._last_heartbeat = time.time()
        return self

    def stop(self) -> AgentBase:
        with self._lifecycle_lock:
            self.on_stop()
            self._state = AgentState.STOPPED
        return self

    def mark_failed(self) -> AgentBase:
        with self._lifecycle_lock:
            self._state = AgentState.FAILED
        return self

    def ready(self) -> bool:
        """True when the agent can accept and process requests."""
        return self._state in (AgentState.STARTED, AgentState.CREATED)

    # -------------------------------------------------------------- dispatch

    def handle(self, request: AgentRequest) -> AgentResult:
        if self._state == AgentState.FAILED:
            return AgentResult(
                role=self.role,
                action=request.action,
                ok=False,
                message=f"agent {self.role!r} is in FAILED state",
                task_id=request.task_id,
            )
        self._request_count += 1
        try:
            result = super().handle(request)
        except Exception as exc:
            self.on_error(exc)
            return AgentResult(
                role=self.role,
                action=request.action,
                ok=False,
                message=f"agent {self.role!r} raised {type(exc).__name__}: {exc}",
                task_id=request.task_id,
            )
        if not result.ok:
            self._error_count += 1
        return result

    # ------------------------------------------------------------- discovery

    def structured_capabilities(self) -> list[Capability]:
        """Return structured capabilities (strings are wrapped minimally)."""
        resolved: list[Capability] = []
        for entry in self.capabilities:
            if isinstance(entry, Capability):
                resolved.append(entry)
            else:
                resolved.append(Capability(name=str(entry), description=self.role))
        return resolved

    def capability_names(self) -> list[str]:
        return [cap.name for cap in self.structured_capabilities()]

    def has_capability(self, name: str) -> bool:
        return name in self.capability_names()

    def describe(self) -> dict[str, Any]:
        data = super().describe()
        data.update(
            {
                "version": str(self.version_obj),
                "state": self.state.value,
                "capabilities": [cap.to_dict() for cap in self.structured_capabilities()],
                "metadata": self.metadata.to_dict(),
                "request_count": self._request_count,
                "error_count": self._error_count,
            }
        )
        return data

    def health(self) -> dict[str, Any]:
        """Health snapshot used by :class:`~cadgenesis.agents.health.AgentHealthMonitor`."""
        return {
            "role": self.role,
            "state": self.state.value,
            "alive": self.is_alive(),
            "last_heartbeat": self._last_heartbeat,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": (self._error_count / self._request_count if self._request_count else 0.0),
        }


class AgentLifecycleManager:
    """Central lifecycle coordinator for a fleet of :class:`AgentBase` agents.

    Tracks every agent's lifecycle state, provides ``start_all`` / ``stop_all``
    and reports a fleet-wide lifecycle snapshot.  Works with plain ``Agent``
    instances too (treated as permanently ready).
    """

    def __init__(self) -> None:
        self._states: dict[str, AgentState] = {}
        self._lock = threading.Lock()

    def register(self, agent: Agent) -> None:
        with self._lock:
            if isinstance(agent, AgentBase):
                self._states[agent.role] = agent.state
            else:
                self._states[agent.role] = AgentState.STARTED

    def unregister(self, role: str) -> bool:
        with self._lock:
            return self._states.pop(role, None) is not None

    def state_of(self, role: str) -> AgentState | None:
        with self._lock:
            return self._states.get(role)

    def start_all(self, agents: list[Agent]) -> list[str]:
        started: list[str] = []
        for agent in agents:
            if isinstance(agent, AgentBase):
                agent.start()
            started.append(agent.role)
            with self._lock:
                self._states[agent.role] = (
                    agent.state if isinstance(agent, AgentBase) else AgentState.STARTED
                )
        return started

    def stop_all(self, agents: list[Agent]) -> list[str]:
        stopped: list[str] = []
        for agent in agents:
            if isinstance(agent, AgentBase):
                agent.stop()
            stopped.append(agent.role)
            with self._lock:
                self._states[agent.role] = (
                    agent.state if isinstance(agent, AgentBase) else AgentState.STOPPED
                )
        return stopped

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {role: {"state": state.value} for role, state in self._states.items()}
