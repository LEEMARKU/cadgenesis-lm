"""cadgenesis.agents.health
=========================
Health monitoring for the agent fleet.

Agents publish heartbeats and expose a ``health()`` snapshot; the
:class:`AgentHealthMonitor` aggregates per-agent status, computes error rates
and produces a fleet-wide health report.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.agents.base import Agent
from cadgenesis.agents.infrastructure import AgentBase


@dataclass
class AgentHealthStatus:
    """Health snapshot for a single agent."""

    role: str
    ok: bool
    state: str = ""
    last_heartbeat: float | None = None
    alive: bool = False
    request_count: int = 0
    error_count: int = 0
    error_rate: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "ok": self.ok,
            "state": self.state,
            "last_heartbeat": self.last_heartbeat,
            "alive": self.alive,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "detail": self.detail,
        }


class AgentHealthMonitor:
    """Aggregates per-agent health across the fleet.

    ``max_error_rate`` (default 0.5) marks an agent as unhealthy once its
    error-to-request ratio exceeds the threshold.  ``timeout`` (default 30 s)
    controls heartbeat-based liveness.
    """

    def __init__(self, timeout: float = 30.0, max_error_rate: float = 0.5) -> None:
        if timeout <= 0 or not (0.0 <= max_error_rate <= 1.0):
            raise ValueError("timeout must be > 0 and max_error_rate in [0, 1]")
        self._timeout = timeout
        self._max_error_rate = max_error_rate
        self._heartbeats: dict[str, float] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ api

    def register(self, agent: Agent) -> None:
        if isinstance(agent, AgentBase):
            agent.heartbeat()
        with self._lock:
            self._heartbeats[agent.role] = time.time()

    def unregister(self, role: str) -> bool:
        with self._lock:
            return self._heartbeats.pop(role, None) is not None

    def heartbeat(self, agent: Agent) -> None:
        """Record a liveness tick from ``agent``."""
        if isinstance(agent, AgentBase):
            agent.heartbeat()
        with self._lock:
            self._heartbeats[agent.role] = time.time()

    def check(self, agent: Agent) -> AgentHealthStatus:
        """Compute the current health status of one agent."""
        raw: dict[str, Any] = {}
        if isinstance(agent, AgentBase):
            raw = agent.health()
        last = self._heartbeats.get(agent.role)
        alive = (
            (time.time() - last) <= self._timeout if last is not None else raw.get("alive", False)
        )
        errors = raw.get("error_count", 0)
        requests = raw.get("request_count", 0)
        error_rate = errors / requests if requests else 0.0
        state = raw.get("state", "")
        failed = state == "failed"
        ok = alive and not failed and error_rate <= self._max_error_rate
        return AgentHealthStatus(
            role=agent.role,
            ok=ok,
            state=state,
            last_heartbeat=last,
            alive=alive,
            request_count=requests,
            error_count=errors,
            error_rate=error_rate,
            detail=raw,
        )

    def check_all(self, agents: list[Agent]) -> list[AgentHealthStatus]:
        return [self.check(agent) for agent in agents]

    def summary(self, agents: list[Agent]) -> dict[str, Any]:
        statuses = self.check_all(agents)
        healthy = [s.role for s in statuses if s.ok]
        unhealthy = [s.role for s in statuses if not s.ok]
        return {
            "total": len(statuses),
            "healthy": healthy,
            "unhealthy": unhealthy,
            "healthy_count": len(healthy),
            "unhealthy_count": len(unhealthy),
            "generated_at": time.time(),
        }
