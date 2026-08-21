"""cadgenesis.agents.orchestrator
===============================
Agent platform — the Pillar 5 primary orchestration layer.

:class:`AgentPlatform` composes the registry, event bus, scheduling, consensus,
layered shared memory, health monitoring and the task-planning pipeline into a
single facade that drives CAD workflows.  It is fully backward compatible with
the existing :class:`~cadgenesis.agents.coordinator.AgentCoordinator`, which
keeps working unchanged.
"""

from __future__ import annotations

import threading
from typing import Any

from cadgenesis.agents.base import Agent, AgentRequest, AgentResult
from cadgenesis.agents.consensus import AgentOpinion, ConsensusEngine
from cadgenesis.agents.event_bus import Event, EventBus
from cadgenesis.agents.fleet import build_fleet
from cadgenesis.agents.health import AgentHealthMonitor
from cadgenesis.agents.infrastructure import AgentBase, AgentLifecycleManager
from cadgenesis.agents.pipeline import TaskPlanningPipeline, _seed_payload
from cadgenesis.agents.registry import AgentRegistry
from cadgenesis.agents.scheduling import DAGScheduler
from cadgenesis.agents.shared_memory import LayeredSharedMemory


class AgentPlatform:
    """Facade over the complete multi-agent orchestration stack."""

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        event_bus: EventBus | None = None,
        scheduler: DAGScheduler | None = None,
        shared_memory: LayeredSharedMemory | None = None,
        consensus: ConsensusEngine | None = None,
        health_monitor: AgentHealthMonitor | None = None,
        lifecycle: AgentLifecycleManager | None = None,
    ) -> None:
        self.registry = registry or AgentRegistry()
        self.bus = event_bus or EventBus()
        self.scheduler = scheduler or DAGScheduler(workers=4)
        self.memory = shared_memory or LayeredSharedMemory()
        self.consensus = consensus or ConsensusEngine()
        self.health = health_monitor or AgentHealthMonitor()
        self.lifecycle = lifecycle or AgentLifecycleManager()
        self._dispatches: list[AgentResult] = []
        self._lock = threading.Lock()
        self.pipeline = TaskPlanningPipeline(self.registry, self.scheduler)

    # ------------------------------------------------------------------ fleet

    def register(self, agent: Agent) -> AgentPlatform:
        self.registry.register(agent)
        self.lifecycle.register(agent)
        self.health.register(agent)
        self.bus.broadcast("agents.registered", {"role": agent.role})
        return self

    def load_fleet(self, memory: Any = None, validator: Any = None) -> AgentPlatform:
        """Register all 18 built-in agents."""
        build_fleet(self.registry, memory=memory, validator=validator)
        for agent in self.registry.agents:
            self.lifecycle.register(agent)
            self.health.register(agent)
        self.lifecycle.start_all(self.registry.agents)
        return self

    # ----------------------------------------------------------------- dispatch

    def dispatch(
        self, role: str, action: str, payload: dict[str, Any] | None = None
    ) -> AgentResult:
        agent = self.registry.get(role)
        if agent is None:
            return AgentResult(role, action, False, message=f"unknown role {role!r}")
        request = AgentRequest(role=role, action=action, payload=payload or {})
        result = agent.handle(request)
        with self._lock:
            self._dispatches.append(result)
        self.bus.broadcast(
            "agents.dispatched",
            {"role": role, "action": action, "ok": result.ok},
            sender="platform",
        )
        return result

    def ask(self, question: str, options: list[Any] | None = None) -> dict[str, Any]:
        """Ask the fleet for consensus on a question.

        Every agent that can handle ``"validate"``/``"check"`` contributes an
        opinion read from its result output (``option`` / ``weight`` /
        ``confidence``) or a vote derived from ``options``.
        """
        engine = ConsensusEngine()
        for agent in self.registry.agents:
            if not (isinstance(agent, AgentBase) and agent.ready()):
                continue
            action = next(
                (a for a in ("validate", "check", "estimate") if a in agent.actions),
                None,
            )
            if action is None:
                continue
            result = agent.handle(
                AgentRequest(
                    role=agent.role,
                    action=action,
                    payload=_seed_payload(agent.role, action, question),
                )
            )
            if not result.ok:
                continue
            option = result.output.get("option", result.output.get("passed"))
            if option is None and result.output.get("summary") is None:
                option = "acceptable" if result.ok else "rejected"
            engine.record(
                AgentOpinion(
                    agent=agent.role,
                    option=option,
                    weight=result.output.get("weight", 1.0),
                    confidence=result.output.get("confidence", 1.0),
                )
            )
        return engine.full_summary()

    # ------------------------------------------------------------ coordination

    def share(self, region: str, key: str, value: Any) -> None:
        self.memory.set(region, key, value)

    def publish(self, topic: str, payload: dict[str, Any] | None = None) -> Event:
        return self.bus.broadcast(topic, payload, sender="platform")

    def submit_pipeline(self, goal: str, decompose: bool = False) -> dict[str, Any]:
        report = self.pipeline.run(goal, decompose=decompose)
        self.bus.broadcast("pipeline.completed", {"goal": goal, "ok": report.validation["passed"]})
        return report.to_dict()

    # ---------------------------------------------------------------- reporting

    def health_summary(self) -> dict[str, Any]:
        return self.health.summary(self.registry.agents)

    def fleet_snapshot(self) -> list[dict[str, Any]]:
        return self.registry.snapshot()

    def stats(self) -> dict[str, Any]:
        return {
            "agents": len(self.registry),
            "dispatches": len(self._dispatches),
            "stored_events": self.bus.store.size,
        }

    def shutdown(self) -> None:
        self.lifecycle.stop_all(self.registry.agents)
        self.scheduler.shutdown()
