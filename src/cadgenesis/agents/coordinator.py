"""cadgenesis.agents.coordinator
===============================
Agent coordinator — the orchestrator of the multi-agent layer.

The coordinator owns an agent registry, a :class:`MessageBus`, a
:class:`SharedMemory` blackboard, a :class:`TaskScheduler` and a
:class:`ConsensusEngine`.  It routes :class:`AgentRequest` objects to the
right agent, runs dependency-ordered task batches, and aggregates opinions
into decisions.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import Agent, AgentRequest, AgentResult
from cadgenesis.agents.consensus import AgentOpinion, ConsensusEngine
from cadgenesis.agents.message_bus import AgentMessage, MessageBus
from cadgenesis.agents.scheduler import AgentTask, TaskScheduler
from cadgenesis.agents.shared_memory import SharedMemory


class AgentCoordinator:
    """Routes requests to agents and orchestrates their collaboration."""

    def __init__(
        self,
        agents: list[Agent] | None = None,
        bus: MessageBus | None = None,
        memory: SharedMemory | None = None,
        scheduler: TaskScheduler | None = None,
        consensus: ConsensusEngine | None = None,
    ) -> None:
        self.bus = bus or MessageBus()
        self.memory = memory or SharedMemory()
        self.scheduler = scheduler or TaskScheduler()
        self.consensus = consensus or ConsensusEngine()
        self._agents: dict[str, Agent] = {}
        if agents:
            for agent in agents:
                self.register(agent)

    # ------------------------------------------------------------- registry

    def register(self, agent: Agent) -> None:
        """Add an agent to the team (replaces an existing same-role agent)."""
        if not isinstance(agent, Agent):
            raise TypeError("registered object must be an Agent")
        self._agents[agent.role] = agent

    def unregister(self, role: str) -> bool:
        return self._agents.pop(role, None) is not None

    def agent(self, role: str) -> Agent | None:
        return self._agents.get(role)

    @property
    def roles(self) -> list[str]:
        return sorted(self._agents)

    def agents(self) -> list[Agent]:
        return list(self._agents.values())

    # -------------------------------------------------------------- dispatch

    def dispatch(self, request: AgentRequest) -> AgentResult:
        """Route a request to the agent matching its role."""
        agent = self._agents.get(request.role)
        if agent is None:
            return AgentResult(
                role=request.role,
                action=request.action,
                ok=False,
                message=f"no agent registered for role {request.role!r}",
                task_id=request.task_id,
            )
        return agent.handle(request)

    def dispatch_action(
        self,
        role: str,
        action: str,
        payload: dict[str, Any] | None = None,
        task_id: str = "",
    ) -> AgentResult:
        """Convenience wrapper building and dispatching a request."""
        return self.dispatch(
            AgentRequest(
                role=role,
                action=action,
                payload=payload or {},
                task_id=task_id,
            )
        )

    # --------------------------------------------------------- collaboration

    def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        sender: str = "coordinator",
        priority: int = 0,
    ) -> AgentMessage:
        return self.bus.publish(topic, payload, sender=sender, priority=priority)

    def share(self, key: str, value: Any) -> None:
        self.memory.set(key, value)

    def ask_consensus(
        self,
        role: str,
        action: str,
        payload: dict[str, Any],
        options: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Collect one opinion per registered agent and aggregate a decision.

        ``options`` is optional; when absent the agents' raw outputs are
        aggregated via the consensus engine.
        """
        opinions: list[AgentOpinion] = []
        for agent in self.agents():
            if not agent.can_handle(action):
                continue
            result = agent.handle(
                AgentRequest(role=agent.role, action=action, payload=dict(payload))
            )
            if not result.ok:
                continue
            choice = result.output.get("option", result.output)
            if options is not None and choice not in options:
                continue
            opinions.append(
                AgentOpinion(
                    agent=agent.role,
                    option=choice,
                    weight=float(result.output.get("weight", 1.0)),
                    confidence=float(result.output.get("confidence", 1.0)),
                )
            )
        engine = ConsensusEngine()
        engine.record_many(opinions)
        return engine.summary()

    # ------------------------------------------------------------ scheduling

    def submit(
        self,
        role: str,
        action: str,
        payload: dict[str, Any] | None = None,
        task_id: str | None = None,
        priority: int = 0,
        depends_on: list[str] | None = None,
    ) -> str:
        task = AgentTask(
            task_id=task_id or f"{role}:{action}:{len(self.scheduler.all_tasks)}",
            role=role,
            action=action,
            payload=payload or {},
            priority=priority,
            depends_on=depends_on or [],
        )
        return self.scheduler.submit(task)

    def run_batch(self, max_tasks: int | None = None) -> list[AgentResult]:
        """Run the next ready batch to completion and return their results."""
        tasks = self.scheduler.next_tasks(max_tasks=max_tasks)
        results: list[AgentResult] = []
        for task in tasks:
            self.scheduler.mark_running(task.task_id)
            result = self.dispatch_action(
                task.role,
                task.action,
                task.payload,
                task_id=task.task_id,
            )
            if result.ok:
                self.scheduler.mark_completed(task.task_id)
            else:
                self.scheduler.mark_failed(task.task_id)
            results.append(result)
        return results

    def run_all(self) -> list[AgentResult]:
        """Keep running batches until no ready tasks remain."""
        results: list[AgentResult] = []
        while True:
            batch = self.scheduler.next_tasks()
            if not batch:
                break
            results.extend(self.run_batch(max_tasks=len(batch)))
        return results

    # ---------------------------------------------------------------- report

    def summary(self) -> dict[str, Any]:
        return {
            "agents": [agent.describe() for agent in self.agents()],
            "scheduler": self.scheduler.progress(),
            "consensus": self.consensus.summary(),
        }
