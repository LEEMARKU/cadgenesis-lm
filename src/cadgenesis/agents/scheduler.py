"""cadgenesis.agents.scheduler
============================
Task scheduler for the multi-agent orchestration layer.

Agents submit tasks; the scheduler tracks status, runs dependency ordering
(topological sort over ``depends_on`` edges) and exposes the next batch of
ready tasks.  Supports priority ordering among ready tasks and basic
bookkeeping (completed / failed / progress).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_STATUSES = ("pending", "ready", "running", "completed", "failed")


@dataclass
class AgentTask:
    """A unit of work submitted to the scheduler."""

    task_id: str
    role: str
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if not self.role:
            raise ValueError("task role must be non-empty")
        if not self.action:
            raise ValueError("task action must be non-empty")
        if self.status not in _STATUSES:
            raise ValueError(f"invalid status {self.status!r}; expected one of {_STATUSES}")


class TaskScheduler:
    """Dependency-aware scheduler for agent tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, AgentTask] = {}

    # ---------------------------------------------------------------- submit

    def submit(self, task: AgentTask) -> str:
        """Add a task; returns its ``task_id``."""
        if task.task_id in self._tasks:
            raise ValueError(f"task {task.task_id!r} already submitted")
        self._tasks[task.task_id] = task
        return task.task_id

    def submit_many(self, tasks: list[AgentTask]) -> list[str]:
        return [self.submit(task) for task in tasks]

    # ----------------------------------------------------------------- reads

    def get(self, task_id: str) -> AgentTask | None:
        return self._tasks.get(task_id)

    @property
    def pending(self) -> list[AgentTask]:
        return [t for t in self._tasks.values() if t.status == "pending"]

    @property
    def ready(self) -> list[AgentTask]:
        return [t for t in self._tasks.values() if t.status == "ready"]

    @property
    def running(self) -> list[AgentTask]:
        return [t for t in self._tasks.values() if t.status == "running"]

    @property
    def completed(self) -> list[AgentTask]:
        return [t for t in self._tasks.values() if t.status == "completed"]

    @property
    def failed(self) -> list[AgentTask]:
        return [t for t in self._tasks.values() if t.status == "failed"]

    @property
    def all_tasks(self) -> list[AgentTask]:
        return list(self._tasks.values())

    # ------------------------------------------------------------ execution

    def _dependencies_met(self, task: AgentTask) -> bool:
        for dep in task.depends_on:
            dep_task = self._tasks.get(dep)
            if dep_task is None or dep_task.status != "completed":
                return False
        return True

    def update_status(self, task_id: str, status: str) -> None:
        """Move a task between statuses (``pending``/``ready``/``running``)."""
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"unknown task {task_id!r}")
        if status not in _STATUSES:
            raise ValueError(f"invalid status {status!r}; expected one of {_STATUSES}")
        task.status = status

    def mark_ready(self, task_id: str) -> None:
        self.update_status(task_id, "ready")

    def mark_running(self, task_id: str) -> None:
        self.update_status(task_id, "running")

    def mark_completed(self, task_id: str) -> None:
        self.update_status(task_id, "completed")

    def mark_failed(self, task_id: str) -> None:
        self.update_status(task_id, "failed")

    def next_tasks(self, max_tasks: int | None = None) -> list[AgentTask]:
        """Tasks ready to run now, ordered by priority (highest first)."""
        ready = [t for t in self.pending if self._dependencies_met(t)]
        ready.sort(key=lambda t: t.priority, reverse=True)
        if max_tasks is not None:
            ready = ready[: max(max_tasks, 0)]
        return ready

    def step(self) -> list[AgentTask]:
        """Promote the next dependency-ready batch from pending → ready."""
        promoted = [t for t in self.pending if self._dependencies_met(t)]
        for task in promoted:
            task.status = "ready"
        return promoted

    def progress(self) -> dict[str, int]:
        """Summary counters for telemetry."""
        return {
            "total": len(self._tasks),
            "pending": len(self.pending),
            "ready": len(self.ready),
            "running": len(self.running),
            "completed": len(self.completed),
            "failed": len(self.failed),
        }

    def has_cycles(self) -> bool:
        """True when the dependency graph contains a cycle."""
        edges: list[tuple[str, str]] = []
        edges.extend(
            (dep, task.task_id)
            for task in self._tasks.values()
            for dep in task.depends_on
            if dep in self._tasks
        )
        indegree = {tid: 0 for tid in self._tasks}
        adjacency: dict[str, list[str]] = {tid: [] for tid in self._tasks}
        for src, dst in edges:
            adjacency[src].append(dst)
            indegree[dst] += 1
        queue = [tid for tid, deg in indegree.items() if deg == 0]
        processed = 0
        while queue:
            node = queue.pop(0)
            processed += 1
            for neighbor in adjacency[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
        return processed != len(self._tasks)
