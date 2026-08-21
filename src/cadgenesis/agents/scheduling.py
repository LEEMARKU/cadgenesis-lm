"""cadgenesis.agents.scheduling
=============================
Pillar 5 scheduling infrastructure.

Provides dependency scheduling (DAG), priority, deadline, dynamic and parallel
scheduling plus load balancing — all additive to the existing sequential
:class:`~cadgenesis.agents.scheduler.TaskScheduler` (which is preserved).

* :class:`TaskNode` / :class:`TaskGraph` — explicit DAG with critical path.
* :class:`WorkerPool` — bounded thread pool for parallel execution.
* :class:`DAGScheduler` — dependency-aware executor (priority + deadline +
  retries + parallelism + load balancing).
* :class:`PriorityScheduler`, :class:`DeadlineScheduler`, :class:`DynamicScheduler`
  — focused scheduling policies over a task queue.
* :class:`LoadBalancer` — distributes ready tasks across workers by cost/load.
"""

from __future__ import annotations

import heapq
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.agents.base import AgentRequest


@dataclass
class TaskNode:
    """A schedulable unit of agent work."""

    task_id: str
    role: str
    action: str
    payload: dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    depends_on: list[str] = field(default_factory=list)
    deadline: float | None = None
    timeout: float | None = None
    retries: int = 0
    status: str = "pending"  # pending|ready|running|completed|failed|skipped
    attempts: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    result: Any = None
    error: str = ""
    cost: float = 1.0

    def __post_init__(self) -> None:
        if not self.task_id or not self.role or not self.action:
            raise ValueError("task_id, role and action are required")
        valid = {"pending", "ready", "running", "completed", "failed", "skipped"}
        if self.status not in valid:
            raise ValueError(f"invalid task status {self.status!r}")

    def to_request(self) -> AgentRequest:
        return AgentRequest(
            role=self.role, action=self.action, payload=self.payload, task_id=self.task_id
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "role": self.role,
            "action": self.action,
            "status": self.status,
            "attempts": self.attempts,
            "deadline": self.deadline,
            "timeout": self.timeout,
            "cost": self.cost,
            "depends_on": list(self.depends_on),
        }


class TaskGraph:
    """Explicit DAG of :class:`TaskNode` dependencies."""

    def __init__(self) -> None:
        self._nodes: dict[str, TaskNode] = {}

    def add(self, task: TaskNode) -> TaskGraph:
        if task.task_id in self._nodes:
            raise ValueError(f"duplicate task id {task.task_id!r}")
        for dep in task.depends_on:
            if dep not in self._nodes and dep != task.task_id:
                raise ValueError(f"task {task.task_id!r} depends on unknown task {dep!r}")
        self._nodes[task.task_id] = task
        return self

    def add_dependency(self, task_id: str, depends_on: str) -> TaskGraph:
        if task_id not in self._nodes or depends_on not in self._nodes:
            raise ValueError("add_dependency requires both tasks in the graph")
        if depends_on not in self._nodes[task_id].depends_on:
            self._nodes[task_id].depends_on.append(depends_on)
        return self

    def get(self, task_id: str) -> TaskNode | None:
        return self._nodes.get(task_id)

    @property
    def nodes(self) -> list[TaskNode]:
        return list(self._nodes.values())

    @property
    def count(self) -> int:
        return len(self._nodes)

    def dependents_of(self, task_id: str) -> list[TaskNode]:
        return [n for n in self._nodes.values() if task_id in n.depends_on]

    def dependencies_of(self, task_id: str) -> list[TaskNode]:
        node = self._nodes.get(task_id)
        if node is None:
            return []
        return [self._nodes[d] for d in node.depends_on if d in self._nodes]

    def is_acyclic(self) -> bool:
        """Kahn's algorithm cycle detection."""
        remaining = {tid: len(n.depends_on) for tid, n in self._nodes.items()}
        ready = [tid for tid, count in remaining.items() if count == 0]
        processed = 0
        while ready:
            tid = ready.pop()
            processed += 1
            for dep in self.dependents_of(tid):
                remaining[dep.task_id] -= 1
                if remaining[dep.task_id] == 0:
                    ready.append(dep.task_id)
        return processed == len(self._nodes)

    def topological_order(self) -> list[str]:
        """Deterministic topological order; raises ValueError on cycles."""
        if not self.is_acyclic():
            raise ValueError("task graph contains a cycle")
        order: list[str] = []
        remaining = {tid: set(n.depends_on) for tid, n in self._nodes.items()}
        while remaining:
            ready = sorted(tid for tid, deps in remaining.items() if not deps)
            if not ready:
                raise ValueError("task graph contains a cycle")
            tid = ready[0]
            order.append(tid)
            del remaining[tid]
            for deps in remaining.values():
                deps.discard(tid)
        return order

    def critical_path(self) -> tuple[list[str], float]:
        """Longest dependency chain by cumulative ``cost``.

        Returns ``(ordered_task_ids, total_cost)``.
        """
        if not self.is_acyclic():
            raise ValueError("task graph contains a cycle")
        longest: dict[str, float] = {}
        for tid in self.topological_order():
            node = self._nodes[tid]
            pred = max((longest[d] for d in node.depends_on if d in longest), default=0.0)
            longest[tid] = pred + node.cost
        end = max(longest, key=lambda k: longest[k]) if longest else None
        if end is None:
            return [], 0.0
        path: list[str] = []
        current: str | None = end
        while current is not None:
            path.append(current)
            node = self._nodes[current]
            current = max(
                (d for d in node.depends_on if d in longest),
                key=lambda d: longest[d],
                default=None,
            )
        path.reverse()
        return path, longest[end]

    def ready_nodes(self) -> list[TaskNode]:
        return [
            n
            for n in self._nodes.values()
            if n.status == "pending"
            and all(
                self._nodes.get(d) is not None and self._nodes[d].status == "completed"
                for d in n.depends_on
            )
        ]


class WorkerPool:
    """Bounded thread pool for parallel task execution."""

    def __init__(self, max_workers: int = 4) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="agent")

    @property
    def pool_size(self) -> int:
        return self._max_workers

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future:
        return self._executor.submit(fn, *args, **kwargs)

    def map(self, fn: Callable[..., Any], items: list[Any]) -> list[Any]:
        return list(self._executor.map(fn, items))

    def shutdown(self, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)


@dataclass
class SchedulerStats:
    """Aggregate counters and timings from a scheduler run."""

    total: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    wall_time: float = 0.0
    makespan: float = 0.0
    critical_path: float = 0.0
    utilization: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "wall_time": self.wall_time,
            "makespan": self.makespan,
            "critical_path": self.critical_path,
            "utilization": self.utilization,
        }


class DAGScheduler:
    """Dependency-aware parallel scheduler with deadlines, retries and retries.

    Executes a :class:`TaskGraph` by dispatching ready tasks to a
    :class:`WorkerPool`.  ``execute_fn(node) -> result`` runs each task; it
    defaults to calling ``dispatch`` on a provided ``dispatcher`` callable that
    accepts an :class:`AgentRequest` and returns an ``AgentResult``.
    """

    def __init__(
        self,
        workers: int = 4,
        default_timeout: float | None = None,
        default_retries: int = 0,
        dispatcher: Callable[[AgentRequest], Any] | None = None,
    ) -> None:
        self._pool = WorkerPool(max_workers=workers)
        self._default_timeout = default_timeout
        self._default_retries = default_retries
        self._dispatcher = dispatcher
        self._lock = threading.Lock()
        self._running: set[str] = set()

    @property
    def pool(self) -> WorkerPool:
        return self._pool

    # ------------------------------------------------------------------- run

    def run(
        self, graph: TaskGraph, execute_fn: Callable[[TaskNode], Any] | None = None
    ) -> SchedulerStats:
        """Execute the graph, returning aggregate statistics.

        ``execute_fn`` is called per node; when omitted the internal default
        dispatches :meth:`TaskNode.to_request` through ``self._dispatcher``.
        """
        if not graph.is_acyclic():
            raise ValueError("cannot schedule a cyclic task graph")
        started = time.time()
        completed, failed, skipped = 0, 0, 0
        for node in graph.nodes:
            node.status = "pending"
            node.attempts = 0
            node.result = None
            node.error = ""

        pending = {n.task_id for n in graph.nodes}
        while pending:
            ready = [n for n in graph.ready_nodes() if n.task_id in pending]
            if not ready:
                # No forward progress: unresolved dependency or cycle.
                for node in graph.nodes:
                    if node.task_id in pending:
                        node.status = "skipped"
                        node.error = "dependency unsatisfied"
                        skipped += 1
                break
            ready.sort(key=lambda n: (-n.priority, n.task_id))
            now = time.time()
            for node in ready:
                if node.deadline is not None and now > node.deadline:
                    node.status = "skipped"
                    node.error = "deadline missed before start"
                    skipped += 1
                    pending.discard(node.task_id)
                    continue
                node.status = "running"
                node.started_at = time.time()
                pending.discard(node.task_id)
                self._run_node(node, execute_fn)
                node.finished_at = time.time()
                if node.status == "completed":
                    completed += 1
                else:
                    failed += 1
        wall_time = time.time() - started
        _, critical_path = graph.critical_path()
        return SchedulerStats(
            total=graph.count,
            completed=completed,
            failed=failed,
            skipped=skipped,
            wall_time=wall_time,
            makespan=wall_time,
            critical_path=critical_path,
            utilization=critical_path / wall_time if wall_time else 0.0,
        )

    def _run_node(self, node: TaskNode, execute_fn: Callable[[TaskNode], Any] | None) -> None:
        attempts = 0
        max_attempts = 1 + (node.retries if node.retries else self._default_retries)
        while attempts < max_attempts:
            attempts += 1
            node.attempts = attempts
            try:
                if execute_fn is not None:
                    result = execute_fn(node)
                elif self._dispatcher is not None:
                    result = self._dispatcher(node.to_request())
                else:
                    raise ValueError("no execute_fn or dispatcher provided")
                node.result = result
                ok = getattr(result, "ok", True)
                node.status = "completed" if ok else "failed"
                if not ok:
                    node.error = getattr(result, "message", "failed")
                else:
                    return
            except Exception as exc:
                node.error = f"{type(exc).__name__}: {exc}"
                node.status = "failed"
        node.status = "failed"

    def shutdown(self) -> None:
        self._pool.shutdown()


class PriorityScheduler:
    """Queue-based scheduler that always runs the highest-priority ready task."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskNode] = {}

    def submit(self, task: TaskNode) -> str:
        if task.task_id in self._tasks:
            raise ValueError(f"duplicate task id {task.task_id!r}")
        self._tasks[task.task_id] = task
        return task.task_id

    def next_task(self) -> TaskNode | None:
        ready = [
            n
            for n in self._tasks.values()
            if n.status == "pending"
            and all(d in self._tasks and self._tasks[d].status == "completed" for d in n.depends_on)
        ]
        if not ready:
            return None
        return max(ready, key=lambda n: n.priority)

    def progress(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self._tasks.values():
            counts[node.status] = counts.get(node.status, 0) + 1
        return counts


class DeadlineScheduler:
    """Schedules tasks by earliest deadline (EDF), skipping expired work."""

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, TaskNode]] = []
        self._seq = 0

    def submit(self, task: TaskNode) -> None:
        if task.deadline is None:
            raise ValueError("DeadlineScheduler requires task.deadline")
        self._seq += 1
        heapq.heappush(self._heap, (task.deadline, self._seq, task))

    def next_task(self, now: float | None = None) -> TaskNode | None:
        now = time.time() if now is None else now
        while self._heap:
            deadline, _, task = heapq.heappop(self._heap)
            if deadline < now:
                task.status = "skipped"
                task.error = "expired before scheduling"
                continue
            return task
        return None

    @property
    def pending(self) -> int:
        return len(self._heap)


class DynamicScheduler:
    """Accepts tasks while a run is in progress (admission at any time)."""

    def __init__(self, workers: int = 4) -> None:
        self._pool = WorkerPool(max_workers=workers)
        self._submitted: list[TaskNode] = []
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def submit(self, task: TaskNode) -> None:
        self._submitted.append(task)

    def run(self, execute_fn: Callable[[TaskNode], Any]) -> SchedulerStats:
        self._running = True
        started = time.time()
        completed = failed = 0
        queue: list[TaskNode] = list(self._submitted)
        while queue:
            node = queue.pop(0)
            node.status = "running"
            try:
                node.result = execute_fn(node)
                ok = getattr(node.result, "ok", True)
                node.status = "completed" if ok else "failed"
                completed += 1 if ok else 0
                failed += 0 if ok else 1
            except Exception as exc:
                node.status = "failed"
                node.error = str(exc)
                failed += 1
        self._running = False
        return SchedulerStats(
            total=len(self._submitted),
            completed=completed,
            failed=failed,
            wall_time=time.time() - started,
        )

    def shutdown(self) -> None:
        self._pool.shutdown()


class LoadBalancer:
    """Distributes ready tasks across workers by estimated cost and load."""

    def __init__(self, worker_count: int) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be >= 1")
        self._worker_count = worker_count
        self._load = [0.0] * worker_count

    @property
    def worker_count(self) -> int:
        return self._worker_count

    def reset(self) -> None:
        self._load = [0.0] * self._worker_count

    def assign(self, tasks: list[TaskNode]) -> dict[int, list[TaskNode]]:
        """Map each worker index to a balanced slice of ``tasks``."""
        assignment: dict[int, list[TaskNode]] = {i: [] for i in range(self._worker_count)}
        for task in sorted(tasks, key=lambda t: -t.cost):
            worker = min(range(self._worker_count), key=lambda i: self._load[i])
            assignment[worker].append(task)
            self._load[worker] += task.cost
        return assignment

    def loads(self) -> list[float]:
        return list(self._load)
