"""cadgenesis.agents.pipeline
===========================
Pillar 5 task-planning pipeline.

Implements the canonical 8-stage workflow:

    user prompt -> intent analysis -> task graph -> task decomposition ->
    agent assignment -> execution -> monitoring -> validation -> result
    aggregation

Each stage is a small composable class; :class:`TaskPlanningPipeline.run`
wires them together and returns a :class:`PipelineReport`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from cadgenesis.agents.base import AgentResult
from cadgenesis.agents.registry import AgentRegistry
from cadgenesis.agents.scheduling import DAGScheduler, TaskGraph, TaskNode

# Intent keywords -> candidate roles (used by the heuristic intent analyser).
_INTENT_HINTS: dict[str, tuple[str, ...]] = {
    "plan": ("planner",),
    "geometry": ("geometry",),
    "constraint": ("constraint",),
    "assembly": ("assembly",),
    "manufacture": ("manufacturing",),
    "material": ("material",),
    "cost": ("cost",),
    "simulate": ("simulation",),
    "optimize": ("optimization",),
    "validate": ("validation",),
    "safety": ("safety",),
    "document": ("documentation",),
    "memor": ("memory",),
    "retriev": ("retrieval",),
    "learn": ("learning",),
    "monitor": ("monitoring",),
    "debug": ("debugging",),
}


@dataclass
class PipelineReport:
    """Output of a full pipeline run."""

    goal: str
    intent: dict[str, Any]
    tasks: list[dict[str, Any]]
    assignment: dict[str, str]
    results: list[AgentResult]
    validation: dict[str, Any]
    aggregated: dict[str, Any]
    stats: dict[str, Any]
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "intent": self.intent,
            "tasks": self.tasks,
            "assignment": self.assignment,
            "validation": self.validation,
            "aggregated": self.aggregated,
            "stats": self.stats,
            "duration_s": self.duration_s,
        }


class IntentAnalyser:
    """Heuristic intent extraction from a user prompt."""

    def analyse(self, goal: str) -> dict[str, Any]:
        lowered = goal.lower()
        roles: list[str] = []
        for hint, candidates in _INTENT_HINTS.items():
            if hint in lowered:
                roles.extend(candidates)
        roles = list(dict.fromkeys(roles))
        return {
            "goal": goal,
            "detected_roles": roles,
            "complexity": "high" if len(roles) > 3 else "low",
        }


class TaskGraphBuilder:
    """Builds a DAG of tasks from an intent description."""

    def build(self, intent: dict[str, Any], goal: str, prefix: str = "task") -> TaskGraph:
        graph = TaskGraph()
        roles = intent.get("detected_roles") or ["planner"]
        if "planner" not in roles:
            roles = ["planner", *roles]
        previous: str | None = None
        for index, role in enumerate(roles):
            task_id = f"{prefix}:{index}:{role}"
            node = TaskNode(
                task_id=task_id,
                role=role,
                action=_default_action(role),
                payload=_seed_payload(role, _default_action(role), goal),
                depends_on=[previous] if previous else [],
                priority=len(roles) - index,
            )
            graph.add(node)
            previous = task_id
        return graph


def _default_action(role: str) -> str:
    mapping = {
        "planner": "create_plan",
        "geometry": "validate",
        "constraint": "check",
        "assembly": "check_clearance",
        "manufacturing": "assess",
        "optimization": "optimize",
        "simulation": "check_safety",
        "validation": "validate",
        "material": "lookup",
        "cost": "estimate",
        "documentation": "summarize",
        "safety": "check",
        "memory": "recall",
        "retrieval": "retrieve",
        "user": "ask",
        "learning": "recall_lessons",
        "monitoring": "health",
        "debugging": "inspect",
    }
    return mapping.get(role, "validate")


def _seed_payload(role: str, action: str, goal: str) -> dict[str, Any]:
    """Return a valid default payload for a role/action pair."""
    part = {
        "kind": "box",
        "dims": {"length": 10.0, "width": 5.0, "height": 2.0},
        "position": [0.0, 0.0, 0.0],
        "name": "part",
        "processes": ["machining"],
    }
    primitive = {
        "kind": "box",
        "dims": {"length": 10.0, "width": 5.0, "height": 2.0},
        "position": [0.0, 0.0, 0.0],
        "name": "part",
    }
    templates: dict[str, dict[str, Any]] = {
        "planner": {"goal": goal},
        "geometry": {**primitive},
        "constraint": {
            "variables": [{"name": "L", "initial": 5.0, "lower": 1.0, "upper": 10.0}],
            "constraints": [{"name": "c0", "terms": {"L": 1.0}, "operator": "<=", "rhs": 8.0}],
        },
        "assembly": {
            "a": primitive,
            "b": {**primitive, "position": [20.0, 0.0, 0.0]},
            "gap": 1.0,
        },
        "manufacturing": {"part": part},
        "optimization": {"params": {"current": 1.0, "target": 1.0}, "objective": "mass"},
        "simulation": {"safety_factor": 2.0, "required_safety_factor": 1.5},
        "validation": {"context": {}},
        "material": {"material": "Al 6061-T6"},
        "cost": {"mass_kg": 1.5, "material": "Al 6061-T6", "quantity": 100},
        "documentation": {"topic": goal},
        "safety": {"context": {}},
        "memory": {"key": "session:note"},
        "retrieval": {"query": goal},
        "user": {"question": goal},
        "learning": {"query": goal},
        "monitoring": {"agents": []},
        "debugging": {"results": []},
    }
    return templates.get(role, {"goal": goal})


class TaskDecomposer:
    """Decomposes high-complexity tasks into sub-graphs (recursive splitting)."""

    def decompose(self, graph: TaskGraph, max_depth: int = 2) -> TaskGraph:
        """Expand every leaf node once when it is a planning/geometry milestone."""
        if max_depth <= 0:
            return graph
        leaves = [n for n in graph.nodes if not graph.dependents_of(n.task_id)]
        for leaf in leaves:
            if leaf.role not in ("planner", "geometry"):
                continue
            sub = TaskGraphBuilder().build(
                {"detected_roles": ["planner", "validation"]},
                str(leaf.payload.get("goal", "")),
                prefix=f"{leaf.task_id}:sub",
            )
            for node in sub.nodes:
                node.depends_on.append(leaf.task_id)
                graph.add(node)
        return graph


class AgentAssigner:
    """Maps each task to the concrete role that will execute it."""

    def assign(self, graph: TaskGraph, registry: AgentRegistry) -> dict[str, str]:
        assignment: dict[str, str] = {}
        for node in graph.nodes:
            if registry.get(node.role) is not None:
                assignment[node.task_id] = node.role
                continue
            candidates = registry.find_by_action(node.action)
            if candidates:
                assignment[node.task_id] = candidates[0].role
            else:
                assignment[node.task_id] = node.role  # unresolved, reported
        return assignment


class TaskValidator:
    """Validation gate over execution results."""

    def validate(self, results: list[AgentResult]) -> dict[str, Any]:
        passed = [r for r in results if r.ok]
        failed = [r for r in results if not r.ok]
        return {
            "passed": len(results) == len(passed),
            "passed_count": len(passed),
            "failed_count": len(failed),
            "failed": [{"role": r.role, "action": r.action, "message": r.message} for r in failed],
        }


class ResultAggregator:
    """Combines per-task results into a single structured answer."""

    def aggregate(self, results: list[AgentResult]) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for result in results:
            if result.ok:
                outputs[f"{result.role}:{result.action}"] = result.output
        return {
            "ok": len([r for r in results if r.ok]),
            "outputs": outputs,
            "summary": " ".join(r.message for r in results if r.ok),
        }


class TaskPlanningPipeline:
    """The 8-stage Pillar 5 pipeline, composed from the stage classes above."""

    def __init__(
        self,
        registry: AgentRegistry,
        scheduler: DAGScheduler | None = None,
        analyser: IntentAnalyser | None = None,
        builder: TaskGraphBuilder | None = None,
        decomposer: TaskDecomposer | None = None,
        assigner: AgentAssigner | None = None,
        validator: TaskValidator | None = None,
        aggregator: ResultAggregator | None = None,
    ) -> None:
        self.registry = registry
        self.scheduler = scheduler or DAGScheduler(workers=2)
        self.analyser = analyser or IntentAnalyser()
        self.builder = builder or TaskGraphBuilder()
        self.decomposer = decomposer or TaskDecomposer()
        self.assigner = assigner or AgentAssigner()
        self.validator = validator or TaskValidator()
        self.aggregator = aggregator or ResultAggregator()

    def run(self, goal: str, decompose: bool = False) -> PipelineReport:
        started = time.time()
        intent = self.analyser.analyse(goal)
        graph = self.builder.build(intent, goal)
        if decompose:
            graph = self.decomposer.decompose(graph)
        assignment = self.assigner.assign(graph, self.registry)

        def execute(node: TaskNode) -> AgentResult:
            agent = self.registry.get(assignment[node.task_id])
            if agent is None:
                return AgentResult(
                    role=node.role,
                    action=node.action,
                    ok=False,
                    message=f"no agent assigned for role {node.role!r}",
                    task_id=node.task_id,
                )
            return agent.handle(node.to_request())

        stats = self.scheduler.run(graph, execute_fn=execute)
        results: list[AgentResult] = [n.result for n in graph.nodes if n.result is not None]
        validation = self.validator.validate(results)
        aggregated = self.aggregator.aggregate(results)
        return PipelineReport(
            goal=goal,
            intent=intent,
            tasks=[n.to_dict() for n in graph.nodes],
            assignment=assignment,
            results=results,
            validation=validation,
            aggregated=aggregated,
            stats=stats.to_dict(),
            duration_s=time.time() - started,
        )

    def shutdown(self) -> None:
        self.scheduler.shutdown()
