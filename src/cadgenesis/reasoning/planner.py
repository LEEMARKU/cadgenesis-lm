"""cadgenesis.reasoning.planner
==============================
CAD workflow planning: structured plans with dependency ordering, cycle
detection, topological execution order, and template-driven plan creation.

A plan is a sequence of typed steps (``model``, ``constrain``, ``simulate``,
``validate``, …) with explicit ``depends_on`` edges.  :class:`TaskPlanner`
builds plans from named workflow templates and can refine an existing plan by
inserting steps when a registered rule triggers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_VALID_ACTIONS = (
    "model",
    "sketch",
    "constrain",
    "assemble",
    "simulate",
    "validate",
    "manufacture",
    "export",
)


@dataclass
class PlanningStep:
    """A single step in a CAD workflow plan."""

    id: str
    action: str
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("step id must be non-empty")
        if self.action not in _VALID_ACTIONS:
            raise ValueError(f"invalid action {self.action!r}; expected one of {_VALID_ACTIONS}")


@dataclass
class CADPlan:
    """An ordered collection of steps with dependency edges."""

    goal: str
    steps: list[PlanningStep] = field(default_factory=list)

    def add_step(self, step: PlanningStep) -> None:
        if any(s.id == step.id for s in self.steps):
            raise ValueError(f"step {step.id!r} already exists")
        self.steps.append(step)

    def get_step(self, step_id: str) -> PlanningStep | None:
        return next((s for s in self.steps if s.id == step_id), None)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def depends_on(self, step_id: str) -> list[str]:
        step = self.get_step(step_id)
        return list(step.depends_on) if step else []

    def is_cyclic(self) -> bool:
        """True if the dependency graph contains a cycle.

        Unknown dependencies are skipped, so a single broken edge does not
        count as a cycle (use :meth:`validate` for the full problem list).
        """
        edges: list[tuple[str, str]] = [
            (dep, step.id)
            for step in self.steps
            for dep in step.depends_on
            if self.get_step(dep) is not None
        ]
        indegree: dict[str, int] = {s.id: 0 for s in self.steps}
        adjacency: dict[str, list[str]] = {s.id: [] for s in self.steps}
        for src, dst in edges:
            adjacency[src].append(dst)
            indegree[dst] += 1
        queue = [s.id for s in self.steps if indegree[s.id] == 0]
        processed = 0
        while queue:
            node = queue.pop(0)
            processed += 1
            for neighbor in adjacency[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
        return processed != len(self.steps)

    def topological_order(self) -> list[str]:
        """Steps ordered so every step follows its dependencies.

        Raises ``ValueError`` when the plan contains a cycle or an unknown
        dependency.
        """
        order = self._topological_order()
        if order is None:
            raise ValueError("plan contains a dependency cycle or unknown step")
        return order

    def _topological_order(self) -> list[str] | None:
        edges: list[tuple[str, str]] = [
            (dep, step.id) for step in self.steps for dep in step.depends_on
        ]
        indegree: dict[str, int] = {s.id: 0 for s in self.steps}
        adjacency: dict[str, list[str]] = {s.id: [] for s in self.steps}
        for src, dst in edges:
            if src not in adjacency or dst not in adjacency:
                raise ValueError(f"step dependency references an unknown step: {src} -> {dst}")
            adjacency[src].append(dst)
            indegree[dst] += 1
        queue = [s.id for s in self.steps if indegree[s.id] == 0]
        order: list[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in adjacency[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)
        if len(order) != len(self.steps):
            return None
        return order

    def validate(self) -> list[str]:
        """Return a list of plan problems (empty when the plan is sound)."""
        ids = {s.id for s in self.steps}
        problems: list[str] = []
        if not ids:
            problems.append("plan has no steps")
        problems.extend(
            f"step {step.id!r} depends on unknown {dep!r}"
            for step in self.steps
            for dep in step.depends_on
            if dep not in ids
        )
        if self.is_cyclic():
            problems.append("dependency cycle detected")
        return problems

    def critical_path(self) -> list[str]:
        """Longest dependency chain (by number of steps)."""
        order = self.topological_order()
        longest: dict[str, int] = {s.id: 1 for s in self.steps}
        predecessor: dict[str, str | None] = {s.id: None for s in self.steps}
        for step_id in order:
            step = self.get_step(step_id)
            if step is None:
                continue
            for dep in step.depends_on:
                if longest[dep] + 1 > longest[step_id]:
                    longest[step_id] = longest[dep] + 1
                    predecessor[step_id] = dep
        end = max(longest, key=lambda k: longest[k])
        reversed_path: list[str] = []
        current: str | None = end
        while current is not None:
            reversed_path.append(current)
            current = predecessor[current]
        reversed_path.reverse()
        return reversed_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [
                {
                    "id": s.id,
                    "action": s.action,
                    "description": s.description,
                    "depends_on": list(s.depends_on),
                    "params": dict(s.params),
                }
                for s in self.steps
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CADPlan:
        plan = cls(goal=data["goal"])
        for step_data in data.get("steps", []):
            plan.add_step(
                PlanningStep(
                    id=step_data["id"],
                    action=step_data["action"],
                    description=step_data.get("description", ""),
                    depends_on=list(step_data.get("depends_on", [])),
                    params=dict(step_data.get("params", {})),
                )
            )
        return plan


# Built-in workflow templates: goal keyword -> step factory.
def _box_template(goal: str) -> CADPlan:
    plan = CADPlan(goal=goal)
    plan.add_step(
        PlanningStep(
            "s1",
            "sketch",
            "Sketch the base rectangle",
            depends_on=[],
            params={"width": 0, "depth": 0},
        )
    )
    plan.add_step(
        PlanningStep(
            "s2", "model", "Extrude the base to height", depends_on=["s1"], params={"height": 0}
        )
    )
    plan.add_step(
        PlanningStep("s3", "constrain", "Add dimension constraints", depends_on=["s2"], params={})
    )
    plan.add_step(
        PlanningStep("s4", "validate", "Validate the solid", depends_on=["s3"], params={})
    )
    return plan


def _assembly_template(goal: str) -> CADPlan:
    plan = CADPlan(goal=goal)
    plan.add_step(PlanningStep("a1", "model", "Create the base part", depends_on=[], params={}))
    plan.add_step(PlanningStep("a2", "model", "Create the mating part", depends_on=[], params={}))
    plan.add_step(
        PlanningStep("a3", "assemble", "Mate the two parts", depends_on=["a1", "a2"], params={})
    )
    plan.add_step(
        PlanningStep(
            "a4", "simulate", "Run a motion/interference check", depends_on=["a3"], params={}
        )
    )
    return plan


_TEMPLATES: dict[str, Any] = {
    "box": _box_template,
    "bracket": _box_template,
    "enclosure": _box_template,
    "assembly": _assembly_template,
    "mechanism": _assembly_template,
}


class TaskPlanner:
    """Creates and refines CAD workflow plans."""

    def __init__(self, rules=None) -> None:
        self._rules = rules  # optional cadgenesis.reasoning.rule_engine.RuleEngine

    def create_plan(self, goal: str) -> CADPlan:
        """Build a plan from a workflow template.

        Unknown goals fall back to the box template (the most common CAD
        workflow), guaranteeing the planner always produces a usable plan.
        """
        if not goal or not isinstance(goal, str):
            raise ValueError("goal must be a non-empty string")
        lowered = goal.lower()
        template = _TEMPLATES.get(lowered, _box_template)
        return template(lowered)

    def refine(self, plan: CADPlan, context: dict[str, Any]) -> CADPlan:
        """Insert or adjust steps using registered refinement rules.

        Rules must live in a ``RuleEngine`` and may append steps to
        ``context["plan"]``.  Returns the (possibly unchanged) plan.
        """
        if self._rules is None:
            return plan
        context = dict(context)
        context["plan"] = plan
        self._rules.run(context)
        refined = context.get("plan")
        if isinstance(refined, CADPlan):
            return refined
        return plan

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CADPlan:
        return CADPlan.from_dict(data)


__all__ = [
    "CADPlan",
    "PlanningStep",
    "TaskPlanner",
]
