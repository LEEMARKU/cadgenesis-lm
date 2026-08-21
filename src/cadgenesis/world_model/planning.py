"""cadgenesis.world_model.planning
=================================
World-model planning and execution (Pillar 4).

:class:`WorldModelPlanner` turns a goal into a plan (reusing
``cadgenesis.reasoning.planner.TaskPlanner``), expands each plan step into
concrete world-model operations and can *execute* those operations against an
:class:`~cadgenesis.world_model.objects.ObjectGraph`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.reasoning.planner import CADPlan, TaskPlanner
from cadgenesis.world_model.objects import ObjectGraph, WorldObject, make_object


@dataclass
class WorldStep:
    """A concrete world-model operation derived from a plan step."""

    id: str
    op: str  # create | position | assemble | validate | check_mechanical | simulate | finish
    target: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "op": self.op,
            "target": self.target,
            "params": dict(self.params),
        }


@dataclass
class StepOutcome:
    """Result of executing one world step."""

    step_id: str
    op: str
    passed: bool
    details: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "op": self.op,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass
class ExecutionResult:
    """Aggregated outcome of plan execution."""

    outcomes: list[StepOutcome] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(o.passed for o in self.outcomes)

    def summary(self) -> dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "outcomes": [o.summary() for o in self.outcomes],
        }


_OP_MAP = {
    "model": "create",
    "sketch": "create",
    "constrain": "assemble",
    "assemble": "assemble",
    "simulate": "simulate",
    "validate": "validate",
    "manufacture": "check_mechanical",
    "export": "finish",
}


class WorldModelPlanner:
    """Create, expand and execute plans over the world model.

    Optionally wired to the CAD execution engine (Pillar 8): when an engine
    is provided, ``validate`` / ``simulate`` / ``check_mechanical`` steps run
    real analytic checks instead of no-ops.
    """

    def __init__(
        self,
        planner: TaskPlanner | None = None,
        execution: Any = None,
    ) -> None:
        self.planner = planner or TaskPlanner()
        self.execution = execution

    def _ensure_execution(self) -> Any:
        if self.execution is None:
            from cadgenesis.execution.execution_engine import CADExecutionEngine

            self.execution = CADExecutionEngine()
        return self.execution

    def plan(self, goal: str) -> CADPlan:
        """Plan a workflow for ``goal``."""
        return self.planner.create_plan(goal)

    def expand(self, plan: CADPlan) -> list[WorldStep]:
        """Map every plan step to a concrete world-model operation."""
        steps: list[WorldStep] = []
        for step in plan.steps:
            op = _OP_MAP.get(step.action, "create")
            target = str(step.params.get("name", step.id))
            params = dict(step.params)
            steps.append(WorldStep(id=step.id, op=op, target=target, params=params))
        return steps

    def validate_plan(self, plan: CADPlan) -> list[str]:
        """Return problems (unknown deps / cycles / unsupported actions)."""
        problems = list(plan.validate())
        problems.extend(
            f"step {s.id!r} uses unsupported action {s.action!r}"
            for s in plan.steps
            if s.action not in _OP_MAP
        )
        return problems

    # --------------------------------------------------------------- execute

    def execute(
        self,
        plan: CADPlan,
        graph: ObjectGraph,
        material: str = "steel",
    ) -> ExecutionResult:
        """Execute the plan against ``graph``, mutating it as steps run."""
        problems = self.validate_plan(plan)
        if problems:
            return ExecutionResult([StepOutcome("", "plan", False, "; ".join(problems))])
        world_steps = self.expand(plan)
        by_id = {s.id: s for s in world_steps}
        outcome_map: dict[str, StepOutcome] = {}
        created: dict[str, WorldObject] = {}
        for step_id in plan.topological_order():
            step = by_id[step_id]
            outcome = self._run_step(step, graph, created, material)
            outcome_map[step_id] = outcome
        return ExecutionResult(list(outcome_map.values()))

    def _run_step(
        self,
        step: WorldStep,
        graph: ObjectGraph,
        created: dict[str, WorldObject],
        material: str,
    ) -> StepOutcome:
        name = step.target or step.id
        params = step.params
        if step.op == "create":
            feature = str(params.get("feature", "block"))
            obj = make_object(feature, name, dict(params.get("params", {})), material=material)
            graph.add(obj)
            created[name] = obj
            created[step.id] = obj
            return StepOutcome(step.id, step.op, True, f"created {feature} {name}")
        if step.op == "position":
            positioned = created.get(name) or created.get(step.id)
            if positioned is None:
                return StepOutcome(step.id, step.op, False, f"unknown part {name}")
            graph.set_pose(positioned.object_id, dict(params.get("pose", {})))
            return StepOutcome(step.id, step.op, True, f"positioned {name}")
        if step.op == "assemble":
            parent = created.get(str(params.get("parent", "")))
            child = created.get(str(params.get("child", ""))) or created.get(name)
            if parent is None or child is None:
                # no explicit parents -> relate the two most recently created
                created_names = list(created)
                if len(created_names) >= 2 and parent is None:
                    parent = created[created_names[-2]]
                if child is None and len(created_names) >= 1:
                    child = created[created_names[-1]]
            if parent is None or child is None or parent.object_id == child.object_id:
                return StepOutcome(step.id, step.op, False, "missing parent/child")
            graph.relate(parent.object_id, child.object_id, str(params.get("relation", "mounts")))
            return StepOutcome(step.id, step.op, True, f"related {child.name} -> {parent.name}")
        if step.op == "simulate":
            if self.execution is None:
                return StepOutcome(step.id, step.op, True, "simulation queued")
            analysis = str(params.get("analysis_type", "structural"))
            load = dict(params.get("load", {})) or {"force_n": 1000.0}
            summary = self._ensure_execution().simulation.run(analysis, **load).summary()
            detail = f"{analysis}: {'passed' if summary['passed'] else 'failed'}"
            return StepOutcome(step.id, step.op, summary["passed"], detail)
        if step.op == "validate":
            if self.execution is None:
                return StepOutcome(step.id, step.op, True, "world consistent")
            target = created.get(name) or created.get(step.id)
            design = {"name": name}
            if target is not None:
                design.update(target.to_dict() if hasattr(target, "to_dict") else {})
                if hasattr(target, "mesh") and target.mesh is not None:
                    design["mesh"] = target.mesh
            result = self._ensure_execution().execute(design=design)
            if not result.is_valid_geometry:
                return StepOutcome(step.id, step.op, False, "; ".join(result.errors or ["invalid"]))
            return StepOutcome(step.id, step.op, True, "geometry valid")
        if step.op == "check_mechanical":
            if self.execution is None:
                return StepOutcome(step.id, step.op, True, "mechanical proxy ok")
            target = created.get(name) or created.get(step.id)
            design = {"name": name}
            if target is not None:
                design.update(target.to_dict() if hasattr(target, "to_dict") else {})
            result = self._ensure_execution().execute(design=design)
            if not result.is_manufacturable:
                return StepOutcome(step.id, step.op, False, "not manufacturable")
            return StepOutcome(step.id, step.op, True, "manufacturable")
        if step.op == "finish":
            return StepOutcome(step.id, step.op, True, f"finished {plan_goal(params)}")
        return StepOutcome(step.id, step.op, False, f"unhandled op {step.op}")


def plan_goal(params: dict[str, Any]) -> str:
    return str(params.get("goal", "design"))


__all__ = ["ExecutionResult", "StepOutcome", "WorldModelPlanner", "WorldStep"]
