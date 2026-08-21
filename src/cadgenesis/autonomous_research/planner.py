"""
Research Planner - Research objective planning, hypothesis generation, experiment
prioritization.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class ObjectivePriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ObjectiveStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResearchObjective:
    """A research objective to be pursued."""

    objective_id: str
    title: str
    description: str
    priority: ObjectivePriority
    success_criteria: list[str]
    estimated_effort: str  # e.g., "2 weeks", "1 month"
    dependencies: list[str] = field(default_factory=list)  # other objective_ids
    tags: list[str] = field(default_factory=list)
    status: ObjectiveStatus = ObjectiveStatus.PROPOSED
    created_at: float = field(default_factory=time.time)
    created_by: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchPlan:
    """A plan comprising multiple research objectives."""

    plan_id: str
    name: str
    description: str
    objectives: list[ResearchObjective] = field(default_factory=list)
    timeline: dict[str, Any] = field(
        default_factory=dict
    )  # objective_id -> {start, end, resources}
    resources: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    created_by: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_objective(self, objective: ResearchObjective) -> None:
        self.objectives.append(objective)

    def get_ordered_objectives(self) -> list[ResearchObjective]:
        """Return objectives in dependency order."""
        # Simple topological sort
        obj_map = {o.objective_id: o for o in self.objectives}
        visited = set()
        ordered = []

        def visit(obj_id: str) -> None:
            if obj_id in visited:
                return
            visited.add(obj_id)
            obj = obj_map.get(obj_id)
            if obj:
                for dep in obj.dependencies:
                    visit(dep)
                ordered.append(obj)

        for obj in self.objectives:
            visit(obj.objective_id)

        return ordered


class ResearchPlanner:
    """Plans research objectives and creates research plans."""

    def __init__(self):
        self._objectives: dict[str, ResearchObjective] = {}
        self._plans: dict[str, ResearchPlan] = {}
        self._lock = RLock()

    def create_objective(
        self,
        title: str,
        description: str,
        priority: ObjectivePriority = ObjectivePriority.MEDIUM,
        success_criteria: list[str] | None = None,
        estimated_effort: str = "1 week",
        dependencies: list[str] | None = None,
        tags: list[str] | None = None,
        created_by: str = "",
    ) -> ResearchObjective:
        objective = ResearchObjective(
            objective_id=str(uuid.uuid4()),
            title=title,
            description=description,
            priority=priority,
            success_criteria=success_criteria or [],
            estimated_effort=estimated_effort,
            dependencies=dependencies or [],
            tags=tags or [],
            created_by=created_by,
        )
        with self._lock:
            self._objectives[objective.objective_id] = objective
        return objective

    def get_objective(self, objective_id: str) -> ResearchObjective | None:
        with self._lock:
            return self._objectives.get(objective_id)

    def update_objective_status(self, objective_id: str, status: ObjectiveStatus) -> bool:
        with self._lock:
            obj = self._objectives.get(objective_id)
            if not obj:
                return False
            obj.status = status
            return True

    def create_plan(
        self,
        name: str,
        description: str,
        objective_ids: list[str],
        created_by: str = "",
    ) -> ResearchPlan:
        objectives = []
        for oid in objective_ids:
            obj = self._objectives.get(oid)
            if obj:
                objectives.append(obj)

        plan = ResearchPlan(
            plan_id=str(uuid.uuid4()),
            name=name,
            description=description,
            objectives=objectives,
            created_by=created_by,
        )
        with self._lock:
            self._plans[plan.plan_id] = plan
        return plan

    def get_plan(self, plan_id: str) -> ResearchPlan | None:
        with self._lock:
            return self._plans.get(plan_id)

    def list_objectives(
        self,
        status: ObjectiveStatus | None = None,
        priority: ObjectivePriority | None = None,
    ) -> list[ResearchObjective]:
        with self._lock:
            objectives = list(self._objectives.values())
            if status:
                objectives = [o for o in objectives if o.status == status]
            if priority:
                objectives = [o for o in objectives if o.priority == priority]
            return objectives

    def prioritize_objectives(self, objectives: list[ResearchObjective]) -> list[ResearchObjective]:
        """Prioritize objectives by priority, dependencies, and effort."""
        priority_order = {
            ObjectivePriority.CRITICAL: 0,
            ObjectivePriority.HIGH: 1,
            ObjectivePriority.MEDIUM: 2,
            ObjectivePriority.LOW: 3,
        }
        return sorted(objectives, key=lambda o: (priority_order[o.priority], o.estimated_effort))
