"""cadgenesis.agents.planner
==========================
Specialised planner agent: turns a design goal into an executable CAD plan.
"""

from __future__ import annotations

from cadgenesis.agents.base import Agent, AgentRequest, AgentResult
from cadgenesis.reasoning.planner import TaskPlanner


class PlannerAgent(Agent):
    """Creates and refines CAD workflow plans."""

    role = "planner"
    actions = ("create_plan", "refine_plan")

    def __init__(self, planner: TaskPlanner | None = None) -> None:
        super().__init__()
        self.planner = planner or TaskPlanner()

    def process(self, request: AgentRequest) -> AgentResult:
        if request.action == "create_plan":
            goal = request.payload.get("goal", "")
            if not goal:
                return AgentResult(
                    role=self.role,
                    action=request.action,
                    ok=False,
                    message="create_plan requires a non-empty 'goal'",
                    task_id=request.task_id,
                )
            plan = self.planner.create_plan(goal)
            return AgentResult(
                role=self.role,
                action=request.action,
                ok=True,
                output={
                    "goal": plan.goal,
                    "steps": len(plan.steps),
                    "plan": plan.to_dict(),
                },
                message=f"planned {len(plan.steps)} steps",
                task_id=request.task_id,
            )
        if request.action == "refine_plan":
            raw = request.payload.get("plan")
            if raw is None:
                return AgentResult(
                    role=self.role,
                    action=request.action,
                    ok=False,
                    message="refine_plan requires a 'plan'",
                    task_id=request.task_id,
                )
            plan = TaskPlanner.from_dict(raw)
            refined = self.planner.refine(plan, request.payload.get("context", {}))
            return AgentResult(
                role=self.role,
                action=request.action,
                ok=True,
                output={"plan": refined.to_dict()},
                message="plan refined",
                task_id=request.task_id,
            )
        return AgentResult(
            role=self.role,
            action=request.action,
            ok=False,
            message=f"unsupported action {request.action!r}",
            task_id=request.task_id,
        )
