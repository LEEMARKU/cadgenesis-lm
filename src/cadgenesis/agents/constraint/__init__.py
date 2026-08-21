"""cadgenesis.agents.constraint
=============================
Specialised constraint agent: solves and checks parametric constraints.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import Agent, AgentRequest, AgentResult
from cadgenesis.reasoning.constraint_solver import (
    Constraint,
    ConstraintSolver,
    Variable,
)


class ConstraintAgent(Agent):
    """Solves bounded linear constraint systems and reports feasibility."""

    role = "constraint"
    actions = ("solve", "check")

    def __init__(self, solver: ConstraintSolver | None = None) -> None:
        super().__init__()
        self.solver = solver or ConstraintSolver()

    def _build_variables(self, raw: list[dict[str, Any]]) -> list[Variable]:
        return [
            Variable(
                name=str(item["name"]),
                initial=float(item.get("initial", 0.0)),
                lower=item.get("lower"),
                upper=item.get("upper"),
            )
            for item in raw
        ]

    def _build_constraints(self, raw: list[dict[str, Any]]) -> list[Constraint]:
        return [
            Constraint(
                name=str(item["name"]),
                terms={str(k): float(v) for k, v in item["terms"].items()},
                operator=str(item["operator"]),
                rhs=float(item.get("rhs", 0.0)),
            )
            for item in raw
        ]

    def process(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        try:
            variables = self._build_variables(payload.get("variables", []))
            constraints = self._build_constraints(payload.get("constraints", []))
            solution = self.solver.solve(variables, constraints)
            if request.action == "check":
                return AgentResult(
                    role=self.role,
                    action=request.action,
                    ok=solution.feasible,
                    output=solution.summary(),
                    message=(
                        "constraints feasible"
                        if solution.feasible
                        else "; ".join(solution.messages)
                    ),
                    task_id=request.task_id,
                )
            return AgentResult(
                role=self.role,
                action=request.action,
                ok=solution.feasible,
                output=solution.summary(),
                message="system solved",
                task_id=request.task_id,
            )
        except (KeyError, ValueError, TypeError) as exc:
            return AgentResult(
                role=self.role,
                action=request.action,
                ok=False,
                message=str(exc),
                task_id=request.task_id,
            )
