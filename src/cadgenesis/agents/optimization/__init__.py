"""cadgenesis.agents.optimization
===============================
Specialised optimization agent: recommends design adjustments toward goals.
"""

from __future__ import annotations

from cadgenesis.agents.base import Agent, AgentRequest, AgentResult


class OptimizationAgent(Agent):
    """Suggests parameter changes (mass, stress, cost) for a design."""

    role = "optimization"
    actions = ("optimize", "suggest")

    def __init__(self, target_cost: float | None = None) -> None:
        super().__init__()
        self.target_cost = target_cost

    def _objective_score(self, params: dict[str, float], objective: str) -> float:
        current = float(params.get("current", 0.0))
        target = float(params.get("target", current))
        if target == 0.0:
            return 1.0 if abs(current) < 1e-9 else 0.0
        return max(0.0, 1.0 - abs(current - target) / abs(target))

    def _recommendation(self, params: dict[str, float], objective: str) -> str:
        current = float(params.get("current", 0.0))
        target = float(params.get("target", current))
        delta = target - current
        if abs(delta) < 1e-9:
            return f"{objective} already at target ({target:.3f})"
        action = "increase" if delta > 0 else "decrease"
        return f"{action} {objective} by {abs(delta):.3f} toward {target:.3f}"

    def process(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        objective = payload.get("objective", "mass")
        params = payload.get("params")
        if not isinstance(params, dict):
            return AgentResult(
                role=self.role,
                action=request.action,
                ok=False,
                message="optimize requires a 'params' dictionary",
                task_id=request.task_id,
            )
        current = float(params.get("current", 0.0))
        score = self._objective_score(params, objective)
        cost = self.target_cost
        if cost is not None:
            cost_score = self._objective_score({"current": current, "target": cost}, "cost")
            score = 0.5 * score + 0.5 * cost_score
        return AgentResult(
            role=self.role,
            action=request.action,
            ok=score >= 0.999,
            output={
                "objective": objective,
                "score": round(score, 4),
                "recommendation": self._recommendation(params, objective),
                "confidence": min(1.0, score + 0.1),
                "option": score >= 0.999,
            },
            message=self._recommendation(params, objective),
            task_id=request.task_id,
        )


__all__ = ["OptimizationAgent"]
