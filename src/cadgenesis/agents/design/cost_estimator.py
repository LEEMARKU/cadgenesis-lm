"""cadgenesis.agents.design.cost_estimator
=========================================
Cost-estimation agent for the design swarm.

:class:`CostEstimatorAgent` prices a part from its mass (computed by the
world-model :class:`~cadgenesis.world_model.objects.WorldObject`), the
material database and a process cost factor, and compares the cost drift
across design iterations so the orchestration loop can report how much the
structural reinforcement adds to the part price.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import AgentRequest, AgentResult
from cadgenesis.agents.design._helpers import build_world_object
from cadgenesis.agents.infrastructure import AgentBase, Capability

# Representative process cost factors (USD per kg of base material cost) —
# engineering nominals for comparative estimation, not quotes.
_PROCESS_FACTOR = {
    "machining": 1.0,
    "injection_molding": 0.4,
    "3d_printing": 1.6,
    "sheet_metal": 0.7,
    "casting": 0.5,
    "extrusion": 0.3,
}


class CostEstimatorAgent(AgentBase):
    """Estimates unit/total cost and tracks cost across iterations."""

    role = "cost_estimator"
    actions = ("estimate", "compare_iterations")
    version = "1.0.0"
    capabilities = (
        Capability(
            "cost.estimate",
            "estimate unit and total manufacturing cost from the part geometry",
            inputs=("part", "process", "quantity"),
            outputs=("mass_kg", "unit_cost_usd", "total_cost_usd"),
        ),
        Capability(
            "cost.compare_iterations",
            "compare cumulative cost across design iterations",
            inputs=("iterations",),
            outputs=("rows",),
        ),
    )

    def __init__(self, database: Any = None) -> None:
        super().__init__()
        if database is None:
            from cadgenesis.cad.materials.database import MaterialDatabase

            database = MaterialDatabase()
        self.database = database

    def process(self, request: AgentRequest) -> AgentResult:
        try:
            if request.action == "estimate":
                return self._estimate(request)
            if request.action == "compare_iterations":
                return self._compare(request)
            return self._fail(request, f"unsupported action {request.action!r}")
        except (KeyError, TypeError, ValueError) as exc:
            return self._fail(request, f"{type(exc).__name__}: {exc}")

    # -------------------------------------------------------------- estimate

    def _estimate(self, request: AgentRequest) -> AgentResult:
        part = request.payload.get("part")
        if not isinstance(part, dict):
            return self._fail(request, "estimate requires a 'part' dictionary")
        estimate = self._estimate_part(
            part,
            quantity=request.payload.get("quantity", 1),
            process=request.payload.get("process", "machining"),
            setup_usd=request.payload.get("setup_usd", 50.0),
        )
        return AgentResult(
            self.role,
            request.action,
            ok=True,
            output=estimate,
            message=f"estimated ${estimate['total_cost_usd']:.2f} for {estimate['quantity']} units",
            task_id=request.task_id,
        )

    def _estimate_part(
        self,
        part: dict[str, Any],
        quantity: int,
        process: Any,
        setup_usd: float,
    ) -> dict[str, Any]:
        obj = build_world_object(part)
        mass_kg = obj.mass()
        material_name = obj.material.name if obj.material is not None else "steel"
        try:
            material = self.database[material_name]
            unit_cost = float(material.cost_per_kg_usd)
        except (KeyError, TypeError):
            unit_cost = float(part.get("material_cost_per_kg_usd", 2.0))
        quantity = max(1, int(quantity))
        factor = _PROCESS_FACTOR.get(str(process), 1.0)
        base = unit_cost * mass_kg * factor
        per_unit = base + float(setup_usd) / quantity
        total = per_unit * quantity
        return {
            "part": str(part.get("name", "part")),
            "material": material_name,
            "mass_kg": round(mass_kg, 6),
            "process": str(process),
            "quantity": quantity,
            "unit_cost_usd": round(per_unit, 2),
            "total_cost_usd": round(total, 2),
        }

    # -------------------------------------------------------- iteration cost

    def _compare(self, request: AgentRequest) -> AgentResult:
        iterations = request.payload.get("iterations")
        if not isinstance(iterations, list) or not iterations:
            return self._fail(request, "compare_iterations requires a non-empty 'iterations' list")
        rows: list[dict[str, Any]] = []
        previous: float | None = None
        for index, item in enumerate(iterations):
            if not isinstance(item, dict):
                raise TypeError("every iteration must be a dict")
            estimate = self._estimate_part(
                item.get("part", {}),
                quantity=item.get("quantity", 1),
                process=item.get("process", "machining"),
                setup_usd=item.get("setup_usd", 50.0),
            )
            total = float(estimate["total_cost_usd"])
            delta = None if previous is None else round(total - previous, 2)
            rows.append({"iteration": index, **estimate, "delta_usd": delta})
            previous = total
        return AgentResult(
            self.role,
            request.action,
            ok=True,
            output={"rows": rows},
            message="iteration cost comparison complete",
            task_id=request.task_id,
        )

    # ----------------------------------------------------------------- misc

    def _fail(self, request: AgentRequest, message: str) -> AgentResult:
        return AgentResult(self.role, request.action, False, {}, message, request.task_id)


__all__ = ["CostEstimatorAgent"]
