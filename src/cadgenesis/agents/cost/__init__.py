"""cadgenesis.agents.cost
========================
Cost estimation agent.

Estimates manufacturing cost from material mass, process and quantity using
simple engineering cost models, and compares design alternatives.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import AgentRequest, AgentResult
from cadgenesis.agents.infrastructure import AgentBase, Capability

# Representative process cost factors (USD per unit / per kg) — engineering
# nominal values used for comparative estimation, not quotes.
_PROCESS_FACTOR = {
    "machining": 1.0,
    "injection_molding": 0.4,
    "3d_printing": 1.6,
    "sheet_metal": 0.7,
    "casting": 0.5,
    "extrusion": 0.3,
}


class CostAgent(AgentBase):
    """Estimates and compares manufacturing costs."""

    role = "cost"
    actions = ("estimate", "compare")
    version = "1.0.0"
    capabilities = (
        Capability("cost.estimate", "estimate unit cost from mass/process/material"),
        Capability("cost.compare", "compare cost across alternatives"),
    )

    def __init__(self, database: Any = None) -> None:
        super().__init__()
        if database is None:
            from cadgenesis.cad.materials.database import MaterialDatabase

            database = MaterialDatabase()
        self.database = database

    def process(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        try:
            if request.action == "estimate":
                estimate = self._estimate(payload)
                if estimate is None:
                    return self._fail(request, "payload requires 'mass_kg'")
                return self._ok(
                    request,
                    estimate,
                    f"estimated cost ${estimate['total_cost_usd']:.2f}",
                )
            if request.action == "compare":
                options = payload.get("options", [])
                if not isinstance(options, list) or not options:
                    return self._fail(request, "payload requires non-empty 'options'")
                rows = []
                for option in options:
                    estimate = self._estimate(option)
                    if estimate is None:
                        rows.append({"option": option, "error": "missing mass_kg"})
                    else:
                        rows.append({"option": option, **estimate})
                rows.sort(key=lambda row: row.get("total_cost_usd", float("inf")))
                return self._ok(request, {"rows": rows}, "cost comparison complete")
            return self._fail(request, f"unsupported action {request.action!r}")
        except Exception as exc:
            return self._fail(request, f"{type(exc).__name__}: {exc}")

    def _estimate(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        mass_kg = payload.get("mass_kg")
        if mass_kg is None:
            return None
        mass_kg = float(mass_kg)
        process = str(payload.get("process", "machining"))
        quantity = max(1, int(payload.get("quantity", 1)))
        material = str(payload.get("material", "steel"))
        unit_cost = 0.0
        try:
            mat = self.database[material]
            unit_cost = mat.cost_per_kg_usd
        except KeyError:
            unit_cost = float(payload.get("material_cost_per_kg_usd", 2.0))
        process_factor = _PROCESS_FACTOR.get(process, 1.0)
        base = unit_cost * mass_kg * process_factor
        setup = float(payload.get("setup_usd", 50.0))
        per_unit = base + setup / quantity
        total = per_unit * quantity
        return {
            "mass_kg": mass_kg,
            "process": process,
            "quantity": quantity,
            "material": material,
            "material_cost_per_kg_usd": unit_cost,
            "unit_cost_usd": round(per_unit, 2),
            "total_cost_usd": round(total, 2),
        }

    def _ok(self, request: AgentRequest, output: dict[str, Any], message: str) -> AgentResult:
        return AgentResult(self.role, request.action, True, output, message, request.task_id)

    def _fail(self, request: AgentRequest, message: str) -> AgentResult:
        return AgentResult(self.role, request.action, False, {}, message, request.task_id)
