"""cadgenesis.agents.material
==========================
Material selection agent.

Thin orchestration wrapper over :class:`~cadgenesis.cad.materials.database.MaterialDatabase`
exposed as a Pillar 5 fleet agent.
"""

from __future__ import annotations

from typing import Any

from cadgenesis.agents.base import AgentRequest, AgentResult
from cadgenesis.agents.infrastructure import AgentBase, Capability


class MaterialAgent(AgentBase):
    """Selects, compares and validates engineering materials."""

    role = "material"
    actions = ("lookup", "compare", "select")
    version = "1.0.0"
    capabilities = (
        Capability("material.lookup", "look up material properties by name"),
        Capability("material.compare", "compare a set of materials"),
        Capability("material.select", "select materials meeting property targets"),
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
            if request.action == "lookup":
                name = str(payload.get("material", ""))
                if not name:
                    return self._fail(request, "payload requires 'material'")
                material = self.database[name]
                return self._ok(
                    request,
                    {
                        "material": material.name,
                        "category": material.category,
                        "properties": material.properties,
                        "cost_per_kg_usd": material.cost_per_kg_usd,
                    },
                    "material lookup succeeded",
                )
            if request.action == "compare":
                names = [str(n) for n in payload.get("materials", [])]
                if not names:
                    return self._fail(request, "payload requires 'materials'")
                rows = []
                for name in names:
                    mat = self.database[name]
                    rows.append(
                        {
                            "material": mat.name,
                            "category": mat.category,
                            "density_kg_m3": mat.density(),
                            "youngs_modulus_gpa": mat.youngs_modulus(),
                            "yield_strength_mpa": mat.yield_strength(),
                            "cost_per_kg_usd": mat.cost_per_kg_usd,
                        }
                    )
                return self._ok(request, {"rows": rows}, "material comparison complete")
            if request.action == "select":
                required = payload.get("required", {})
                if not isinstance(required, dict):
                    return self._fail(request, "payload 'required' must be a mapping")
                matches = []
                for name in self.database.names():
                    mat = self.database[name]
                    if self._meets(mat, required):
                        matches.append(
                            {
                                "material": mat.name,
                                "category": mat.category,
                                "cost_per_kg_usd": mat.cost_per_kg_usd,
                            }
                        )
                matches.sort(key=lambda row: row["cost_per_kg_usd"])
                return self._ok(
                    request,
                    {"matches": matches, "count": len(matches)},
                    "material selection complete",
                )
            return self._fail(request, f"unsupported action {request.action!r}")
        except KeyError as exc:
            return self._fail(request, f"material not found: {exc}")
        except Exception as exc:
            return self._fail(request, f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _meets(mat: Any, required: dict[str, Any]) -> bool:
        for key, value in required.items():
            if key.startswith("density_min") and mat.density() < float(value):
                return False
            if key.startswith("density_max") and mat.density() > float(value):
                return False
            if key.startswith("yield_min") and mat.yield_strength() < float(value):
                return False
            if key.startswith("youngs_min") and mat.youngs_modulus() < float(value):
                return False
            if key.startswith("cost_max") and mat.cost_per_kg_usd > float(value):
                return False
        return True

    def _ok(self, request: AgentRequest, output: dict[str, Any], message: str) -> AgentResult:
        return AgentResult(self.role, request.action, True, output, message, request.task_id)

    def _fail(self, request: AgentRequest, message: str) -> AgentResult:
        return AgentResult(self.role, request.action, False, {}, message, request.task_id)
