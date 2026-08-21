"""cadgenesis.agents.design.fea
==============================
FEA stress agent for the design swarm.

:class:`FEAStressAgent` evaluates the von Mises equivalent stress of a part
under its load cases (via :class:`cadgenesis.world_model.mechanical.MechanicalReasoner`)
and, when the factor of safety falls below the yield target, issues an
automatic structural reinforcement through :class:`ReinforcementPolicy`.

Reinforcement rule
------------------
For an axial-dominated load, working stress scales with the inverse
cross-section area.  Scaling a circular section's radius (or a rectangular
one's width/height) by ``k`` multiplies the area by ``k^2``, so the growth
that restores the target safety factor is ``k = sqrt(target / current_sf)``.
"""

from __future__ import annotations

import math
from typing import Any

from cadgenesis.agents.base import AgentRequest, AgentResult
from cadgenesis.agents.design._helpers import build_load_cases, build_world_object
from cadgenesis.agents.infrastructure import AgentBase, Capability
from cadgenesis.world_model.mechanical import MechanicalReasoner
from cadgenesis.world_model.objects import WorldObject

_REINFORCEABLE = ("block", "cylinder", "revolve", "cone", "sphere")


class ReinforcementPolicy:
    """Cross-section scaling rule that restores a target safety factor."""

    def __init__(self, max_growth_per_step: float = 1.5) -> None:
        if max_growth_per_step <= 1.0:
            raise ValueError("max_growth_per_step must be > 1.0")
        self.max_growth_per_step = max_growth_per_step

    @staticmethod
    def required_growth(current_sf: float, target_sf: float) -> float:
        """Factor ``k`` such that ``A_new = A_old * k^2`` restores the SF."""
        if target_sf <= 0:
            raise ValueError("target_sf must be positive")
        if current_sf <= 1e-9:
            return float("inf")
        return math.sqrt(target_sf / current_sf)

    def reinforce(
        self,
        obj: WorldObject,
        target_sf: float,
        current_sf: float,
    ) -> tuple[dict[str, Any], float]:
        """Return ``(new_parameters, growth_factor)`` for the part.

        The growth is clamped to ``max_growth_per_step`` so a single step can
        never overshoot; the orchestration loop re-checks until the target is
        met.  Non-reinforceable features are returned unchanged with growth 1.
        """
        growth = min(self.required_growth(current_sf, target_sf), self.max_growth_per_step)
        if growth <= 1.0 or obj.feature not in _REINFORCEABLE:
            return dict(obj.parameters), 1.0
        params = dict(obj.parameters)
        if obj.feature in ("cylinder", "revolve", "cone", "sphere"):
            params["radius"] = float(params.get("radius", 5.0)) * growth
        else:
            width = float(params.get("width") or params.get("y") or 10.0)
            height = float(params.get("height") or params.get("z") or 10.0)
            params["width"] = width * growth
            params["height"] = height * growth
        return params, growth


class FEAStressAgent(AgentBase):
    """Evaluates von Mises stress and reinforces parts below the target SF."""

    role = "fea_stress"
    actions = ("analyze", "reinforce")
    version = "1.0.0"
    capabilities = (
        Capability(
            "fea.von_mises",
            "evaluate von Mises equivalent stress against yield strength",
            inputs=("object", "load_cases", "target_safety_factor"),
            outputs=("sigma_vm_mpa", "sigma_yield_mpa", "factor_of_safety", "passed"),
        ),
        Capability(
            "fea.reinforce",
            "scale the cross-section until the target safety factor is met",
            inputs=("object", "current_safety_factor", "target_safety_factor"),
            outputs=("parameters", "growth_factor"),
        ),
    )

    def __init__(
        self,
        reasoner: MechanicalReasoner | None = None,
        policy: ReinforcementPolicy | None = None,
    ) -> None:
        super().__init__()
        self.reasoner = reasoner or MechanicalReasoner()
        self.policy = policy or ReinforcementPolicy()

    def process(self, request: AgentRequest) -> AgentResult:
        try:
            if request.action == "analyze":
                return self._analyze(request)
            if request.action == "reinforce":
                return self._reinforce(request)
            return self._fail(request, f"unsupported action {request.action!r}")
        except (KeyError, TypeError, ValueError) as exc:
            return self._fail(request, f"{type(exc).__name__}: {exc}")

    # --------------------------------------------------------------- analyze

    def _analyze(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        obj = build_world_object(payload.get("object"))
        loads = build_load_cases(payload.get("load_cases"))
        if not loads:
            return self._fail(request, "analyze requires a non-empty 'load_cases' list")
        target = float(payload.get("target_safety_factor", 0.0)) or None
        result = self.reasoner.worst_von_mises(obj, loads, target)
        output = {
            "passed": result.passed,
            "sigma_vm_mpa": result.values["sigma_vm_mpa"],
            "sigma_yield_mpa": result.values["sigma_yield_mpa"],
            "factor_of_safety": result.values["factor_of_safety"],
            "target_safety_factor": result.values["target_safety_factor"],
            "load_case": result.name,
            "model": result.model,
            "details": result.details,
        }
        return AgentResult(
            self.role,
            request.action,
            ok=result.passed,
            output=output,
            message=(
                f"von Mises stress {output['sigma_vm_mpa']:.2f} MPa, "
                f"factor of safety {output['factor_of_safety']:.2f} "
                f"(target {output['target_safety_factor']:.2f})"
            ),
            task_id=request.task_id,
        )

    # ------------------------------------------------------------ reinforce

    def _reinforce(self, request: AgentRequest) -> AgentResult:
        payload = request.payload
        obj = build_world_object(payload.get("object"))
        target = float(payload.get("target_safety_factor", 1.5))
        if target <= 0:
            return self._fail(request, "target_safety_factor must be positive")
        current = float(payload.get("current_safety_factor", 0.0))
        if current <= 0:
            loads = build_load_cases(payload.get("load_cases"))
            if loads:
                current = self.reasoner.worst_von_mises(obj, loads, target).values[
                    "factor_of_safety"
                ]
        if current >= target:
            return AgentResult(
                self.role,
                request.action,
                ok=True,
                output={
                    "parameters": dict(obj.parameters),
                    "growth_factor": 1.0,
                    "reinforced": False,
                },
                message="no reinforcement needed",
                task_id=request.task_id,
            )
        parameters, growth = self.policy.reinforce(obj, target, current)
        reinforced = growth > 1.0
        output = {
            "parameters": parameters,
            "growth_factor": growth,
            "reinforced": reinforced,
            "current_safety_factor": current,
            "target_safety_factor": target,
        }
        return AgentResult(
            self.role,
            request.action,
            ok=True,
            output=output,
            message=(
                f"reinforced cross-section by x{growth:.3f} "
                f"(SF {current:.2f} -> target {target:.2f})"
                if reinforced
                else "feature not reinforceable; parameters unchanged"
            ),
            task_id=request.task_id,
        )

    # ----------------------------------------------------------------- misc

    def _fail(self, request: AgentRequest, message: str) -> AgentResult:
        return AgentResult(self.role, request.action, False, {}, message, request.task_id)


__all__ = ["FEAStressAgent", "ReinforcementPolicy"]
