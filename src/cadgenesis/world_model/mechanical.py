"""cadgenesis.world_model.mechanical
===================================
Mechanical reasoning (Pillar 4).

The :class:`MechanicalReasoner` answers physical questions about a design:
working stress under a load case, factor of safety versus the material's
yield strength, von Mises equivalent stress, stability margins, and
load-bearing capacity.

The physics is deliberately *proxy-level* (closed-form first-order
estimates) so the world model can reason quickly without a FEM solver;
results are returned with an explicit ``model`` tag so downstream consumers
know the approximation level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.world_model.objects import (
    BoundaryCondition,
    LoadCase,
    Material,
    WorldObject,
)


@dataclass
class MechanicalResult:
    """Outcome of a mechanical check."""

    name: str
    passed: bool
    details: str = ""
    values: dict[str, float] = field(default_factory=dict)
    model: str = "first_order_proxy"

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
            "values": dict(self.values),
            "model": self.model,
        }


class MechanicalReasoner:
    """First-order mechanical reasoning over world-model objects.

    ``safety_factor`` is the default target factor of safety used when a load
    case does not specify one.
    """

    def __init__(self, safety_factor: float = 2.5) -> None:
        if safety_factor <= 0:
            raise ValueError("safety factor must be positive")
        self.safety_factor = safety_factor

    # ------------------------------------------------------------ sections

    @staticmethod
    def _cross_section_area(obj: WorldObject, axis: str = "z") -> float:
        """Minimum cross-section area (mm^2) normal to ``axis``."""
        feature = obj.feature
        p = obj.parameters
        if feature == "block":
            u = float(p.get("width", p.get("y", 10.0)))
            v = float(p.get("height", p.get("z", 10.0)))
        elif feature in ("cylinder", "revolve") or feature == "sphere":
            r = float(p.get("radius", 5.0))
            u = v = 2.0 * r
        else:
            u = float(p.get("width", 10.0))
            v = float(p.get("height", 10.0))
        if axis == "x":
            return u * v
        if axis == "y":
            return u * v
        return u * v

    def working_stress_mpa(
        self,
        obj: WorldObject,
        load: BoundaryCondition,
    ) -> float:
        """Axial working stress ``F / A`` in MPa (proxy, no stress concentrators)."""
        if load.kind == "force":
            area = self._cross_section_area(obj)
            if area <= 0:
                return 0.0
            return load.magnitude / area  # N / mm^2 = MPa
        if load.kind == "pressure":
            return load.magnitude  # MPa
        if load.kind == "torque":
            # Tau ~ T / (2 * A * r_effective)

            r = float(obj.parameters.get("radius", 5.0))
            area = self._cross_section_area(obj)
            if area <= 0 or r <= 0:
                return 0.0
            return load.magnitude / (2.0 * area * r) * 1000.0
        return 0.0

    def factor_of_safety(
        self,
        obj: WorldObject,
        load: BoundaryCondition,
    ) -> float:
        """Yield strength / working stress; ``inf``-like large value when no load."""
        stress = self.working_stress_mpa(obj, load)
        material = obj.material or Material()
        yield_strength = material.yield_strength_mpa
        if stress <= 1e-9:
            return 1e9
        return yield_strength / stress

    def check_load(
        self,
        obj: WorldObject,
        load: LoadCase,
        target_safety_factor: float | None = None,
    ) -> MechanicalResult:
        """Evaluate an entire load case and report the worst margin."""
        target = target_safety_factor or load.safety_factor_target or self.safety_factor
        worst_stress = 0.0
        worst_condition: BoundaryCondition | None = None
        for condition in load.conditions:
            stress = self.working_stress_mpa(obj, condition)
            if stress > worst_stress:
                worst_stress = stress
                worst_condition = condition
        fos = self.factor_of_safety(obj, worst_condition) if worst_condition else 1e9
        passed = fos >= target
        return MechanicalResult(
            name=f"load_case.{load.name}",
            passed=passed,
            details=(
                f"worst stress {worst_stress:.2f} MPa, "
                f"factor of safety {fos:.2f} vs target {target:.2f}"
            ),
            values={
                "worst_stress_mpa": worst_stress,
                "factor_of_safety": fos,
                "target_safety_factor": target,
            },
        )

    def stability(
        self,
        obj: WorldObject,
    ) -> MechanicalResult:
        """Buckling proxy: compare height/width slenderness to a threshold.

        Passes when the object is not excessively slender (height <= 8x the
        minimum lateral dimension) — a cheap stand-in for critical buckling.
        """
        p = obj.parameters
        height = float(p.get("height", p.get("length", 10.0)))
        lateral = float(p.get("width", p.get("radius", p.get("y", p.get("z", 10.0)))))
        if lateral <= 0:
            lateral = 1e-6
        slenderness = height / lateral
        ok = slenderness <= 8.0
        return MechanicalResult(
            name="stability.buckling",
            passed=ok,
            details=f"slenderness {slenderness:.2f} (threshold 8.0)",
            values={"slenderness": slenderness, "threshold": 8.0},
        )

    def mass_budget(
        self,
        objects: list[WorldObject],
        limit_kg: float | None = None,
    ) -> MechanicalResult:
        """Total mass vs an optional budget."""
        total = sum(o.mass() for o in objects)
        ok = limit_kg is None or total <= limit_kg
        return MechanicalResult(
            name="mass_budget",
            passed=ok,
            details=(
                f"total mass {total:.3f} kg"
                + (f" (limit {limit_kg:.3f})" if limit_kg is not None else "")
            ),
            values={"total_mass_kg": total, "limit_kg": limit_kg if limit_kg is not None else 0.0},
        )

    def strength_check(
        self,
        obj: WorldObject,
        loads: list[LoadCase],
    ) -> MechanicalResult:
        """Aggregate the strictest result across all load cases."""
        worst: MechanicalResult | None = None
        for load in loads:
            result = self.check_load(obj, load)
            if (
                worst is None
                or result.values["factor_of_safety"] < worst.values["factor_of_safety"]
            ):
                worst = result
        assert worst is not None
        return worst

    # ---------------------------------------------------------- von Mises

    def _polar_moment(self, obj: WorldObject) -> float:
        """Polar moment of inertia ``J`` (mm^4) of the load cross-section.

        Circular families use the solid-shaft formula ``pi*r^4/2``; other
        families approximate the solid rectangle ``a*b*(a^2 + b^2)/12``.
        """
        if obj.feature in ("cylinder", "revolve", "sphere", "cone"):
            r = float(obj.parameters.get("radius", 5.0))
            return math.pi * r**4 / 2.0
        b = float(obj.parameters.get("width", obj.parameters.get("y", 10.0)))
        h = float(obj.parameters.get("height", obj.parameters.get("z", 10.0)))
        return b * h * (b * b + h * h) / 12.0

    def _radius_effective(self, obj: WorldObject) -> float:
        """Outer radius (mm) used to convert torque into surface shear."""
        if obj.feature in ("cylinder", "revolve", "sphere", "cone"):
            return float(obj.parameters.get("radius", 5.0))
        b = float(obj.parameters.get("width", obj.parameters.get("y", 10.0)))
        h = float(obj.parameters.get("height", obj.parameters.get("z", 10.0)))
        return math.hypot(b, h) / 2.0

    def stress_state_mpa(self, obj: WorldObject, load: LoadCase) -> dict[str, float]:
        """Decompose a load case into proxy stress components (MPa).

        Components (all magnitudes in the payload units used by
        :meth:`working_stress_mpa`):

        * ``sigma_axial_mpa`` — normal stress from ``force`` conditions ``F/A``.
        * ``sigma_pressure_mpa`` — normal stress from ``pressure`` conditions.
        * ``tau_torsion_mpa`` — shear stress from ``torque`` conditions
          ``T*r/J`` (torque treated as N*mm, matching the existing proxy).
        """
        sigma_axial = 0.0
        sigma_pressure = 0.0
        tau_torsion = 0.0
        area = self._cross_section_area(obj)
        for condition in load.conditions:
            if condition.kind == "force":
                if area > 0:
                    sigma_axial += condition.magnitude / area
            elif condition.kind == "pressure":
                sigma_pressure += condition.magnitude
            elif condition.kind == "torque":
                j = self._polar_moment(obj)
                r = self._radius_effective(obj)
                if j > 0 and r > 0:
                    tau_torsion += condition.magnitude * r / j
        return {
            "sigma_axial_mpa": sigma_axial,
            "sigma_pressure_mpa": sigma_pressure,
            "tau_torsion_mpa": tau_torsion,
        }

    def von_mises_stress_mpa(self, obj: WorldObject, load: LoadCase) -> float:
        """Von Mises equivalent stress (MPa) for a load case.

        Combines the normal components (axial + pressure, superposed) with the
        torsional shear through ``sqrt(sigma^2 + 3*tau^2)`` — the classic
        shaft/rod closed-form for a uniaxial-plus-torsion state.
        """
        state = self.stress_state_mpa(obj, load)
        sigma = state["sigma_axial_mpa"] + state["sigma_pressure_mpa"]
        tau = state["tau_torsion_mpa"]
        return math.sqrt(sigma * sigma + 3.0 * tau * tau)

    def check_von_mises(
        self,
        obj: WorldObject,
        load: LoadCase,
        target_safety_factor: float | None = None,
    ) -> MechanicalResult:
        """Yield check on the von Mises equivalent stress.

        The part fails (``passed`` is False) when
        ``sigma_vm > sigma_yield / target``, i.e. the von Mises factor of
        safety drops below the target — the trigger used by the
        ``FEAStressAgent`` to request automatic structural reinforcement.
        """
        target = target_safety_factor or load.safety_factor_target or self.safety_factor
        sigma_vm = self.von_mises_stress_mpa(obj, load)
        material = obj.material or Material()
        yield_strength = material.yield_strength_mpa
        fos = 1e9 if sigma_vm <= 1e-9 else yield_strength / sigma_vm
        passed = fos >= target
        return MechanicalResult(
            name=f"von_mises.{load.name}",
            passed=passed,
            details=(
                f"sigma_vm {sigma_vm:.2f} MPa vs yield {yield_strength:.2f} MPa, "
                f"factor of safety {fos:.2f} vs target {target:.2f}"
            ),
            values={
                "sigma_vm_mpa": sigma_vm,
                "sigma_yield_mpa": yield_strength,
                "factor_of_safety": fos,
                "target_safety_factor": target,
            },
            model="von_mises_first_order",
        )

    def worst_von_mises(
        self,
        obj: WorldObject,
        loads: list[LoadCase],
        target_safety_factor: float | None = None,
    ) -> MechanicalResult:
        """Strictest von Mises result across every load case."""
        if not loads:
            raise ValueError("at least one load case is required")
        worst: MechanicalResult | None = None
        for load in loads:
            result = self.check_von_mises(obj, load, target_safety_factor)
            if (
                worst is None
                or result.values["factor_of_safety"] < worst.values["factor_of_safety"]
            ):
                worst = result
        assert worst is not None
        return worst


__all__ = ["MechanicalReasoner", "MechanicalResult"]
