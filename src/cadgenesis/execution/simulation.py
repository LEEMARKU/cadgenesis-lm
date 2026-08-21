"""cadgenesis.execution.simulation
=================================
Analytic first-order simulation interfaces for the CAD execution pipeline.

Pluggable analysis types — structural, thermal, fluid, motion and tolerance —
backed by closed-form first-order solvers (pure Python).  External solvers
can be registered per analysis type through ``register_solver``; the default
analytics stay available as fallbacks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.reasoning.geometry_reasoner import GeometryReasoner
from cadgenesis.world_model.simulator import MotionSimulator, SimulatedPose

ANALYSIS_TYPES = ("structural", "thermal", "fluid", "motion", "tolerance")


@dataclass
class SimulationResult:
    """Result of one analytic simulation run."""

    analysis_type: str
    passed: bool
    values: dict[str, float] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    model: str = "first_order_analytic"

    def summary(self) -> dict[str, Any]:
        return {
            "analysis_type": self.analysis_type,
            "passed": self.passed,
            "values": self.values,
            "messages": list(self.messages),
            "model": self.model,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.summary()


class SimulationEngine:
    """Analytic simulation engine with pluggable per-type solvers."""

    def __init__(self, safety_factor: float = 1.5) -> None:
        if safety_factor <= 0:
            raise ValueError("safety_factor must be > 0")
        self.safety_factor = safety_factor
        self._solvers: dict[str, Callable[..., SimulationResult]] = {}

    # ------------------------------------------------------------- dispatch

    def run(self, analysis_type: str, **params: Any) -> SimulationResult:
        """Run an analysis; raises ``ValueError`` on unknown analysis types."""
        if analysis_type not in ANALYSIS_TYPES:
            raise ValueError(f"unknown analysis type {analysis_type!r}")
        solver = self._solvers.get(analysis_type)
        if solver is not None:
            return solver(**params)
        methods: dict[str, Callable[..., SimulationResult]] = {
            "structural": self.structural,
            "thermal": self.thermal,
            "fluid": self.fluid,
            "motion": self.motion,
            "tolerance": self.tolerance,
        }
        return methods[analysis_type](**params)

    def register_solver(self, analysis_type: str, solver: Callable[..., SimulationResult]) -> None:
        """Register an external solver for an analysis type."""
        if analysis_type not in ANALYSIS_TYPES:
            raise ValueError(f"unknown analysis type {analysis_type!r}")
        self._solvers[analysis_type] = solver

    def supported(self) -> list[str]:
        return list(ANALYSIS_TYPES)

    # ------------------------------------------------------------ structural

    def structural(self, part: dict[str, Any]) -> SimulationResult:
        """Closed-form axial stress check: ``stress = F / A``.

        ``part`` carries ``material`` (``yield_strength_pa``) and ``load``
        (``force_n``, optional ``area_m2``); a ``dimensions`` dict with
        ``width_m``/``height_m`` provides the area when not given.
        """
        material = part.get("material") or {}
        load = part.get("load") or {}
        force = float(load.get("force_n") or 0.0)
        area = load.get("area_m2")
        if area is None:
            dims = part.get("dimensions") or {}
            area = float(dims.get("width_m") or 0.01) * float(dims.get("height_m") or 0.01)
        yield_strength = float(material.get("yield_strength_pa") or 0.0)
        values: dict[str, float] = {}
        messages: list[str] = []
        if force <= 0.0:
            values.update({"stress_pa": 0.0, "factor_of_safety": 1e9})
            messages.append("no load case")
            return SimulationResult("structural", True, values, messages)
        if area <= 0.0 or yield_strength <= 0.0:
            values.update({"stress_pa": 0.0, "factor_of_safety": 0.0})
            messages.append("missing area or yield strength")
            return SimulationResult("structural", False, values, messages)
        stress = force / area
        safety = yield_strength / stress
        values.update(
            {
                "stress_pa": round(stress, 6),
                "factor_of_safety": round(safety, 6),
            }
        )
        passed = safety >= self.safety_factor
        messages.append(f"factor of safety {safety:.2f} vs required {self.safety_factor}")
        return SimulationResult("structural", passed, values, messages)

    # --------------------------------------------------------------- thermal

    def thermal(self, part: dict[str, Any]) -> SimulationResult:
        """Steady-state 1-D conduction: ``dT = q * L / (k * A)``.

        ``part`` carries ``material`` (``thermal_conductivity_w_mk``) and
        ``load`` (``heat_w``, optional ``area_m2``), plus ``dimensions`` with
        ``thickness_m``.
        """
        material = part.get("material") or {}
        load = part.get("load") or {}
        dims = part.get("dimensions") or {}
        heat = float(load.get("heat_w") or 0.0)
        conductivity = float(material.get("thermal_conductivity_w_mk") or 0.0)
        thickness = float(dims.get("thickness_m") or 0.001)
        area = load.get("area_m2")
        if area is None:
            area = float(dims.get("area_m2") or 0.01)
        values: dict[str, float] = {}
        messages: list[str] = []
        if heat <= 0.0:
            values.update({"delta_t_k": 0.0})
            messages.append("no heat load")
            return SimulationResult("thermal", True, values, messages)
        if conductivity <= 0.0 or area <= 0.0:
            messages.append("missing conductivity or area")
            return SimulationResult("thermal", False, values, messages)
        delta_t = heat * thickness / (conductivity * area)
        limit = float(part.get("max_temperature_rise_k") or float("inf"))
        values.update({"delta_t_k": round(delta_t, 6)})
        passed = delta_t <= limit
        messages.append(f"temperature rise {delta_t:.2f} K")
        return SimulationResult("thermal", passed, values, messages)

    # ----------------------------------------------------------------- fluid

    def fluid(self, part: dict[str, Any]) -> SimulationResult:
        """Darcy-style pressure drop: ``dP = f * rho * v^2 * L / (2 * D)``."""
        load = part.get("load") or {}
        dims = part.get("dimensions") or {}
        velocity = float(load.get("velocity_m_s") or 0.0)
        density = float(load.get("density_kg_m3") or 1000.0)
        length = float(dims.get("length_m") or 0.1)
        diameter = float(dims.get("diameter_m") or 0.01)
        friction = float(load.get("friction_factor") or 0.02)
        values: dict[str, float] = {}
        if velocity <= 0.0:
            values.update({"pressure_drop_pa": 0.0})
            return SimulationResult("fluid", True, values, ["no flow"])
        if diameter <= 0.0 or length <= 0.0:
            return SimulationResult("fluid", False, values, ["missing geometry"])
        drop = friction * density * velocity**2 * length / (2.0 * diameter)
        limit = float(part.get("max_pressure_drop_pa") or float("inf"))
        values.update({"pressure_drop_pa": round(drop, 6)})
        return SimulationResult("fluid", drop <= limit, values, [f"dP {drop:.2f} Pa"])

    # ---------------------------------------------------------------- motion

    def motion(self, mechanism: Any, states: dict[str, float]) -> SimulationResult:
        """Forward-kinematics motion simulation via ``MotionSimulator``."""
        simulator = MotionSimulator()
        try:
            pose: SimulatedPose = simulator.simulate(mechanism, states)
        except (TypeError, ValueError, KeyError) as exc:
            return SimulationResult("motion", False, {}, [f"motion failed: {exc}"])
        link = next(iter(pose.link_poses)) if pose.link_poses else ""
        position = pose.position_of(link) if link else (0.0, 0.0, 0.0)
        values = {"position_x": position[0], "position_y": position[1], "position_z": position[2]}
        return SimulationResult("motion", True, values, [f"pose at t={pose.time}"])

    # ------------------------------------------------------------- tolerance

    def tolerance(self, chain: list[tuple[float, float]]) -> SimulationResult:
        """Stack-up analysis over (nominal, tolerance) chain pairs."""
        result = GeometryReasoner.tolerance_stack(chain)
        values = {
            "nominal": round(result["nominal"], 6),
            "worst_case": round(result["worst"], 6),
            "rss": round(result["rss"], 6),
        }
        limit = float(result.get("limit") or float("inf"))
        passed = result["worst"] <= limit
        return SimulationResult("tolerance", passed, values, ["stack-up computed"])

    # ----------------------------------------------------------------- misc

    def summary(self) -> dict[str, Any]:
        return {
            "analysis_types": list(ANALYSIS_TYPES),
            "safety_factor": self.safety_factor,
            "external_solvers": sorted(self._solvers),
        }


__all__ = [
    "ANALYSIS_TYPES",
    "SimulationEngine",
    "SimulationResult",
]
