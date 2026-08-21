"""cadgenesis.execution.ir_execution
=================================
IR-native execution (v6.4) with simulation integration (v7.1).

Walks a :class:`~cadgenesis.ir.program.CadProgram` in topological order and
materialises every operation into a world-model
:class:`~cadgenesis.world_model.objects.WorldObject`, producing a queryable
:class:`IRExecutionState` (per-op poses, bounds, volume, mass, applied
features).  No token-stream round-trip, no heuristics on raw text.

Simulation (v7.1): after materialisation, an optional
:class:`~cadgenesis.execution.simulation.SimulationEngine` can be invoked
to run analytic FEA/thermal/fluid/motion analyses on the materialised part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cadgenesis.execution.simulation import SimulationEngine, SimulationResult
from cadgenesis.ir.diff import IrDiffReport, ir_diff
from cadgenesis.ir.program import CadOperation, CadProgram
from cadgenesis.ir.validator import validate_program_ir

#: IR primitive kinds with a world-model family (kind -> feature family).
_KIND_TO_FAMILY = {
    "PRIM_BOX": "block",
    "PRIM_CYLINDER": "cylinder",
    "PRIM_SPHERE": "sphere",
}


@dataclass
class IRObjectState:
    """One executed IR operation, materialised into the world model."""

    op_id: str
    kind: str
    position: int
    feature: str
    object_id: str
    params: dict[str, Any] = field(default_factory=dict)
    pose: Any = None
    bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None
    volume_m3: float = 0.0
    mass_kg: float = 0.0
    applied_features: list[dict[str, Any]] = field(default_factory=list)
    parent_op_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_id": self.op_id,
            "kind": self.kind,
            "position": self.position,
            "feature": self.feature,
            "object_id": self.object_id,
            "params": self.params,
            "pose": self.pose.to_list() if self.pose is not None else None,
            "bounds": self.bounds,
            "volume_m3": self.volume_m3,
            "mass_kg": self.mass_kg,
            "applied_features": self.applied_features,
            "parent_op_id": self.parent_op_id,
        }


@dataclass
class IRExecutionState:
    """Queryable executed state of an IR program."""

    program_id: str
    schema_version: str = "1.0.0"
    objects: list[IRObjectState] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    validated: bool = False

    def object(self, op_id: str) -> IRObjectState | None:
        for obj in self.objects:
            if obj.op_id == op_id:
                return obj
        return None

    def objects_of(self, kind: str) -> list[IRObjectState]:
        return [obj for obj in self.objects if obj.feature == kind]

    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
        if not self.objects:
            return None
        all_lo = min(o.bounds[0] for o in self.objects)
        all_hi = max(o.bounds[1] for o in self.objects)
        return (all_lo, all_hi)

    def total_volume(self) -> float:
        return sum(o.volume_m3 for o in self.objects)

    def total_mass(self) -> float:
        return sum(o.mass_kg for o in self.objects)

@dataclass
class IRExecutionResult:
    """Outcome of executing one IR program."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    state: IRExecutionState | None = None
    simulation: SimulationResult | None = None

    def summary(self) -> dict[str, Any]:
        if self.state is None:
            return {
                "valid": self.valid,
                "errors": self.errors,
                "objects": 0,
                "unresolved": [],
                "simulation": None,
            }
        base = {
            "valid": self.valid,
            "errors": self.errors,
            "objects": len(self.state.objects),
            "unresolved": self.state.unresolved,
            "program_id": self.state.program_id,
        }
        if self.simulation is not None:
            base["simulation"] = self.simulation.summary()
        return base

    def to_dict(self) -> dict[str, Any]:
        data = self.summary()
        if self.state is not None:
            data["state"] = self.state.to_dict()
        return data


def _primitive_parameters(kind: str, params: dict[str, Any]) -> dict[str, Any]:
    """Map IR d0/d1/d2 parameters onto world-model feature parameters."""
    if kind == "PRIM_BOX":
        mapped: dict[str, Any] = {}
        if "d0" in params:
            mapped["length"] = float(params["d0"])
        if "d1" in params:
            mapped["width"] = float(params["d1"])
        if "d2" in params:
            mapped["height"] = float(params["d2"])
        if "fillet" in params:
            mapped["fillet"] = float(params["fillet"])
        return mapped
    if kind == "PRIM_CYLINDER":
        mapped = {}
        if "d0" in params:
            mapped["radius"] = float(params["d0"]) / 2.0
        if "d1" in params:
            mapped["height"] = float(params["d1"])
        if "fillet" in params:
            mapped["fillet"] = float(params["fillet"])
        return mapped
    if kind == "PRIM_SPHERE":
        mapped = {}
        if "d0" in params:
            mapped["radius"] = float(params["d0"]) / 2.0
        if "fillet" in params:
            mapped["fillet"] = float(params["fillet"])
        return mapped
    return dict(params)


def _topological_steps(program: CadProgram) -> list[CadOperation]:
    """Steps in dependency order (Kahn); falls back to file order on cycles."""
    by_id = {s.op_id: s for s in program.steps}
    order = program.topological_order()
    if len(order) != len(program.steps):
        return list(program.steps)
    return [by_id[oid] for oid in order]


class IRExecutionEngine:
    """IR-native executor: typed program graph -> materialised world state.

    v7.1: accepts an optional ``simulation`` engine that runs analytic
    analyses after materialisation.
    """

    def __init__(
        self,
        vocab=None,
        material: Any = None,
        simulation: Any = None,
    ) -> None:
        self.vocab = vocab
        self.material = material
        self.simulation = simulation or SimulationEngine()

    def execute(
        self, program: CadProgram
    ) -> IRExecutionResult:
        if not isinstance(program, CadProgram):
            raise TypeError(f"expected CadProgram, got {type(program).__name__}")

        from cadgenesis.world_model import make_object

        report = validate_program_ir(program, vocab=self.vocab)
        if not report.passed:
            errors = [f"[{c.name}] {c.detail}" for c in report.checks if not c.passed]
            return IRExecutionResult(valid=False, errors=errors)

        objects: list[IRObjectState] = []
        unresolved: list[str] = []
        last_primitive: IRObjectState | None = None

        for step in _topological_steps(program):
            kind = step.kind
            if kind in _KIND_TO_FAMILY:
                family = _KIND_TO_FAMILY[kind]
                parameters = _primitive_parameters(kind, step.params)
                obj = make_object(
                    feature=family,
                    name=f"{family}_{step.position}",
                    parameters=parameters,
                    material=self.material,
                )
                lo, hi = obj.bounds()
                state = IRObjectState(
                    op_id=step.op_id,
                    kind=kind,
                    position=step.position,
                    feature=family,
                    object_id=obj.object_id,
                    params=dict(parameters),
                    pose=obj.pose,
                    bounds=(
                        (lo.x, lo.y, lo.z),
                        (hi.x, hi.y, hi.z),
                    ),
                    volume_m3=obj.volume_estimate() * 1e-9,
                    mass_kg=obj.mass(),
                    parent_op_id=last_primitive.op_id if last_primitive else None,
                )
                objects.append(state)
                last_primitive = state
                continue

            if kind.startswith("FEAT_"):
                parent = last_primitive
                if step.depends_on:
                    dep_prim = next(
                        (o for o in objects if o.op_id in step.depends_on), None
                    )
                    if dep_prim is not None:
                        parent = dep_prim
                record = {
                    "op_id": step.op_id,
                    "kind": kind,
                    "params": dict(step.params),
                }
                if parent is None:
                    unresolved.append(f"{step.op_id}:{kind} (no parent primitive)")
                    continue
                parent.applied_features.append(record)
                if kind == "FEAT_FILLET" and isinstance(step.params.get("d0"), (int, float)):
                    current = float(parent.params.get("fillet", 0.0))
                    parent.params["fillet"] = current + float(step.params["d0"])
                continue

            unresolved.append(f"{step.op_id}:{kind} (no world-model family)")

        state = IRExecutionState(
            program_id=program.program_id,
            schema_version=program.schema_version,
            objects=objects,
            unresolved=unresolved,
            validated=True,
        )

        # v7.1: run simulation on the materialised part
        sim_result: SimulationResult | None = None
        if self.simulation is not None and objects:
            # Build a part dict from the materialised objects for simulation
            part = {
                "features": [
                    {
                        "type": obj.feature,
                        "params": obj.params,
                        "volume_m3": obj.volume_m3,
                        "mass_kg": obj.mass_kg,
                    }
                    for obj in objects
                ]
            }
            try:
                sim_result = self.simulation.run("structural", part=part)
            except Exception:
                # Simulation is optional; fail gracefully
                sim_result = SimulationResult(
                    analysis_type="structural",
                    passed=False,
                    messages=["simulation unavailable"],
                    model="first_order_analytic",
                )

        return IRExecutionResult(valid=True, errors=[], state=state, simulation=sim_result)


def execution_diff(
    before: CadProgram, after: CadProgram
) -> tuple[IrDiffReport, list[str]]:
    """Op-level diff of two revisions plus human-readable change lines."""
    report = ir_diff(before, after)
    lines = [f"added: {d['kind']} @ {d['position']}" for d in report.added]
    lines += [f"removed: {d['kind']} @ {d['position']}" for d in report.removed]
    lines += [
        f"changed: {d['kind']} @ {d['position']} ({', '.join(d['changed_params'])})"
        for d in report.changed
    ]
    return report, lines


__all__ = [
    "IRExecutionEngine",
    "IRExecutionResult",
    "IRExecutionState",
    "IRObjectState",
    "_primitive_parameters",
    "_topological_steps",
    "execution_diff",
]