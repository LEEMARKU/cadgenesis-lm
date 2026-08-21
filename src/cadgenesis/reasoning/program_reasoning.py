"""cadgenesis.reasoning.program_reasoning
========================================
Program-level reasoning bridge (M5).

Wires the CAD-IR :class:`~cadgenesis.ir.CadProgram` (M2) into the hybrid
neuro-symbolic stack (:mod:`cadgenesis.reasoning.hybrid`): primitives are
derived from decoded parameters, DFM constraints come from the toolkit's own
manufacturing thresholds, and infeasible systems can be repaired through
constraint relaxation.

    generated program -> CAD-IR -> hybrid reasoning -> report / repair

The pipeline stages behave as in :class:`HybridReasoningPipeline`:

* **rules** — DFM rules (min wall 0.8 mm, max size 1,000 mm, min hole 1.0 mm)
  consistent with :class:`~cadgenesis.reasoning.manufacturing_rules.ManufacturingRules`;
* **constraints** — every decoded parameter gets ``[0.8, 1000]`` feasibility
  bounds, solved by :class:`~cadgenesis.reasoning.constraint_solver.ConstraintSolver`;
* **geometry** — PRIM_* operations become :class:`Primitive` objects with the
  parameter order convention of the execution backends
  (``box`` = length/width/height, ``cylinder`` = radius/height, ...);
* **manufacturing** — DFM assessment on a ``part`` dict derived from the
  program's smallest dimension / hole parameters.

Repair is constraint-level relaxation (drop the least-important conflicting
constraint and re-solve).  Token-level rewriting of a program is deliberately
out of scope here: every parameter change would need to stay inside the
registered tokenizer vocabulary.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any

from cadgenesis.reasoning.constraint_solver import Constraint, ConstraintSolver, Variable
from cadgenesis.reasoning.geometry_reasoner import GeometryReasoner, Primitive
from cadgenesis.reasoning.hybrid import HybridReasoningPipeline, HybridReasoningReport
from cadgenesis.reasoning.manufacturing_rules import ManufacturingRules
from cadgenesis.reasoning.rule_engine import RuleEngine, make_rule

#: DFM thresholds (mm) shared by the rules, constraints and part assessment.
MIN_WALL_MM = 0.8
MAX_SIZE_MM = 1_000.0
MIN_HOLE_MM = 1.0

#: Default dimension values (mm) per geometry kind when a program omits them.
_DEFAULT_DIMS: dict[str, tuple[float, ...]] = {
    "box": (10.0, 10.0, 10.0),
    "cylinder": (5.0, 10.0),
    "sphere": (5.0,),
    "cone": (5.0, 10.0),
    "torus": (10.0, 5.0),
    "prism": (25.0, 10.0),
    "pyramid": (25.0, 10.0),
}

#: PRIM_* op -> geometry kind + parameter order (execution-engine convention).
_KIND_PARAM_ORDER: dict[str, tuple[str, ...]] = {
    "PRIM_BOX": ("length", "width", "height"),
    "PRIM_CYLINDER": ("radius", "height"),
    "PRIM_SPHERE": ("radius",),
    "PRIM_CONE": ("radius", "height"),
    "PRIM_TORUS": ("major_radius", "minor_radius"),
    "PRIM_PRISM": ("base_area", "height"),
    "PRIM_PYRAMID": ("base_area", "height"),
}

#: PRIM_* op -> GeometryReasoner kind.
_KIND_TO_GEOMETRY: dict[str, str] = {
    "PRIM_BOX": "box",
    "PRIM_CYLINDER": "cylinder",
    "PRIM_SPHERE": "sphere",
    "PRIM_CONE": "cone",
    "PRIM_TORUS": "torus",
    "PRIM_PRISM": "prism",
    "PRIM_PYRAMID": "pyramid",
}


def _params_of(op: Any) -> list[float]:
    """Positional decoded parameters (``d0``, ``d1``, ...) of an operation."""
    values: list[float] = []
    for key, value in sorted(op.params.items()):
        if key.startswith("d") and isinstance(value, (int, float)):
            values.append(float(value))
    return values


class ProgramReasoningEngine:
    """Reasons about (and optionally repairs) CAD-IR programs."""

    def __init__(
        self,
        material: str = "steel",
        processes: Iterable[str] | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.material = material
        self.processes = list(processes) if processes is not None else ["machining"]
        self.threshold = threshold
        self.geometry_reasoner = GeometryReasoner()
        self.constraint_solver = ConstraintSolver()
        self.manufacturing_rules = ManufacturingRules()
        self.rule_engine = RuleEngine()
        self.rule_engine.add_rules(
            [
                make_rule(
                    "wall_too_thin",
                    lambda ctx: ctx.get("min_dim", MIN_WALL_MM) < MIN_WALL_MM,
                    severity="error",
                ),
                make_rule(
                    "oversize_part",
                    lambda ctx: ctx.get("max_dim", 0.0) > MAX_SIZE_MM,
                    severity="error",
                ),
                make_rule(
                    "hole_too_small",
                    lambda ctx: (ctx.get("hole_diameter") or MIN_HOLE_MM) < MIN_HOLE_MM,
                    severity="warning",
                ),
            ]
        )
        self.pipeline = HybridReasoningPipeline(
            rule_engine=self.rule_engine,
            constraint_solver=self.constraint_solver,
            geometry_reasoner=self.geometry_reasoner,
            manufacturing_rules=self.manufacturing_rules,
            threshold=threshold,
        )
        # Token programs carry no spatial layout, so the built-in interference
        # stage is replaced by an honest "well-formed dimensions" check.
        self.pipeline.add_stage("geometry", self._geometry_stage)

    # ---------------------------------------------------------------- context

    def primitives(self, program: Any) -> list[Primitive]:
        """PRIM_* operations -> geometry :class:`Primitive` objects."""
        primitives: list[Primitive] = []
        for op in program.steps:
            geometry_kind = _KIND_TO_GEOMETRY.get(op.kind)
            if geometry_kind is None:
                continue
            order = _KIND_PARAM_ORDER[op.kind]
            values = _params_of(op)
            defaults = _DEFAULT_DIMS[geometry_kind]
            dims: dict[str, float] = {}
            for i, dim_name in enumerate(order):
                dims[dim_name] = values[i] if i < len(values) else defaults[i]
            primitives.append(Primitive(kind=geometry_kind, dims=dims, name=op.tokens[0]))
        return primitives

    def context_for(self, program: Any) -> dict[str, Any]:
        """Build the hybrid-reasoning context for a CAD-IR program."""
        primitives = self.primitives(program)
        all_params = [v for op in program.steps for v in _params_of(op)]
        min_dim = min(all_params) if all_params else MIN_WALL_MM
        max_dim = max(all_params) if all_params else 0.0

        variables: list[Variable] = []
        constraints: list[Constraint] = []
        for i, op in enumerate(program.steps):
            for j, value in enumerate(_params_of(op)):
                name = f"{op.tokens[0]}_{i}_{j}"
                variables.append(
                    Variable(name, initial=value, lower=MIN_WALL_MM, upper=MAX_SIZE_MM)
                )
                constraints.append(Constraint(f"{name}_min", {name: 1.0}, ">=", MIN_WALL_MM))
                constraints.append(Constraint(f"{name}_max", {name: 1.0}, "<=", MAX_SIZE_MM))

        hole_diameter: float | None = None
        for op in program.steps:
            if op.kind in ("FEAT_HOLE", "FEAT_COUNTERBORE", "FEAT_THREAD"):
                values = _params_of(op)
                if values:
                    hole_diameter = values[0]
                    break

        part: dict[str, Any] = {
            "material": self.material,
            "processes": list(self.processes),
            "min_wall_thickness": min_dim,
        }
        if hole_diameter is not None:
            part["hole_diameter"] = hole_diameter

        return {
            "id": program.program_id,
            "query": program.steps[0].tokens[0] if program.steps else "",
            "min_dim": min_dim,
            "max_dim": max_dim,
            "hole_diameter": hole_diameter,
            "constraint_variables": variables,
            "constraints": constraints,
            "geometry_primitives": primitives,
            "part": part,
        }

    def _geometry_stage(self, context: dict[str, Any]) -> bool:
        """Custom geometry stage: every primitive must be well-formed.

        Interference cannot be assessed for flat token programs (no layout
        information), so this stage validates positive/finite dimensions only
        and records that fact in the report detail.
        """
        primitives = context.get("geometry_primitives") or []
        return all(self.geometry_reasoner.validate(p).valid for p in primitives)

    # ----------------------------------------------------------------- reason

    def reason(self, program: Any) -> HybridReasoningReport:
        """Run the hybrid reasoning pipeline over a program."""
        return self.pipeline.reason(self.context_for(program))

    def repair(
        self,
        program: Any,
        extra_constraints: Iterable[Constraint] | None = None,
    ) -> dict[str, Any]:
        """Relax infeasible parameter constraints and re-solve.

        The bridge's own bounds constraints are always satisfiable, so
        ``dropped == []`` for them; caller-supplied ``extra_constraints``
        (e.g. design rules that fight the bounds) are dropped in residual
        order when they conflict.  Returns ``{"feasible", "dropped",
        "assignment", "messages"}`` from :meth:`ConstraintSolver.repair`.
        """
        context = self.context_for(program)
        constraints = [*context["constraints"]]
        if extra_constraints:
            constraints.extend(extra_constraints)
        return self.constraint_solver.repair(context["constraint_variables"], constraints)

    # -------------------------------------------------------------- benchmark

    def benchmark(self, programs: Iterable[Any]) -> dict[str, Any]:
        """Reason over many programs and aggregate measured statistics.

        Returns pass rate, per-stage failure counts and timing.
        """
        programs = list(programs)
        passed = 0
        stage_failures: dict[str, int] = {}
        total_ms = 0.0
        for program in programs:
            t0 = time.perf_counter()
            report = self.reason(program)
            total_ms += (time.perf_counter() - t0) * 1000.0
            if report.passed:
                passed += 1
            for stage in report.stages:
                if not stage.passed:
                    stage_failures[stage.stage] = stage_failures.get(stage.stage, 0) + 1
        n = len(programs)
        return {
            "n": n,
            "passed": passed,
            "pass_rate": passed / n if n else 0.0,
            "stage_failures": stage_failures,
            "total_ms": round(total_ms, 3),
            "mean_ms": round(total_ms / n, 3) if n else 0.0,
        }


__all__ = [
    "MAX_SIZE_MM",
    "MIN_HOLE_MM",
    "MIN_WALL_MM",
    "ProgramReasoningEngine",
]
