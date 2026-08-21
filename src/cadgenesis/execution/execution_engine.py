"""
cadgenesis.execution.execution_engine
===================================
CAD Execution Intelligence Engine for CADGenesis-LM v2.0:
- Parametric JSON Feature Tree Generator
- B-Rep Topology Analyzer
- Geometry & Validity Verifier
- Manufacturing & DFM Analyzer
- Cost Estimator
- Optimization & Feedback loop provider

Pillar 8 extends the engine with a full execution pipeline
(intent -> program -> execute -> validate -> simulate -> optimize -> repair
-> export -> feedback) while keeping the original ``execute_and_evaluate``
token-prefix behavior intact as a compatibility path.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.execution.cost_estimation import CostEstimator
from cadgenesis.execution.feedback import FeedbackLoop
from cadgenesis.execution.geometry_validation import GeometryValidator
from cadgenesis.execution.manufacturing import ManufacturabilityAnalyzer
from cadgenesis.execution.optimization import OptimizationEngine
from cadgenesis.execution.simulation import SimulationEngine
from cadgenesis.execution.topology_analysis import TopologyAnalyzer
from cadgenesis.ir.program import CadProgram


@dataclass
class CADExecutionResult:
    """Detailed output result of the CAD Execution Engine execution pipeline."""

    is_valid_geometry: bool = True
    is_manufacturable: bool = True
    safety_factor: float = 2.5
    estimated_cost_usd: float = 45.0
    confidence_score: float = 0.95
    errors: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    parametric_json: dict[str, Any] = field(default_factory=dict)
    # Pillar 8: pipeline reports (additive, all defaulted)
    geometry_report: dict[str, Any] = field(default_factory=dict)
    topology_report: dict[str, Any] = field(default_factory=dict)
    manufacturing_report: dict[str, Any] = field(default_factory=dict)
    simulation_report: dict[str, Any] = field(default_factory=dict)
    optimization_report: dict[str, Any] = field(default_factory=dict)
    repair_report: dict[str, Any] = field(default_factory=dict)
    cost_breakdown: dict[str, float] = field(default_factory=dict)
    exports: list[str] = field(default_factory=list)
    # v6.4: IR-native execution report (typed program graph -> world state)
    ir_report: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Compact serializable summary."""
        return {
            "is_valid_geometry": self.is_valid_geometry,
            "is_manufacturable": self.is_manufacturable,
            "safety_factor": self.safety_factor,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "confidence_score": round(self.confidence_score, 4),
            "errors": list(self.errors),
            "suggestions": list(self.suggestions),
        }

    def to_dict(self) -> dict[str, Any]:
        """Full serializable snapshot (all fields)."""
        return {
            "is_valid_geometry": self.is_valid_geometry,
            "is_manufacturable": self.is_manufacturable,
            "safety_factor": self.safety_factor,
            "estimated_cost_usd": self.estimated_cost_usd,
            "confidence_score": self.confidence_score,
            "errors": list(self.errors),
            "suggestions": list(self.suggestions),
            "parametric_json": dict(self.parametric_json),
            "geometry_report": dict(self.geometry_report),
            "topology_report": dict(self.topology_report),
            "manufacturing_report": dict(self.manufacturing_report),
            "simulation_report": dict(self.simulation_report),
            "optimization_report": dict(self.optimization_report),
            "repair_report": dict(self.repair_report),
            "cost_breakdown": dict(self.cost_breakdown),
            "exports": list(self.exports),
            "ir_report": dict(self.ir_report),
        }


class CADExecutionEngine:
    """
    Execution Intelligence Engine providing the full evaluation pipeline:
    Generated Tokens -> Parametric JSON -> Topology & Geometry Validation
    -> DFM & Cost Analysis -> Feedback Scores to Model.

    ``execute_and_evaluate`` (legacy token path) is preserved unchanged;
    ``execute`` runs the full Pillar 8 pipeline over designs and programs.
    """

    def __init__(
        self,
        validator: GeometryValidator | None = None,
        topology: TopologyAnalyzer | None = None,
        manufacturing: ManufacturabilityAnalyzer | None = None,
        simulation: SimulationEngine | None = None,
        optimization: OptimizationEngine | None = None,
        estimator: CostEstimator | None = None,
        feedback: FeedbackLoop | None = None,
    ) -> None:
        self.validator = validator or GeometryValidator()
        self.topology = topology or TopologyAnalyzer()
        self.manufacturing = manufacturing or ManufacturabilityAnalyzer()
        self.simulation = simulation or SimulationEngine()
        self.optimization = optimization or OptimizationEngine()
        self.estimator = estimator or CostEstimator()
        self.feedback = feedback or FeedbackLoop()

    def execute_and_evaluate(self, cad_tokens: list[str]) -> CADExecutionResult:
        """
        Executes the CAD verification pipeline over a generated token stream.
        Accepts a flat token list or a typed CAD-IR program.
        """
        if hasattr(cad_tokens, "to_tokens"):
            program_id = getattr(cad_tokens, "program_id", None)
            cad_tokens = cad_tokens.to_tokens()
        else:
            program_id = None
        result = CADExecutionResult()

        if not cad_tokens:
            result.is_valid_geometry = False
            result.errors.append("Empty token sequence")
            return result

        # Basic shape detection & parametric JSON construction
        head = cad_tokens[0] if cad_tokens else "UNKNOWN"
        result.parametric_json = {"primitive": head, "tokens": cad_tokens}
        if program_id is not None:
            result.parametric_json["program_id"] = program_id

        if "BOX" in head or "PRIM_BOX" in head:
            result.estimated_cost_usd = 25.0
            result.suggestions.append("Consider adding 1mm fillets to sharp external edges.")
        elif "CYLINDER" in head or "PRIM_CYLINDER" in head:
            result.estimated_cost_usd = 35.0
        else:
            result.estimated_cost_usd = 50.0

        return result

    # -------------------------------------------------------- pipeline (P8)

    def execute(
        self,
        program: Sequence[str] | None = None,
        design: dict[str, Any] | None = None,
        validate: bool = True,
        simulate: bool = True,
        optimize: bool = True,
        export_fmt: str | None = None,
        export_path: str | None = None,
        memory: Any = None,
        memory_key: str = "exec:latest",
    ) -> CADExecutionResult:
        """Run the full execution pipeline and return a :class:`CADExecutionResult`.

        ``program`` is a token stream (legacy-compatible); ``design`` is a
        part descriptor dict consumed by the analytic validators.  When
        ``export_fmt`` is given the design mesh is exported (STL/OBJ/PLY/
        GLTF/DXF/...).  ``memory`` optionally receives the result via
        ``memory.remember(pool, key, content)``.
        """
        result = CADExecutionResult()
        if program is not None:
            result = self.execute_and_evaluate(list(program))
        result.parametric_json.setdefault("program", list(program) if program else [])

        part = design or {}
        mesh = self._design_mesh(part)

        if validate:
            geometry = self.validator.validate_design(_DesignView(part, mesh))
            result.geometry_report = geometry.summary()
            result.is_valid_geometry = geometry.valid

            if mesh is not None:
                topology = self.topology.analyze_mesh(mesh)
                result.topology_report = topology.summary()
            elif part:
                result.topology_report = {}

        if part:
            manufacturing = self.manufacturing.assess(part)
            result.manufacturing_report = manufacturing.summary()
            result.is_manufacturable = manufacturing.passed
            cost = self.estimator.estimate(part)
            result.cost_breakdown = cost.to_dict()
            if program is None:
                result.estimated_cost_usd = cost.total
            self._apply_part_safety(result, part)
        else:
            cost = self.estimator.estimate(
                {
                    "processes": [],
                    "feature_count": len(program or []),
                    "volume_m3": (mesh.volume() * 1e-9) if mesh is not None else None,
                }
            )
            result.cost_breakdown = cost.to_dict()

        if simulate and part:
            analysis = part.get("analysis") or {}
            if analysis.get("type"):
                sim = self.simulation.run(
                    analysis["type"],
                    part={
                        **part,
                        "material": part.get("material") or {},
                        "load": analysis.get("load") or {},
                    },
                )
            else:
                sim = self.simulation.structural(part)
            result.simulation_report = sim.summary()
            fos = sim.values.get("factor_of_safety")
            if fos is not None and 0.0 < fos < 1e6:
                result.safety_factor = float(fos)

        if optimize and part:
            optimization = self.optimization.optimize(part)
            result.optimization_report = optimization.summary()
            result.suggestions.extend(optimization.suggestions)

        if export_fmt is not None and mesh is not None:
            from cadgenesis.execution.exporter import ExportEngine

            path = export_path or f"exported.{export_fmt}"
            result.exports.append(ExportEngine().export(mesh, path, export_fmt))

        result.confidence_score = self.compute_confidence(result)

        if part and not result.is_manufacturable:
            self._apply_repair(result, part)

        self.feedback.apply(result, self._feedback_reports(result))
        result.suggestions = list(dict.fromkeys(result.suggestions))

        if memory is not None and part:
            with suppress(TypeError, ValueError, KeyError, AttributeError):
                memory.remember("project", memory_key, result.to_dict())
        return result

    def compute_confidence(self, result: CADExecutionResult) -> float:
        """Aggregate confidence from pipeline flags (0..1)."""
        flags = [
            result.is_valid_geometry,
            result.is_manufacturable,
            not result.errors,
        ]
        passed = sum(bool(f) for f in flags)
        return round(0.5 + 0.5 * (passed / len(flags)), 4)

    # ------------------------------------------------------- IR-native (v6.4)

    def execute_ir(
        self,
        program: CadProgram,
        previous: CadProgram | None = None,
        vocab=None,
        material: Any = None,
        memory: Any = None,
        memory_key: str = "exec:ir:latest",
    ) -> CADExecutionResult:
        """Execute a typed IR program directly (no token-stream round-trip).

        Materialises each operation into the world model via
        :class:`~cadgenesis.execution.ir_execution.IRExecutionEngine`,
        optionally diffs the revision against ``previous`` (feedback loop
        items folded into ``suggestions``), and estimates cost from the
        materialised volume.  ``vocab`` gates token registration (mini vs
        default dialect).  ``material`` supplies density for mass/cost.
        """
        from cadgenesis.execution.ir_execution import IRExecutionEngine, execution_diff

        if not isinstance(program, CadProgram):
            raise TypeError(f"expected CadProgram, got {type(program).__name__}")

        result = CADExecutionResult()
        result.parametric_json = {
            "program_id": program.program_id,
            "schema_version": program.schema_version,
        }

        ir_engine = IRExecutionEngine(vocab=vocab, material=material)
        ir_result = ir_engine.execute(program)

        if not ir_result.valid:
            result.is_valid_geometry = False
            result.is_manufacturable = False
            result.confidence_score = 0.0
            result.errors = [f"[ir] {e}" for e in ir_result.errors]
            result.ir_report = {
                "valid": False,
                "program_id": program.program_id,
                "errors": list(ir_result.errors),
            }
            if memory is not None:
                with suppress(TypeError, ValueError, KeyError, AttributeError):
                    memory.remember("project", memory_key, result.to_dict())
            return result

        state = ir_result.state
        assert state is not None

        diff_data = None
        if previous is not None:
            if not isinstance(previous, CadProgram):
                raise TypeError(f"expected CadProgram for previous, got {type(previous).__name__}")
            diff_report, _ = execution_diff(previous, program)
            diff_data = diff_report.to_dict()
            for item in self.feedback.feedback_on_diff(diff_report):
                result.suggestions.append(f"[{item.source}] {item.message}")

        material_dict: dict[str, Any] = {}
        if material is not None:
            material_dict = {
                "name": getattr(material, "name", str(material)),
                "density_kg_m3": getattr(material, "density_kg_m3", 0.0),
            }
        cost = self.estimator.estimate(
            {
                "volume_m3": state.total_volume(),
                "feature_count": len(state.objects),
                "material": material_dict,
                "processes": [],
            }
        )
        result.cost_breakdown = cost.to_dict()
        result.estimated_cost_usd = cost.total
        result.ir_report = {
            "valid": True,
            "program_id": program.program_id,
            "schema_version": program.schema_version,
            "objects": len(state.objects),
            "unresolved": list(state.unresolved),
            "total_volume_m3": state.total_volume(),
            "total_mass_kg": state.total_mass(),
            "bounds": state.bounds(),
            "diff": diff_data,
            "state": state.to_dict(),
        }
        result.confidence_score = self.compute_confidence(result)
        result.suggestions = list(dict.fromkeys(result.suggestions))

        if memory is not None:
            with suppress(TypeError, ValueError, KeyError, AttributeError):
                memory.remember("project", memory_key, result.to_dict())
        return result

    def execute_assembly(
        self,
        assembly: dict[str, Any],
        constraints: Sequence[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Validate an assembly dict: references, mates, mobility."""
        from cadgenesis.cad.assembly.mates import AssemblyConstraint, MateSolver

        names = list(assembly.get("parts") or [])
        mates = [AssemblyConstraint.from_dict(raw) for raw in constraints or []]
        solver = MateSolver()
        analysis = solver.analyze_assembly(names, mates)
        rigid = solver.is_rigid(names, mates)
        report = {
            "parts": len(names),
            "mates": len(mates),
            "rigid": bool(rigid),
            "dof_total": solver.total_dof(names, mates),
            "per_component": [a.to_dict() for a in analysis],
        }
        return report

    def simulate_mechanism(self, mechanism: Any, states: dict[str, float]) -> dict[str, Any]:
        """Motion simulation for a mechanism (analytic forward kinematics)."""
        return self.simulation.motion(mechanism, states).summary()

    def analyze_strength(
        self,
        part: dict[str, Any],
        force_n: float,
        area_m2: float | None = None,
    ) -> dict[str, Any]:
        """Axial strength check for a part descriptor."""
        load = {"force_n": force_n}
        if area_m2 is not None:
            load["area_m2"] = area_m2
        return self.simulation.structural(
            {**part, "load": {**part.get("load", {}), **load}}
        ).summary()

    # ------------------------------------------------------------ internals

    def _design_mesh(self, part: dict[str, Any]) -> Mesh | None:
        mesh = part.get("mesh")
        if isinstance(mesh, Mesh):
            return mesh
        if isinstance(mesh, dict):
            with suppress(TypeError, ValueError):
                return Mesh.from_dict(mesh)
        return None

    def _apply_part_safety(self, result: CADExecutionResult, part: dict[str, Any]) -> None:
        material = part.get("material") or {}
        safety = material.get("target_safety_factor")
        if safety is not None:
            with suppress(TypeError, ValueError):
                result.safety_factor = float(safety)

    def _apply_repair(self, result: CADExecutionResult, part: dict[str, Any]) -> None:
        """Layered repair attempt over mesh repair for non-manufacturable parts."""
        from cadgenesis.cad.mesh.repair import (
            diagnose,
            fill_holes,
            remove_duplicate_vertices,
        )

        mesh = self._design_mesh(part)
        if mesh is None:
            result.repair_report = {"attempted": False, "message": "no mesh to repair"}
            return
        diagnosis = diagnose(mesh)
        repaired = fill_holes(
            remove_duplicate_vertices(mesh),
            max_hole_edges=12,
        )
        after = diagnose(repaired)
        fixed = after["watertight"] and not diagnosis["watertight"]
        result.repair_report = {
            "attempted": True,
            "before": diagnosis,
            "after": after,
            "fixed": fixed,
        }
        if fixed:
            result.is_valid_geometry = True
            result.errors.append("geometry repaired (boundary loops filled)")

    def _feedback_reports(self, result: CADExecutionResult) -> dict[str, Any]:
        return {
            "geometry": result.geometry_report,
            "topology": result.topology_report,
            "manufacturing": result.manufacturing_report,
            "simulation": result.simulation_report,
            "optimization": result.optimization_report,
        }


class _DesignView:
    """Duck-typed design facade for ``GeometryValidator.validate_design``."""

    def __init__(self, part: dict[str, Any], mesh: Mesh | None) -> None:
        self.mesh = mesh
        self.faces = None
        self.vertices = None
        if mesh is None:
            faces = part.get("faces")
            vertices = part.get("vertices")
            if isinstance(faces, list) and isinstance(vertices, list):
                self.faces = faces
                self.vertices = vertices


__all__ = ["CADExecutionEngine", "CADExecutionResult"]
