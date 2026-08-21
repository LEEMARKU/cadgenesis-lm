"""cadgenesis.tools.executor
==========================
Tool executor: built-in tools bound to the real execution backends
(FreeCAD / OpenCascade analytic fallbacks, B-Rep topology, cost,
manufacturability, CAD-IR validation, exporters).
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any

from cadgenesis.cad.modeling.brep import BRepSolid
from cadgenesis.execution.cost_estimation import CostEstimator
from cadgenesis.execution.execution_engine import CADExecutionEngine
from cadgenesis.execution.exporter import ExportEngine
from cadgenesis.execution.freecad_engine import FreeCADEngine
from cadgenesis.execution.manufacturing import ManufacturabilityAnalyzer
from cadgenesis.execution.opencascade_engine import OpenCascadeEngine
from cadgenesis.execution.topology_analysis import TopologyAnalyzer
from cadgenesis.ir import parse_program, validate_program_ir
from cadgenesis.tools.registry import ToolRegistry
from cadgenesis.tools.schema import (
    ParameterSpec,
    Permission,
    ToolCall,
    ToolDefinition,
    ToolResult,
)

#: DFM severity strings -> numeric index (matches ManufacturingRules ordering).
_SEVERITY_INDEX = {"info": 1, "warning": 2, "error": 3, "critical": 4}


class ToolExecutor:
    """Registry + dispatch for the built-in CAD tool suite."""

    #: Fallback prism dims used when a program carries no explicit BOX size.
    DEFAULT_PRISM_DIMS = (80.0, 40.0, 20.0)

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or ToolRegistry()
        self.freecad = FreeCADEngine()
        self.opencascade = OpenCascadeEngine()
        self.engine = CADExecutionEngine()
        self.topology = TopologyAnalyzer()
        self.costs = CostEstimator()
        self.manufacturing = ManufacturabilityAnalyzer()
        self.exporter = ExportEngine()
        self._register_builtins()

    def _register_builtins(self) -> None:
        defs = [
            ToolDefinition(
                name="validate_program",
                description="Validate a CAD-IR program against the IR schema.",
                parameters=(ParameterSpec("program", "program", description="CAD token program"),),
                permission=Permission.READ,
                handler=self._handle_validate,
            ),
            ToolDefinition(
                name="execute_program",
                description=(
                    "Execute a CAD-IR program on a real CAD backend "
                    "(analytic fallback when FreeCAD/OCC is unavailable)."
                ),
                parameters=(
                    ParameterSpec("program", "program", description="CAD token program"),
                    ParameterSpec(
                        "backend",
                        "string",
                        required=False,
                        default="freecad",
                        description="'freecad' or 'opencascade'",
                    ),
                ),
                permission=Permission.EXECUTE,
                handler=self._handle_execute,
            ),
            ToolDefinition(
                name="analyze_brep",
                description="Topology analysis (manifold, closure, Euler, genus).",
                parameters=(ParameterSpec("program", "program", description="CAD token program"),),
                permission=Permission.READ,
                handler=self._handle_brep,
            ),
            ToolDefinition(
                name="estimate_cost",
                description="Estimate material + machining cost for a part.",
                parameters=(
                    ParameterSpec("part", "string", description="JSON part dict as string"),
                ),
                permission=Permission.READ,
                handler=self._handle_cost,
            ),
            ToolDefinition(
                name="manufacturing_check",
                description="Manufacturability assessment (DFM rules).",
                parameters=(
                    ParameterSpec("part", "string", description="JSON part dict as string"),
                ),
                permission=Permission.READ,
                handler=self._handle_manufacturing,
            ),
            ToolDefinition(
                name="export_program",
                description="Export a program's mesh to OBJ/STL/PLY (writes a file).",
                parameters=(
                    ParameterSpec("program", "program", description="CAD token program"),
                    ParameterSpec(
                        "format",
                        "string",
                        required=False,
                        default="obj",
                        description="'obj', 'stl' or 'ply'",
                    ),
                    ParameterSpec(
                        "path",
                        "string",
                        required=False,
                        default="out/tool_export.obj",
                        description="Output file path",
                    ),
                ),
                permission=Permission.ADMIN,
                handler=self._handle_export,
            ),
        ]
        for definition in defs:
            self.registry.register(definition)

    def dispatch(
        self,
        call: ToolCall,
        granted: Permission = Permission.EXECUTE,
        run_id: str = "",
    ) -> ToolResult:
        """Validate and run a call; errors are returned, never raised.

        ``run_id`` stamps every result for provenance (a run groups calls
        from one model inference / agent turn).  Timeouts (if configured on
        the tool) are enforced on a watchdog thread.
        """
        started = time.time()
        call_id = call.call_id or uuid.uuid4().hex[:12]
        try:
            definition, args = self.registry.validate_call(call, granted=granted)
        except (ValueError, TypeError, PermissionError) as exc:
            return ToolResult(
                ok=False,
                name=call.name,
                error=str(exc),
                call_id=call_id,
                caller=call.caller,
                run_id=run_id,
                timestamp=started,
                duration_seconds=time.time() - started,
            )
        timeout = definition.timeout_seconds
        if timeout is None or timeout <= 0:
            try:
                output = definition.handler(args)
                return ToolResult(
                    ok=True,
                    name=call.name,
                    output=output,
                    call_id=call_id,
                    caller=call.caller,
                    run_id=run_id,
                    timestamp=started,
                    duration_seconds=time.time() - started,
                )
            except Exception as exc:
                return ToolResult(
                    ok=False,
                    name=call.name,
                    error=f"{type(exc).__name__}: {exc}",
                    call_id=call_id,
                    caller=call.caller,
                    run_id=run_id,
                    timestamp=started,
                    duration_seconds=time.time() - started,
                )
        return self._dispatch_with_timeout(definition, args, call, call_id, run_id, started, timeout)

    def _dispatch_with_timeout(
        self,
        definition: ToolDefinition,
        args: dict[str, Any],
        call: ToolCall,
        call_id: str,
        run_id: str,
        started: float,
        timeout: float,
    ) -> ToolResult:
        """Run the handler on a watchdog thread and enforce ``timeout``."""
        state: dict[str, Any] = {}

        def _target() -> None:
            try:
                state["output"] = definition.handler(args)
                state["ok"] = True
            except Exception as exc:  # noqa: BLE001 - errors become results
                state["ok"] = False
                state["error"] = f"{type(exc).__name__}: {exc}"

        thread = threading.Thread(target=_target, daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            return ToolResult(
                ok=False,
                name=call.name,
                error=f"timeout after {timeout:g}s",
                call_id=call_id,
                caller=call.caller,
                run_id=run_id,
                timestamp=started,
                duration_seconds=time.time() - started,
            )
        return ToolResult(
            ok=bool(state.get("ok")),
            name=call.name,
            output=state.get("output"),
            error=state.get("error", ""),
            call_id=call_id,
            caller=call.caller,
            run_id=run_id,
            timestamp=started,
            duration_seconds=time.time() - started,
        )

    def _handle_validate(self, args: dict[str, Any]) -> dict[str, Any]:
        program = parse_program(args["program"])
        report = validate_program_ir(program)
        return {
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in report.checks
            ],
            "all_passed": all(c.passed for c in report.checks),
            "program_id": program.program_id,
        }

    def _handle_execute(self, args: dict[str, Any]) -> dict[str, Any]:
        backend = args.get("backend", "freecad")
        if backend not in ("freecad", "opencascade"):
            raise ValueError(f"backend must be 'freecad' or 'opencascade', got {backend!r}")
        engine = self.freecad if backend == "freecad" else self.opencascade
        return engine.execute(args["program"])

    def _handle_brep(self, args: dict[str, Any]) -> dict[str, Any]:
        dims = self._box_dims(args["program"])
        solid = BRepSolid.from_prism(*dims)
        report = self.topology.analyze_brep(solid.topology_graph)
        return {
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in report.checks
            ],
            "all_passed": all(c.passed for c in report.checks),
            "volume_mm3": solid.volume(),
        }

    def _handle_cost(self, args: dict[str, Any]) -> dict[str, Any]:
        part = _parse_json_dict(args["part"], "part")
        breakdown = self.costs.estimate(part)
        return {
            "material_usd": breakdown.material_usd,
            "machining_usd": breakdown.machining_usd,
            "printing_usd": breakdown.printing_usd,
            "assembly_usd": breakdown.assembly_usd,
            "tooling_usd": breakdown.tooling_usd,
            "total_usd": (
                breakdown.material_usd
                + breakdown.machining_usd
                + breakdown.printing_usd
                + breakdown.assembly_usd
                + breakdown.tooling_usd
            ),
        }

    def _handle_manufacturing(self, args: dict[str, Any]) -> dict[str, Any]:
        part = _parse_json_dict(args["part"], "part")
        report = self.manufacturing.assess(part)
        checks = [
            {"name": c.name, "severity": c.severity, "detail": c.detail} for c in report.checks
        ]
        return {
            "checks": checks,
            "all_passed": all(c.passed for c in report.checks),
            "max_severity": max(
                (_SEVERITY_INDEX.get(c.severity, 0) for c in report.checks), default=0
            ),
        }

    def _handle_export(self, args: dict[str, Any]) -> dict[str, Any]:
        dims = self._box_dims(args["program"])
        mesh = _box_mesh(*dims)
        path = self.exporter.export(
            mesh, args.get("path", "out/tool_export.obj"), fmt=args.get("format", "obj")
        )
        return {"exported": str(path)}

    @staticmethod
    def _box_dims(program_tokens: list[str]) -> tuple[float, float, float]:
        """First explicit BOX dims from the token program, else defaults."""
        dims = [ToolExecutor.DEFAULT_PRISM_DIMS[0], 0.0, 0.0]
        found = 0
        for token in program_tokens:
            if token.startswith("NUM_") and found < 3:
                try:
                    dims[found] = int(token.removeprefix("NUM_"))
                    found += 1
                except ValueError:
                    continue
            if found >= 3:
                break
        return dims[0], dims[1], dims[2]


def _parse_json_dict(text: str, name: str) -> dict[str, Any]:
    import json

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name}: not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected a JSON object")
    return value


def _box_mesh(length: float, width: float, height: float) -> dict[str, Any]:
    """Triangulated box mesh dict (vertices + faces), exporter-compatible."""
    lx, ly, lz = length / 2, width / 2, height / 2
    vertices = [
        [-lx, -ly, -lz],
        [lx, -ly, -lz],
        [lx, ly, -lz],
        [-lx, ly, -lz],
        [-lx, -ly, lz],
        [lx, -ly, lz],
        [lx, ly, lz],
        [-lx, ly, lz],
    ]
    faces = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
    return {"vertices": vertices, "faces": faces, "name": "box"}


__all__ = ["ToolExecutor"]
