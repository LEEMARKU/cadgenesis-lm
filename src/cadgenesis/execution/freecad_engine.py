"""cadgenesis.execution.freecad_engine
===================================
FreeCAD execution backend (plugin interface with pure-Python fallback).

The backend is optional: when the real ``FreeCAD`` module is available it is
used for solid construction; otherwise a pure-Python analytic fallback
evaluates the same program against the `cad` substrate.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.cad.modeling.primitives import SolidPrimitive

BACKEND = "freecad"


class FreeCADEngine:
    """FreeCAD backend with analytic fallback.

    ``execute(program)`` accepts a token program (list of ``PRIM_*``/feature
    tokens) or a list of primitive dicts and returns an execution summary.
    """

    def __init__(self, backend: Any = None) -> None:
        self._backend = backend
        self._available: bool | None = None

    def available(self) -> bool:
        """True when a real FreeCAD module is importable."""
        if self._available is None:
            self._available = (
                self._backend is not None or importlib.util.find_spec("FreeCAD") is not None
            )
        return self._available

    # ----------------------------------------------------------------- exec

    def execute(self, program: Any) -> dict[str, Any]:
        """Execute a program; analytic fallback when FreeCAD is absent."""
        primitives = _primitives_from_program(program)
        solid = (
            self._backend_execute(primitives)
            if self.available()
            else self._analytic_execute(primitives)
        )
        return {"backend": "freecad" if self.available() else "analytic", **solid}

    def export(self, solid: Any, path: str, fmt: str) -> str:
        """Export a mesh to ``path`` in ``fmt`` via the export engine."""
        from cadgenesis.execution.exporter import ExportEngine

        mesh = solid if isinstance(solid, Mesh) else Mesh.from_dict(solid)
        return ExportEngine().export(mesh, path, fmt)

    # ------------------------------------------------------------ internals

    def _backend_execute(self, primitives: list[SolidPrimitive]) -> dict[str, Any]:
        """Construct real FreeCAD solids when the module is present."""
        import FreeCAD  # type: ignore[import-not-found]

        doc = FreeCAD.newDocument("cadgenesis")
        names: list[str] = []
        for primitive in primitives:
            name = _freecad_feature(doc, primitive)
            if name:
                names.append(name)
        return {
            "status": "ok",
            "solid_count": len(names),
            "solids": names,
            "volume_mm3": round(sum(p.volume() for p in primitives), 6),
        }

    def _analytic_execute(self, primitives: list[SolidPrimitive]) -> dict[str, Any]:
        volumes = [p.volume() for p in primitives]
        return {
            "status": "ok",
            "solid_count": len(primitives),
            "solids": [p.name for p in primitives],
            "volume_mm3": round(sum(volumes), 6),
            "mesh": [p.to_dict() for p in primitives],
        }

    def summary(self) -> dict[str, Any]:
        return {"backend": BACKEND, "available": self.available()}


def _primitives_from_program(program: Any) -> list[SolidPrimitive]:
    """Normalize a token program, primitive-dict list, or CAD-IR program
    into solids."""
    if hasattr(program, "to_tokens"):
        program = program.to_tokens()
    if isinstance(program, list):
        solids: list[SolidPrimitive] = []
        for entry in program:
            if isinstance(entry, SolidPrimitive):
                solids.append(entry)
            elif isinstance(entry, dict):
                solids.append(SolidPrimitive.from_dict(entry))
            else:
                token = str(entry).upper()
                if "BOX" in token:
                    dims = {"length": 10.0, "width": 10.0, "height": 10.0}
                    solids.append(SolidPrimitive("box", dims))
                elif "CYLINDER" in token:
                    solids.append(SolidPrimitive("cylinder", {"radius": 5.0, "height": 10.0}))
                elif "SPHERE" in token:
                    solids.append(SolidPrimitive("sphere", {"radius": 5.0}))
        return solids
    if isinstance(program, dict):
        return [SolidPrimitive.from_dict(program)]
    return []


def _freecad_feature(doc: Any, primitive: SolidPrimitive) -> str:
    """Create one FreeCAD object for a solid primitive."""
    import FreeCAD  # type: ignore[import-not-found]

    if primitive.kind == "box":
        obj = doc.addObject("Part::Box", primitive.name)
        obj.Length = primitive.dims["length"]
        obj.Width = primitive.dims["width"]
        obj.Height = primitive.dims["height"]
        return primitive.name
    if primitive.kind == "cylinder":
        obj = doc.addObject("Part::Cylinder", primitive.name)
        obj.Radius = primitive.dims["radius"]
        obj.Height = primitive.dims["height"]
        return primitive.name
    if primitive.kind == "sphere":
        obj = doc.addObject("Part::Sphere", primitive.name)
        obj.Radius = primitive.dims["radius"]
        return primitive.name
    obj = doc.addObject("Part::Feature", primitive.name)
    obj.Shape = FreeCAD.Part.Shape()
    return primitive.name


__all__ = ["BACKEND", "FreeCADEngine"]
