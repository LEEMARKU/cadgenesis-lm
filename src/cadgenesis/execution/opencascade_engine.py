"""cadgenesis.execution.opencascade_engine
=======================================
Open CASCADE (OCC) execution backend (plugin interface with pure-Python
fallback).

When the optional ``OCC`` module (pythonOCC) is importable it drives the real
kernel; otherwise an analytic fallback evaluates the same program against the
`cad` substrate.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.cad.modeling.primitives import SolidPrimitive

BACKEND = "opencascade"


class OpenCascadeEngine:
    """OCC backend with analytic fallback (mirrors ``FreeCADEngine``)."""

    def __init__(self, backend: Any = None) -> None:
        self._backend = backend
        self._available: bool | None = None

    def available(self) -> bool:
        """True when a pythonOCC ``OCC`` module is importable."""
        if self._available is None:
            self._available = (
                self._backend is not None or importlib.util.find_spec("OCC") is not None
            )
        return self._available

    def execute(self, program: Any) -> dict[str, Any]:
        """Execute a program; analytic fallback when OCC is absent."""
        from cadgenesis.execution.freecad_engine import _primitives_from_program

        primitives = _primitives_from_program(program)
        if self.available():
            return {"backend": "opencascade", **self._backend_execute(primitives)}
        volumes = [p.volume() for p in primitives]
        return {
            "backend": "analytic",
            "status": "ok",
            "solid_count": len(primitives),
            "solids": [p.name for p in primitives],
            "volume_mm3": round(sum(volumes), 6),
            "mesh": [p.to_dict() for p in primitives],
        }

    def export(self, solid: Any, path: str, fmt: str) -> str:
        """Export a mesh to ``path`` in ``fmt`` via the export engine."""
        from cadgenesis.execution.exporter import ExportEngine

        mesh = solid if isinstance(solid, Mesh) else Mesh.from_dict(solid)
        return ExportEngine().export(mesh, path, fmt)

    def _backend_execute(self, primitives: list[SolidPrimitive]) -> dict[str, Any]:
        """Construct OCC solids when the pythonOCC module is present."""
        from OCC.Core.BRepPrimAPI import (  # type: ignore[import-not-found]
            BRepPrimAPI_MakeBox,
            BRepPrimAPI_MakeCylinder,
            BRepPrimAPI_MakeSphere,
        )

        shapes: list[Any] = []
        for primitive in primitives:
            if primitive.kind == "box":
                d = primitive.dims
                shapes.append(BRepPrimAPI_MakeBox(d["length"], d["width"], d["height"]).Shape())
            elif primitive.kind == "cylinder":
                shapes.append(
                    BRepPrimAPI_MakeCylinder(
                        primitive.dims["radius"], primitive.dims["height"]
                    ).Shape()
                )
            elif primitive.kind == "sphere":
                shapes.append(BRepPrimAPI_MakeSphere(primitive.dims["radius"]).Shape())
        return {
            "status": "ok",
            "solid_count": len(shapes),
            "volume_mm3": round(sum(p.volume() for p in primitives), 6),
        }

    def summary(self) -> dict[str, Any]:
        return {"backend": BACKEND, "available": self.available()}


__all__ = ["BACKEND", "OpenCascadeEngine"]
