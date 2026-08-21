"""cadgenesis.execution.exporter
===============================
Export engine for the CAD execution pipeline.

Real mesh exporters (STL binary/ASCII, OBJ, PLY via ``cad.mesh.io``), an
analytic GLTF JSON manifest, DXF polylines, structured neutral CAD formats
(STEP/IGES/Parasolid) and script/manifest exporters for DWG, Fusion 360,
SolidWorks, FreeCAD and OpenSCAD.  Binary/proprietary formats get documented
structured fallbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cadgenesis.cad.mesh.io import read_obj, read_ply, read_stl, write_obj, write_ply, write_stl
from cadgenesis.cad.mesh.mesh import Mesh

REAL_FORMATS = ("stl", "obj", "ply")
STRUCTURED_FORMATS = ("gltf", "dxf", "step", "iges", "parasolid")
SCRIPT_FORMATS = ("dwg", "fusion360", "solidworks", "freecad", "openscad")
ALL_FORMATS = (*REAL_FORMATS, *STRUCTURED_FORMATS, *SCRIPT_FORMATS)


class ExportEngine:
    """Mesh/design export to CAD and 3D formats.

    ``export()`` writes to a path (created on demand) and returns it;
    ``to_text()`` returns the payload string for formats that are textual.
    """

    def __init__(self) -> None:
        self._readers = {
            "stl": read_stl,
            "obj": read_obj,
            "ply": read_ply,
        }

    def supported(self) -> list[str]:
        return list(ALL_FORMATS)

    # ---------------------------------------------------------------- export

    def export(
        self,
        mesh: Mesh | dict[str, Any],
        path: str | Path,
        fmt: str,
        binary: bool = True,
    ) -> str:
        """Write ``mesh`` to ``path`` in ``fmt``; returns the written path."""
        fmt = fmt.lower().lstrip(".")
        if fmt not in ALL_FORMATS:
            raise ValueError(f"unsupported export format {fmt!r}")
        solid = self._as_mesh(mesh)
        target = Path(path)
        if target.suffix == "":
            target = target.with_suffix(f".{fmt}")
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        if fmt in REAL_FORMATS:
            writer = {
                "stl": lambda: write_stl(solid, target, binary=binary),
                "obj": lambda: write_obj(solid, target),
                "ply": lambda: write_ply(solid, target, binary=binary),
            }[fmt]
            writer()
            return str(target)
        text = self.to_text(solid, fmt)
        target.write_text(text, encoding="utf-8")
        return str(target)

    def to_text(self, mesh: Mesh, fmt: str) -> str:
        """Render a mesh as text in ``fmt`` (validated by ``export`` too)."""
        fmt = fmt.lower().lstrip(".")
        if fmt not in ALL_FORMATS:
            raise ValueError(f"unsupported export format {fmt!r}")
        if fmt == "gltf":
            return _gltf(mesh)
        if fmt == "dxf":
            return _dxf(mesh)
        if fmt == "step":
            return _step(mesh)
        if fmt == "iges":
            return _iges(mesh)
        if fmt == "parasolid":
            return _parasolid(mesh)
        if fmt in SCRIPT_FORMATS:
            return _script(mesh, fmt)
        return _text_fallback(mesh, fmt)

    # ------------------------------------------------------------------ misc

    def read(self, path: str | Path, fmt: str) -> Mesh:
        """Read a mesh back from disk for the real formats."""
        fmt = fmt.lower().lstrip(".")
        reader = self._readers.get(fmt)
        if reader is None:
            raise ValueError(f"no reader for format {fmt!r}")
        return reader(Path(path))

    def _as_mesh(self, mesh: Mesh | dict[str, Any]) -> Mesh:
        if isinstance(mesh, Mesh):
            return mesh
        if isinstance(mesh, dict):
            return Mesh.from_dict(mesh)
        raise TypeError("expected Mesh or mesh dict")

    def summary(self) -> dict[str, Any]:
        return {
            "real_formats": list(REAL_FORMATS),
            "structured_formats": list(STRUCTURED_FORMATS),
            "script_formats": list(SCRIPT_FORMATS),
        }


# ------------------------------------------------------------- text renderers


def _gltf(mesh: Mesh) -> str:
    """Analytic glTF 2.0 JSON manifest (positions + indices, no buffers)."""
    positions = [round(v, 6) for point in mesh.vertices for v in (point.x, point.y, point.z)]
    indices = [i for face in mesh.faces for i in face]
    return (
        "{\n"
        '  "asset": {"version": "2.0", "generator": "cadgenesis-export"},'
        "\n"
        '  "scene": 0, "scenes": [{"nodes": [0]}],'
        "\n"
        '  "nodes": [{"mesh": 0, "name": "' + mesh.name + '"}],\n'
        '  "meshes": [{"primitives": [{"attributes": {"POSITION": 0},'
        ' "indices": 1, "mode": 4}]}],\n'
        f'  "buffers": [{{"byteLength": {len(positions) * 4 + len(indices) * 4}}}],\n'
        f'  "bufferViews": [{{"buffer": 0, "byteOffset": 0, '
        f'"byteLength": {len(positions) * 4}}}, '
        f'{{"buffer": 0, "byteOffset": {len(positions) * 4}, '
        f'"byteLength": {len(indices) * 4}}}],\n'
        f'  "accessors": [{{"bufferView": 0, "componentType": 5126, '
        f'"count": {len(mesh.vertices)}, "type": "VEC3"}}, '
        f'{{"bufferView": 1, "componentType": 5123, "count": {len(indices)}, '
        f'"type": "SCALAR"}}]\n'
        "}\n"
    )


def _dxf(mesh: Mesh) -> str:
    """ASCII DXF with a 3DFACE entity per triangle (documented structured form)."""
    lines = ["0", "SECTION", "2", "ENTITIES"]
    for face in mesh.faces:
        points = [mesh.vertices[i] for i in face]
        lines.extend(["0", "3DFACE", "8", "MESH"])
        for index, point in enumerate(points):
            code = 10 + index * 10
            lines.extend(
                [
                    str(c)
                    for pair in zip(
                        (code, code + 1, code + 2),
                        (point.x, point.y, point.z),
                        strict=True,
                    )
                    for c in pair
                ]
            )
    lines.extend(["0", "ENDSEC", "0", "EOF"])
    return "\n".join(lines) + "\n"


def _step(mesh: Mesh) -> str:
    """ISO 10303-21 neutral STEP manifest with analytic shell entities."""
    entries: list[str] = []
    for face_index, face in enumerate(mesh.faces):
        coords = ", ".join(
            f"({_num(mesh.vertices[i].x)},{_num(mesh.vertices[i].y)},{_num(mesh.vertices[i].z)})"
            for i in face
        )
        entries.append(f"#{face_index + 1}=TRIANGULAR_FACE('f{face_index}',({coords}),.T.);")
    body = ";".join(entries)
    return (
        "ISO-10303-21;\n"
        "HEADER;\n"
        f"FILE_DESCRIPTION(('CADGenesis analytic export'),'2;1');\n"
        f"FILE_NAME('{mesh.name}.step','2026-08-06T00:00:00',('CADGenesis'),"
        f"('CADGenesis'),'cadgenesis','cadgenesis','');\n"
        "FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));\n"
        "ENDSEC;\n"
        "DATA;\n"
        f"{body}\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n"
    )


def _iges(mesh: Mesh) -> str:
    """IGES structured manifest: 100-series plane points + 190 entity records."""
    points: list[str] = []
    for index, point in enumerate(mesh.vertices):
        points.append(
            f"{index + 1},100,0,0,1,0,0,0,{_num(point.x)},"
            f"{_num(point.y)},{_num(point.z)},1,{index + 1},0,0;"
        )
    faces: list[str] = []
    for face_index, face in enumerate(mesh.faces):
        faces.append(
            f"{200 + face_index},190,0,0,1,0,0,0,3,0,"
            f"{face[0] + 1},{face[1] + 1},{face[2] + 1},"
            f"{200 + face_index},0,0;"
        )
    return (
        "S      1\n"
        f"G      1CADGenesis analytic IGES manifest,{mesh.name};"
        "\n"
        f"G      2,1,1,20260806,0,1.0,1,4,15,1,1,0.1;"
        "\n" + "\n".join(points) + "\n" + "\n".join(faces) + "\n"
        "S      1G      2D      0E      0\nT      1\n"
    )


def _parasolid(mesh: Mesh) -> str:
    """Parasolid XT structured manifest (analytic facet bodies)."""
    vertices = ", ".join(f"POINT({_num(v.x)},{_num(v.y)},{_num(v.z)})" for v in mesh.vertices)
    faces = ", ".join(f"FACET({f[0]},{f[1]},{f[2]})" for f in mesh.faces)
    return f"PARASOLID XT 30.0;\nBODY('{mesh.name}', {vertices}, {faces});\nEND;\n"


def _script(mesh: Mesh, fmt: str) -> str:
    """Script/manifest exporters for proprietary and parametric tools."""
    if fmt == "openscad":
        return _openscad(mesh)
    if fmt == "freecad":
        return _freecad(mesh)
    if fmt == "dwg":
        return _manifest(mesh, "DWG/AutoCAD", "binary DWG; analytic polyline manifest")
    if fmt == "fusion360":
        return _manifest(mesh, "Autodesk Fusion 360", "binary F3D; JSON design manifest")
    if fmt == "solidworks":
        return _manifest(mesh, "SolidWorks", "binary SLDPRT; parametric manifest")
    return _manifest(mesh, fmt, "structured manifest")


def _openscad(mesh: Mesh) -> str:
    lines = [
        "// CADGenesis analytic OpenSCAD export",
        f"// {mesh.name}",
        "module part() {",
        "  polyhedron(",
        "    points=[",
    ]
    lines.extend(f"      [{_num(v.x)},{_num(v.y)},{_num(v.z)}]," for v in mesh.vertices)
    lines.append("    ],")
    lines.append("    faces=[")
    lines.extend(f"      [{f[0]},{f[1]},{f[2]}]," for f in mesh.faces)
    lines.append("    ], convexity=10);")
    lines.append("}")
    lines.append("part();")
    return "\n".join(lines) + "\n"


def _freecad(mesh: Mesh) -> str:
    lines = [
        "# CADGenesis analytic FreeCAD script export",
        "import FreeCAD as App, Part",
        "doc = App.newDocument('cadgenesis')",
        "mesh = doc.addObject('Mesh::Feature', 'PartMesh')",
        "mesh.Mesh = Part.__class__.__name__ and __import__('Mesh').Mesh(",
        "    [",
    ]
    lines.extend(f"        ({_num(v.x)},{_num(v.y)},{_num(v.z)})," for v in mesh.vertices)
    lines.append("    ], [")
    lines.extend(f"        ({f[0]},{f[1]},{f[2]})," for f in mesh.faces)
    lines.append("    ])")
    lines.append("doc.recompute()")
    return "\n".join(lines) + "\n"


def _manifest(mesh: Mesh, product: str, kind: str) -> str:
    aabb_min, aabb_max = mesh.aabb()
    return (
        f"# {product} manifest (generated by CADGenesis)\n"
        f"# format: {kind}\n"
        f"mesh: {mesh.name}\n"
        f"vertices: {mesh.vertex_count}\n"
        f"faces: {mesh.face_count}\n"
        f"bounds: [{_num(aabb_min.x)},{_num(aabb_min.y)},{_num(aabb_min.z)}] "
        f"to [{_num(aabb_max.x)},{_num(aabb_max.y)},{_num(aabb_max.z)}]\n"
        f"watertight: {str(mesh.is_watertight()).lower()}\n"
        f"surface_area_mm2: {_num(mesh.surface_area())}\n"
    )


def _text_fallback(mesh: Mesh, fmt: str) -> str:
    return _manifest(mesh, fmt.upper(), "analytic fallback")


def _num(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


__all__ = [
    "ALL_FORMATS",
    "REAL_FORMATS",
    "SCRIPT_FORMATS",
    "STRUCTURED_FORMATS",
    "ExportEngine",
]
