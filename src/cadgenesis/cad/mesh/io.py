"""cadgenesis.cad.mesh.io
======================
Mesh file IO: STL (binary + ASCII), OBJ and PLY (ASCII + binary little
endian) readers and writers.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from cadgenesis.cad.mesh.mesh import Mesh

# ---------------------------------------------------------------------------
# STL
# ---------------------------------------------------------------------------


def read_stl(path: str | Path) -> Mesh:
    """Read an STL file (binary or ASCII) into a :class:`Mesh`."""
    path = Path(path)
    data = path.read_bytes()
    if data[:5] == b"solid":
        return _read_stl_ascii(data.decode("ascii", errors="replace"))
    return _read_stl_binary(data)


def _read_stl_binary(data: bytes) -> Mesh:
    # 80-byte header + 4-byte triangle count
    if len(data) < 84:
        raise ValueError("STL binary file is too short")
    count = struct.unpack_from("<I", data, 80)[0]
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    offset = 84
    expected = 84 + count * 50
    if len(data) < expected:
        raise ValueError(f"STL binary truncated: expected {expected} bytes, got {len(data)}")
    for _ in range(count):
        struct.unpack_from("<3f", data, offset)
        tri = struct.unpack_from("<9f", data, offset + 12)
        struct.unpack_from("<H", data, offset + 48)
        offset += 50
        a, b, c = tri[0:3], tri[3:6], tri[6:9]
        base = len(vertices)
        vertices.extend([tuple(a), tuple(b), tuple(c)])
        faces.append((base, base + 1, base + 2))
    return Mesh(vertices, faces, "stl")


def _read_stl_ascii(text: str) -> Mesh:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    current: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "vertex":
            if len(parts) < 4:
                raise ValueError("malformed STL vertex line")
            current.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif parts[0] == "endloop":
            if len(current) == 3:
                base = len(vertices)
                vertices.extend(current)
                faces.append((base, base + 1, base + 2))
            current = []
    return Mesh(vertices, faces, "stl")


def write_stl(mesh: Mesh, path: str | Path, binary: bool = True) -> None:
    """Write a mesh to an STL file (binary by default, ASCII if ``binary`` is False)."""
    path = Path(path)
    if binary:
        with path.open("wb") as fh:
            fh.write(b"cadgenesis".ljust(80, b"\0"))
            fh.write(struct.pack("<I", mesh.face_count))
            for face in mesh.faces:
                a, b, c = (mesh.vertices[i] for i in face)
                normal = (b - a).cross(c - a)
                length = normal.norm()
                if length > 0:
                    normal = normal / length
                fh.write(
                    struct.pack(
                        "<12fH",
                        normal.x,
                        normal.y,
                        normal.z,
                        a.x,
                        a.y,
                        a.z,
                        b.x,
                        b.y,
                        b.z,
                        c.x,
                        c.y,
                        c.z,
                        0,
                    )
                )
    else:
        lines = ["solid cadgenesis"]
        for face in mesh.faces:
            a, b, c = (mesh.vertices[i] for i in face)
            normal = (b - a).cross(c - a)
            lines.append(f"  facet normal {normal.x} {normal.y} {normal.z}")
            lines.append("    outer loop")
            lines.extend(f"      vertex {p.x} {p.y} {p.z}" for p in (a, b, c))
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append("endsolid cadgenesis")
        path.write_text("\n".join(lines) + "\n", encoding="ascii")


# ---------------------------------------------------------------------------
# OBJ
# ---------------------------------------------------------------------------


def read_obj(path: str | Path) -> Mesh:
    """Read an OBJ file (``v`` and ``f`` records) into a triangle mesh."""
    path = Path(path)
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line.startswith("v "):
            parts = line.split()
            vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif line.startswith("f "):
            indices = []
            for part in line.split()[1:]:
                index = part.split("/")[0]
                idx = int(index)
                if idx < 0:
                    idx = len(vertices) + idx + 1
                indices.append(idx - 1)
            faces.extend(
                (indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1)
            )
    return Mesh(vertices, faces, "obj")


def write_obj(mesh: Mesh, path: str | Path) -> None:
    """Write a mesh to an OBJ file."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"o {mesh.name}\n")
        for vertex in mesh.vertices:
            fh.write(f"v {vertex.x} {vertex.y} {vertex.z}\n")
        for face in mesh.faces:
            fh.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------


def read_ply(path: str | Path) -> Mesh:
    """Read an ASCII or binary-little-endian PLY file into a :class:`Mesh`."""
    path = Path(path)
    raw = path.read_bytes()
    header, is_binary = _parse_ply_header(raw)
    if is_binary:
        return _read_ply_binary(raw, header)
    text = raw.decode("ascii", errors="replace")
    return _read_ply_ascii(text, header)


def _parse_ply_header(raw: bytes) -> tuple[dict[str, Any], bool]:
    if raw[:3] != b"ply":
        raise ValueError("not a PLY file")
    lines: list[str] = []
    index = 0
    is_binary = False
    while index < len(raw):
        end = raw.find(b"\n", index)
        if end == -1:
            break
        line = raw[index:end].decode("ascii", errors="replace").strip()
        lines.append(line)
        index = end + 1
        if line == "end_header":
            break
    header: dict[str, Any] = {"format": "ascii", "vertex_count": 0, "face_count": 0}
    for i, line in enumerate(lines):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            header["format"] = parts[1]
            is_binary = parts[1] != "ascii"
        elif parts[0] == "element" and parts[1] == "vertex":
            header["vertex_count"] = int(parts[2])
            header["vertex_props"] = _collect_props(lines, i)
        elif parts[0] == "element" and parts[1] == "face":
            header["face_count"] = int(parts[2])
            header["face_props"] = _collect_props(lines, i)
    header["binary"] = is_binary
    header["header_bytes"] = index
    return header, is_binary


def _collect_props(lines: list[str], element_line: int) -> list[tuple[str, str]]:
    props: list[tuple[str, str]] = []
    for line in lines[element_line + 1 :]:
        parts = line.split()
        if not parts or parts[0] == "element" or parts[0] == "end_header":
            break
        if parts[0] == "property" and parts[1] == "list":
            props.append(("list", parts[-1]))
        elif parts[0] == "property":
            props.append((parts[1], parts[-1]))
    return props


def _read_ply_ascii(text: str, header: dict[str, Any]) -> Mesh:
    data_lines: list[list[str]] = []
    header_end = 0
    non_empty = 0
    for ln in text.splitlines():
        stripped = ln.strip()
        if not stripped:
            continue
        non_empty += 1
        if stripped == "end_header":
            header_end = non_empty
        data_lines.append(stripped.split())
    data = data_lines[header_end:]
    nv = header["vertex_count"]
    vertices = [(float(r[0]), float(r[1]), float(r[2])) for r in data[:nv]]
    faces: list[tuple[int, int, int]] = []
    for row in data[nv:]:
        count = int(row[0])
        indices = [int(x) for x in row[1 : 1 + count]]
        if len(indices) == 3:
            faces.append((indices[0], indices[1], indices[2]))
        elif len(indices) > 3:
            faces.extend(
                (indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1)
            )
    return Mesh(vertices, faces, "ply")


def _read_ply_binary(raw: bytes, header: dict[str, Any]) -> Mesh:
    offset = header["header_bytes"]
    nv = header["vertex_count"]
    vertices: list[tuple[float, float, float]] = []
    for _ in range(nv):
        x, y, z = struct.unpack_from("<3f", raw, offset)
        offset += 12
        vertices.append((x, y, z))
    faces: list[tuple[int, int, int]] = []
    for _ in range(header["face_count"]):
        count = struct.unpack_from("<B", raw, offset)[0]
        offset += 1
        indices = struct.unpack_from(f"<{count}I", raw, offset)
        offset += 4 * count
        if len(indices) == 3:
            faces.append((indices[0], indices[1], indices[2]))
        elif len(indices) > 3:
            faces.extend(
                (indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1)
            )
    return Mesh(vertices, faces, "ply")


def write_ply(mesh: Mesh, path: str | Path, binary: bool = True) -> None:
    """Write a mesh to a PLY file (binary little endian by default)."""
    path = Path(path)
    header_lines = [
        "ply",
        "format " + ("binary_little_endian" if binary else "ascii") + " 1.0",
        f"element vertex {mesh.vertex_count}",
        "property float x",
        "property float y",
        "property float z",
        f"element face {mesh.face_count}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    if binary:
        payload = bytearray()
        for vertex in mesh.vertices:
            payload += struct.pack("<3f", vertex.x, vertex.y, vertex.z)
        for face in mesh.faces:
            payload += struct.pack("<B3I", 3, face[0], face[1], face[2])
        path.write_bytes(("\n".join(header_lines) + "\n").encode("ascii") + bytes(payload))
    else:
        lines = list(header_lines)
        lines.extend(f"{v.x} {v.y} {v.z}" for v in mesh.vertices)
        lines.extend(f"3 {f[0]} {f[1]} {f[2]}" for f in mesh.faces)
        path.write_text("\n".join(lines) + "\n", encoding="ascii")


__all__ = [
    "read_obj",
    "read_ply",
    "read_stl",
    "write_obj",
    "write_ply",
    "write_stl",
]
