"""cadgenesis.cad.mesh.mesh
=========================
Triangle mesh data structure with geometric analysis: edges, normals,
surface area, enclosed volume (divergence theorem), axis-aligned bounds and
topology queries.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from cadgenesis.cad.geometry.core import Vec


class Mesh:
    """A triangle mesh: parallel lists of vertices and triangle faces."""

    def __init__(
        self,
        vertices: Sequence[Sequence[float]] | None = None,
        faces: Sequence[Sequence[int]] | None = None,
        name: str = "mesh",
    ) -> None:
        self.vertices: list[Vec] = [Vec.from_sequence(v) for v in (vertices or [])]
        self.faces: list[tuple[int, int, int]] = [
            (int(face[0]), int(face[1]), int(face[2])) for face in (faces or [])
        ]
        self.name = name
        for face in self.faces:
            if len(face) != 3:
                raise ValueError("only triangle meshes are supported")
            if any(i < 0 or i >= len(self.vertices) for i in face):
                raise ValueError("face references an out-of-range vertex")

    # -- construction -----------------------------------------------------------
    @classmethod
    def from_vertices_faces(
        cls, vertices: Sequence[Sequence[float]], faces: Sequence[Sequence[int]], name: str = "mesh"
    ) -> Mesh:
        return cls(vertices, faces, name)

    @classmethod
    def box(
        cls, length: float = 10.0, width: float = 5.0, height: float = 3.0, name: str = "box"
    ) -> Mesh:
        lx, ly, lz = length / 2, width / 2, height / 2
        vertices = [
            (-lx, -ly, -lz),
            (lx, -ly, -lz),
            (lx, ly, -lz),
            (-lx, ly, -lz),
            (-lx, -ly, lz),
            (lx, -ly, lz),
            (lx, ly, lz),
            (-lx, ly, lz),
        ]
        faces = [
            (0, 1, 2),
            (0, 2, 3),
            (5, 4, 7),
            (5, 7, 6),  # bottom / top
            (4, 0, 3),
            (4, 3, 7),
            (1, 5, 6),
            (1, 6, 2),  # left / right
            (4, 5, 1),
            (4, 1, 0),
            (3, 2, 6),
            (3, 6, 7),  # back / front
        ]
        return cls(vertices, faces, name)

    @classmethod
    def uv_sphere(
        cls, radius: float = 5.0, segments: int = 24, rings: int = 12, name: str = "sphere"
    ) -> Mesh:
        """A UV sphere triangulated mesh."""
        vertices: list[tuple[float, float, float]] = []
        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            for seg in range(segments):
                theta = 2 * math.pi * seg / segments
                vertices.append(
                    (
                        radius * math.sin(phi) * math.cos(theta),
                        radius * math.sin(phi) * math.sin(theta),
                        radius * math.cos(phi),
                    )
                )
        faces: list[tuple[int, int, int]] = []
        for ring in range(rings):
            for seg in range(segments):
                i0 = ring * segments + seg
                i1 = (ring + 1) * segments + seg
                i2 = i1 + 1
                i3 = i0 + 1
                if i2 % segments == 0:
                    i2 -= segments
                if i3 % segments == 0:
                    i3 -= segments
                faces.append((i0, i1, i3))
                faces.append((i3, i1, i2))
        return cls(vertices, faces, name)

    @classmethod
    def cylinder(
        cls, radius: float = 3.0, height: float = 10.0, segments: int = 24, name: str = "cylinder"
    ) -> Mesh:
        vertices: list[tuple[float, float, float]] = [
            (0.0, 0.0, -height / 2),
            (0.0, 0.0, height / 2),
        ]
        for seg in range(segments):
            theta = 2 * math.pi * seg / segments
            c, s = math.cos(theta), math.sin(theta)
            vertices.append((radius * c, radius * s, -height / 2))
            vertices.append((radius * c, radius * s, height / 2))
        faces: list[tuple[int, int, int]] = []
        for seg in range(segments):
            nxt = (seg + 1) % segments
            a, b, c2, d = 2 + 2 * seg, 2 + 2 * seg + 1, 2 + 2 * nxt, 2 + 2 * nxt + 1
            faces.append((a, c2, 0))
            faces.append((b, 1, d))
            faces.append((a, b, d))
            faces.append((a, d, c2))
        return cls(vertices, faces, name)

    # -- topology --------------------------------------------------------------
    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def undirected_edges(self) -> dict[tuple[int, int], int]:
        """Map each undirected edge to its usage count."""
        counts: dict[tuple[int, int], int] = {}
        for face in self.faces:
            for i in range(3):
                a, b = face[i], face[(i + 1) % 3]
                key = (a, b) if a < b else (b, a)
                counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def edge_count(self) -> int:
        return len(self.undirected_edges())

    def boundary_edges(self) -> list[tuple[int, int]]:
        """Edges belonging to exactly one face (open boundaries)."""
        return [edge for edge, count in self.undirected_edges().items() if count == 1]

    def is_watertight(self) -> bool:
        edges = self.undirected_edges()
        return bool(edges) and all(count == 2 for count in edges.values())

    # -- geometry ----------------------------------------------------------------
    def face_normals(self) -> list[Vec]:
        return [
            (self.vertices[f[1]] - self.vertices[f[0]])
            .cross(self.vertices[f[2]] - self.vertices[f[0]])
            .normalized()
            for f in self.faces
        ]

    def vertex_normals(self) -> list[Vec]:
        normals: list[Vec] = [Vec(0, 0, 0)] * self.vertex_count
        for face, normal in zip(self.faces, self.face_normals(), strict=False):
            for i in face:
                normals[i] = normals[i] + normal
        return [n / n.norm() if n.norm() > 0 else Vec(0, 0, 1) for n in normals]

    def surface_area(self) -> float:
        area = 0.0
        for face in self.faces:
            a, b, c = (self.vertices[i] for i in face)
            area += (b - a).cross(c - a).norm() * 0.5
        return area

    def volume(self) -> float:
        """Signed volume via the divergence theorem; abs() gives solid volume."""
        total = 0.0
        for face in self.faces:
            a, b, c = (self.vertices[i] for i in face)
            total += a.dot(b.cross(c))
        return abs(total / 6.0)

    def aabb(self) -> tuple[Vec, Vec]:
        if not self.vertices:
            return Vec(0, 0, 0), Vec(0, 0, 0)
        xs = [v.x for v in self.vertices]
        ys = [v.y for v in self.vertices]
        zs = [v.z for v in self.vertices]
        return Vec(min(xs), min(ys), min(zs)), Vec(max(xs), max(ys), max(zs))

    def translate(self, delta: Vec) -> Mesh:
        return Mesh([(v + delta).to_tuple() for v in self.vertices], self.faces, self.name)

    def transformed(self, transform) -> Mesh:
        from cadgenesis.cad.geometry.core import Transform

        if not isinstance(transform, Transform):
            transform = Transform(transform)
        return Mesh(
            [transform.apply(v).to_tuple() for v in self.vertices],
            self.faces,
            self.name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vertices": [v.to_list() for v in self.vertices],
            "faces": [list(f) for f in self.faces],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Mesh:
        return cls(data["vertices"], data["faces"], str(data.get("name", "mesh")))


__all__ = ["Mesh"]
