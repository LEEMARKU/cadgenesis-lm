"""cadgenesis.reasoning.topology
===============================
Topological analysis of B-Rep style models.

Given vertex / edge / face / shell / solid counts (or an explicit mesh), this
module computes the Euler characteristic, genus, manifold / closed-shell checks
and connected components — the classic sanity checks a CAD kernel performs
before committing geometry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TopologyStats:
    """Summary of topological invariants of a model."""

    vertices: int
    edges: int
    faces: int
    shells: int = 0
    solids: int = 0
    loops: int = 0
    connected_components: int = 1
    genus: int = 0
    is_manifold: bool = True
    is_closed: bool = True
    euler_characteristic: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.notes

    def summary(self) -> dict[str, object]:
        return {
            "vertices": self.vertices,
            "edges": self.edges,
            "faces": self.faces,
            "shells": self.shells,
            "solids": self.solids,
            "euler_characteristic": self.euler_characteristic,
            "genus": self.genus,
            "is_manifold": self.is_manifold,
            "is_closed": self.is_closed,
            "connected_components": self.connected_components,
            "notes": self.notes,
        }


class TopologyAnalyzer:
    """Computes topological invariants and structural validity checks."""

    @staticmethod
    def euler_characteristic(vertices: int, edges: int, faces: int) -> int:
        """Euler characteristic ``V - E + F`` of a mesh."""
        TopologyAnalyzer._check_non_negative(vertices, edges, faces)
        return int(vertices - edges + faces)

    @staticmethod
    def genus_for_surface(euler_characteristic_value: int) -> int:
        """Genus of a closed orientable surface: ``(2 - chi) / 2``.

        Returns a non-negative integer when ``chi`` is even; raises otherwise.
        """
        value = int(euler_characteristic_value)
        if (2 - value) % 2 != 0:
            raise ValueError(
                f"Euler characteristic {value} cannot come from a closed "
                "orientable surface (2 - chi must be even)"
            )
        genus = (2 - value) // 2
        if genus < 0:
            raise ValueError(f"genus must be >= 0, got {genus}")
        return genus

    @staticmethod
    def connected_components(n: int, edges: Sequence[tuple[int, int]]) -> int:
        """Number of connected components via union-find over ``n`` nodes."""
        if n < 0:
            raise ValueError("n must be non-negative")
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for a, b in edges:
            if a < 0 or a >= n or b < 0 or b >= n:
                raise IndexError("edge endpoint out of range")
            union(a, b)
        return len({find(i) for i in range(n)})

    @staticmethod
    def _edge_face_counts(
        faces: Sequence[Sequence[int]],
    ) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for face in faces:
            if len(face) < 3:
                raise ValueError("a face must have at least 3 vertices")
            for i in range(len(face)):
                a, b = face[i], face[(i + 1) % len(face)]
                key = (a, b) if a < b else (b, a)
                counts[key] = counts.get(key, 0) + 1
        return counts

    @classmethod
    def is_manifold(cls, faces: Sequence[Sequence[int]]) -> bool:
        """True when every edge is incident to at most two faces."""
        return all(count <= 2 for count in cls._edge_face_counts(faces).values())

    @classmethod
    def is_closed(cls, faces: Sequence[Sequence[int]]) -> bool:
        """True when every edge is incident to exactly two faces."""
        counts = cls._edge_face_counts(faces)
        if not counts:
            return False
        return all(count == 2 for count in counts.values())

    @classmethod
    def analyze(
        cls,
        vertices: int,
        edges: int,
        faces: int,
        shells: int = 0,
        solids: int = 0,
        loops: int = 0,
    ) -> TopologyStats:
        """Full topological summary from counts (Euler-Poincare consistency).

        The genus (through-hole count) of a closed solid follows from
        ``chi = V - E + F`` and ``chi = 2 * solids + loops - 2 * genus``.
        A non-integer or negative result is reported as a note, marking the
        counts inconsistent.
        """
        cls._check_non_negative(vertices, edges, faces, shells, solids, loops)
        notes: list[str] = []

        chi = int(vertices - edges + faces)
        genus: int = 0
        numerator = 2 * solids + loops - chi
        if numerator % 2 != 0 or numerator < 0:
            notes.append(
                f"Euler-Poincare mismatch: V-E+F={chi} but "
                f"2*S+L-2*G={numerator} is not a valid genus"
            )
        else:
            genus = numerator // 2

        # Counts alone cannot verify per-edge adjacency; manifoldness and
        # closedness are reported as pass-through (use analyze_mesh for the
        # exact mesh-based checks).
        return TopologyStats(
            vertices=vertices,
            edges=edges,
            faces=faces,
            shells=shells,
            solids=solids,
            loops=loops,
            genus=genus,
            is_manifold=True,
            is_closed=True,
            euler_characteristic=chi,
            notes=notes,
        )

    @classmethod
    def analyze_mesh(cls, faces: Sequence[Sequence[int]]) -> TopologyStats:
        """Analyze an explicit triangle/face mesh.

        Vertex and edge counts are derived from the face list; manifold and
        closedness are exact edge-adjacency checks.
        """
        if not faces:
            raise ValueError("empty mesh")
        vertex_set = {v for face in faces for v in face}
        n_vertices = len(vertex_set)
        edge_counts = cls._edge_face_counts(faces)
        n_edges = len(edge_counts)
        n_faces = len(faces)

        is_manifold = all(c <= 2 for c in edge_counts.values())
        is_closed = bool(edge_counts) and all(c == 2 for c in edge_counts.values())

        chi = int(n_vertices - n_edges + n_faces)
        notes: list[str] = []
        if is_closed:
            try:
                genus = cls.genus_for_surface(chi)
            except ValueError as exc:
                genus = 0
                notes.append(str(exc))
        else:
            genus = 0

        return TopologyStats(
            vertices=n_vertices,
            edges=n_edges,
            faces=n_faces,
            shells=1 if is_closed else 0,
            solids=1 if is_closed else 0,
            genus=genus,
            is_manifold=is_manifold,
            is_closed=is_closed,
            euler_characteristic=chi,
            connected_components=cls.connected_components(n_vertices, list(edge_counts)),
            notes=notes,
        )

    @staticmethod
    def _check_non_negative(*values: int) -> None:
        for value in values:
            if value < 0:
                raise ValueError(f"counts must be non-negative, got {value}")

    # ---------------------------------------------- P7 adjacency reasoning

    @staticmethod
    def adjacency_graph(
        faces: Sequence[Sequence[int]],
    ) -> dict[int, list[int]]:
        """Face adjacency: faces sharing an edge (via undirected edges).

        Returns a dict ``face_index -> [neighbour face indices]``.  Faces are
        identified by their position in ``faces``.
        """
        edge_owners: dict[tuple[int, int], list[int]] = {}
        for index, face in enumerate(faces):
            n = len(face)
            for i in range(n):
                a, b = face[i], face[(i + 1) % n]
                edge = (a, b) if a <= b else (b, a)
                edge_owners.setdefault(edge, []).append(index)
        graph: dict[int, list[int]] = {i: [] for i in range(len(faces))}
        for owners in edge_owners.values():
            for owner in owners[1:]:
                for other in owners[: len(owners) - 1]:
                    if owner != other:
                        graph[owner].append(other)
                        graph[other].append(owner)
        for index in graph:
            graph[index] = sorted(set(graph[index]))
        return graph

    @classmethod
    def connectivity_reasoning(
        cls,
        faces: Sequence[Sequence[int]],
    ) -> dict[str, Any]:
        """Connectivity analysis of a face mesh via its adjacency graph.

        Returns the number of connected components, the component of each
        face, whether the mesh is connected, and per-component sizes.
        """
        if not faces:
            raise ValueError("empty mesh")
        graph = cls.adjacency_graph(faces)
        component_of: dict[int, int] = {}
        components: dict[int, list[int]] = {}
        for start in range(len(faces)):
            if start in component_of:
                continue
            component_id = len(components)
            stack = [start]
            component_of[start] = component_id
            while stack:
                current = stack.pop()
                for neighbor in graph[current]:
                    if neighbor not in component_of:
                        component_of[neighbor] = component_id
                        stack.append(neighbor)
            components[component_id] = [
                idx for idx, cid in component_of.items() if cid == component_id
            ]
        return {
            "components": len(components),
            "connected": len(components) == 1,
            "component_of": component_of,
            "component_sizes": [len(members) for members in components.values()],
        }


__all__ = ["TopologyAnalyzer", "TopologyStats"]
