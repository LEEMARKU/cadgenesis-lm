"""cadgenesis.cad.mesh.repair
==========================
Mesh repair utilities: duplicate/degenerate face removal, duplicate vertex
welding, consistent normal orientation and simple hole filling.
"""

from __future__ import annotations

from collections import deque

from cadgenesis.cad.geometry.core import Vec
from cadgenesis.cad.mesh.mesh import Mesh


def remove_duplicate_vertices(mesh: Mesh, tolerance: float = 1e-9) -> Mesh:
    """Weld vertices closer than ``tolerance`` and remap faces."""
    mapping: dict[tuple[float, float, float], int] = {}
    remap: list[int] = []
    vertices: list[Vec] = []

    def key(v):
        return (round(v.x, 9), round(v.y, 9), round(v.z, 9))

    for vertex in mesh.vertices:
        k = key(vertex)
        if k not in mapping:
            mapping[k] = len(vertices)
            vertices.append(vertex)
        remap.append(mapping[k])
    faces = [tuple(remap[i] for i in face) for face in mesh.faces]
    return Mesh(vertices, [f for f in faces if len(set(f)) == 3], mesh.name)


def remove_degenerate_faces(mesh: Mesh, min_area: float = 1e-12) -> Mesh:
    """Drop triangles with zero area or area below ``min_area``."""
    kept: list[tuple[int, int, int]] = []
    for face in mesh.faces:
        a, b, c = (mesh.vertices[i] for i in face)
        if (b - a).cross(c - a).norm() * 0.5 > min_area:
            kept.append(face)
    return Mesh(mesh.vertices, kept, mesh.name)


def remove_unreferenced_vertices(mesh: Mesh) -> Mesh:
    """Drop vertices not used by any face (compacts the index space)."""
    used = {i for face in mesh.faces for i in face}
    remap: dict[int, int] = {}
    vertices: list[Vec] = []
    for index, vertex in enumerate(mesh.vertices):
        if index in used:
            remap[index] = len(vertices)
            vertices.append(vertex)
    faces = [tuple(remap[i] for i in face) for face in mesh.faces]
    return Mesh(vertices, faces, mesh.name)


def orient_faces(mesh: Mesh) -> Mesh:
    """Orient faces consistently (outward/inward coherent) via BFS propagation.

    The result has consistent winding; absolute orientation (outward) is not
    guaranteed for non-closed meshes.
    """
    adjacency: dict[int, set[int]] = {i: set() for i in range(mesh.face_count)}
    edge_owner: dict[tuple[int, int], int] = {}
    for face_index, face in enumerate(mesh.faces):
        for i in range(3):
            a, b = face[i], face[(i + 1) % 3]
            key = (a, b) if a < b else (b, a)
            if key in edge_owner:
                adjacency[face_index].add(edge_owner[key])
                adjacency[edge_owner[key]].add(face_index)
            else:
                edge_owner[key] = face_index

    visited: set[int] = set()
    flipped: set[int] = set()
    for start in range(mesh.face_count):
        if start in visited:
            continue
        queue = deque([start])
        visited.add(start)
        while queue:
            current = queue.popleft()
            current_is_flipped = current in flipped
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                a, b = _shared_edge(mesh.faces[current], mesh.faces[neighbor])
                current_forward = _edge_is_forward(mesh.faces[current], a, b, current_is_flipped)
                neighbor_forward = _edge_is_forward(mesh.faces[neighbor], a, b, False)
                if current_forward == neighbor_forward:
                    flipped.add(neighbor)
                queue.append(neighbor)

    faces = [
        (face[2], face[1], face[0]) if index in flipped else face
        for index, face in enumerate(mesh.faces)
    ]
    return Mesh(mesh.vertices, faces, mesh.name)


def _edge_is_forward(face: tuple[int, int, int], a: int, b: int, face_flipped: bool) -> bool:
    """True if the face traverses undirected edge (a, b) as a -> b."""
    sequence = (face[2], face[1], face[0]) if face_flipped else face
    for i in range(3):
        u, v = sequence[i], sequence[(i + 1) % 3]
        if (u, v) == (a, b):
            return True
        if (u, v) == (b, a):
            return False
    raise ValueError("faces do not share the requested edge")


def _shared_edge(face_a: tuple[int, int, int], face_b: tuple[int, int, int]) -> tuple[int, int]:
    edges_a = {(face_a[i], face_a[(i + 1) % 3]) for i in range(3)}
    edges_b = {(face_b[i], face_b[(i + 1) % 3]) for i in range(3)}
    undirected_a = {(a, b) if a < b else (b, a) for (a, b) in edges_a}
    undirected_b = {(a, b) if a < b else (b, a) for (a, b) in edges_b}
    shared = undirected_a & undirected_b
    if not shared:
        raise ValueError("faces do not share an edge")
    a, b = next(iter(shared))
    return (a, b)


def fill_holes(mesh: Mesh, max_hole_edges: int = 12) -> Mesh:
    """Fill boundary loops (holes) with fan triangles.

    Only loops with ``<= max_hole_edges`` edges are filled.  Returns a new
    mesh; the original is left untouched.
    """
    if mesh.is_watertight():
        return mesh
    boundary: dict[tuple[int, int], int] = {}
    for face in mesh.faces:
        for i in range(3):
            a, b = face[i], face[(i + 1) % 3]
            key = (a, b)
            if key in boundary:
                del boundary[key]
            elif (b, a) in boundary:
                del boundary[(b, a)]
            else:
                boundary[key] = 1

    directed = set(boundary)
    faces = list(mesh.faces)
    added = 0
    while directed and added <= max_hole_edges:
        loop = _extract_loop(directed)
        if loop is None or len(loop) > max_hole_edges:
            break
        if len(loop) == 3:
            faces.append((loop[0], loop[1], loop[2]))
        else:
            anchor = loop[0]
            faces.extend((anchor, loop[i], loop[i + 1]) for i in range(1, len(loop) - 1))
        added += 1
    return Mesh(mesh.vertices, faces, mesh.name)


def _extract_loop(directed: set[tuple[int, int]]) -> list[int] | None:
    if not directed:
        return None
    start = directed.pop()
    loop = [start[0], start[1]]
    current = start[1]
    while current != start[0] and len(loop) < 1000:
        found = None
        for candidate in list(directed):
            if candidate[0] == current:
                found = candidate
                break
        if found is None:
            return None
        directed.discard(found)
        loop.append(found[1])
        current = found[1]
    if current != start[0]:
        return None
    return loop[:-1]


def diagnose(mesh: Mesh) -> dict[str, object]:
    """Report mesh health metrics for the validation pipeline."""
    boundary = mesh.boundary_edges()
    degenerate = sum(
        1
        for f in mesh.faces
        if (mesh.vertices[f[1]] - mesh.vertices[f[0]])
        .cross(mesh.vertices[f[2]] - mesh.vertices[f[0]])
        .norm()
        == 0
    )
    return {
        "vertex_count": mesh.vertex_count,
        "face_count": mesh.face_count,
        "edge_count": mesh.edge_count,
        "watertight": mesh.is_watertight(),
        "boundary_edges": len(boundary),
        "degenerate_faces": degenerate,
    }


__all__ = [
    "diagnose",
    "fill_holes",
    "orient_faces",
    "remove_degenerate_faces",
    "remove_duplicate_vertices",
    "remove_unreferenced_vertices",
]
