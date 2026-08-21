"""cadgenesis.cad.mesh.simplify
=============================
Mesh simplification: quadratic-error edge collapse and grid vertex
clustering.  Both reduce the triangle count while preserving overall shape.
"""

from __future__ import annotations

import heapq
import math

from cadgenesis.cad.geometry.core import Vec
from cadgenesis.cad.mesh.mesh import Mesh

# ---------------------------------------------------------------------------
# Quadric error metrics
# ---------------------------------------------------------------------------


def _quadric_for_plane(
    normal: Vec, point: Vec
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    """Return a quadric matrix (Q[0:9]) + scalar term for one plane.

    Plane: ax + by + cz + d = 0 with |(a,b,c)| = 1.
    Q = [aa ab ac ad; ab bb bc bd; ac bc cc cd; ad bd cd dd] (symmetric).
    """
    a, b, c = normal.x, normal.y, normal.z
    d = -normal.dot(point)
    q = (
        a * a,
        a * b,
        a * c,
        a * d,
        b * b,
        b * c,
        b * d,
        c * c,
        c * d,
        d * d,
    )
    return q


def _quadric_error(q, point: Vec) -> float:
    q00, q01, q02, q03, q11, q12, q13, q22, q23, q33 = q
    x, y, z = point.x, point.y, point.z
    return (
        q00 * x * x
        + 2 * q01 * x * y
        + 2 * q02 * x * z
        + 2 * q03 * x
        + q11 * y * y
        + 2 * q12 * y * z
        + 2 * q13 * y
        + q22 * z * z
        + 2 * q23 * z
        + q33
    )


def _quadric_sum(a, b):
    return tuple(va + vb for va, vb in zip(a, b, strict=False))


def quadric_simplify(mesh: Mesh, target_faces: int, max_iterations: int = 100000) -> Mesh:
    """Simplify ``mesh`` by iteratively collapsing the lowest-error edge.

    ``target_faces`` is the desired number of triangles (>= 4).  The result is
    an approximate manifold-preserving simplification.
    """
    target_faces = max(4, int(target_faces))
    if mesh.face_count <= target_faces:
        return mesh

    vertices = [Vec(v.x, v.y, v.z) for v in mesh.vertices]
    faces = [list(f) for f in mesh.faces]
    active_faces: set[int] = set(range(len(faces)))
    face_vertices = {i: set(f) for i, f in enumerate(faces)}

    # per-vertex quadrics
    quadrics: list[tuple[float, ...] | None] = [None] * len(vertices)
    face_quadrics: list[tuple[float, ...]] = []
    for face in faces:
        a, b, c = (vertices[i] for i in face)
        normal = (b - a).cross(c - a)
        n = normal.norm()
        if n < 1e-12:
            face_quadrics.append(_quadric_for_plane(Vec(0, 0, 1), a))
        else:
            face_quadrics.append(_quadric_for_plane(normal / n, a))
    for fi, face in enumerate(faces):
        for vi in face:
            quadrics[vi] = _quadric_sum(quadrics[vi] or (0,) * 10, face_quadrics[fi])

    def compute_cost(va: int, vb: int) -> tuple[float, Vec]:
        q = _quadric_sum(quadrics[va], quadrics[vb])
        mid = vertices[va] + (vertices[vb] - vertices[va]) * 0.5
        return _quadric_error(q, mid), mid

    # priority queue of (cost, move_to, collapse_from, tiebreak)
    heap: list[tuple[float, int, int, int]] = []
    for fi, face in enumerate(faces):
        for i in range(3):
            va, vb = face[i], face[(i + 1) % 3]
            cost, _ = compute_cost(va, vb)
            heapq.heappush(heap, (cost, va, vb, fi))

    collapsed_to: dict[int, int] = {}
    while len(active_faces) > target_faces and heap and max_iterations > 0:
        max_iterations -= 1
        cost, keep, drop, _fi = heapq.heappop(heap)
        if keep in collapsed_to or drop in collapsed_to:
            continue
        # only collapse edges that exist in the current active mesh
        found = False
        for fi in active_faces:
            if drop in face_vertices[fi]:
                found = True
                break
        if not found:
            continue
        # update quadric of the kept vertex
        quadrics[keep] = _quadric_sum(quadrics[keep], quadrics[drop])
        quadrics[drop] = None
        collapsed_to[drop] = keep
        # reposition
        mid = vertices[keep] + (vertices[drop] - vertices[keep]) * 0.5
        vertices[keep] = mid
        # remove faces containing the dropped vertex
        doomed = [fi for fi in active_faces if drop in face_vertices[fi]]
        for fi in doomed:
            active_faces.discard(fi)
        # re-insert affected edges into the heap (faces referencing the kept vertex)
        for fi in list(active_faces):
            if keep in face_vertices[fi]:
                fv_list = list(face_vertices[fi])
                for i in range(len(fv_list)):
                    va = fv_list[i]
                    vb = fv_list[(i + 1) % len(fv_list)]
                    cost, _ = compute_cost(va, vb)
                    heapq.heappush(heap, (cost, va, vb, fi))

    remap: dict[int, int] = {}
    new_vertices: list[Vec] = []
    for index, _vertex in enumerate(vertices):
        root = index
        while root in collapsed_to:
            root = collapsed_to[root]
        if root not in remap:
            remap[root] = len(new_vertices)
            new_vertices.append(vertices[root])
        remap[index] = remap[root]

    new_faces: list[tuple[int, int, int]] = []
    for fi in sorted(active_faces):
        fv = [remap[i] for i in faces[fi]]
        if len(set(fv)) == 3:
            new_faces.append((fv[0], fv[1], fv[2]))
    return Mesh(new_vertices, new_faces, mesh.name)


def simplify_cluster(mesh: Mesh, cell_size: float, name: str = "clustered") -> Mesh:
    """Simplify by clustering vertices into a uniform grid of ``cell_size``.

    Vertices within the same cell are replaced by their centroid and faces
    referencing a duplicated cell vertex are remapped (degenerate triangles
    are dropped).
    """
    if cell_size <= 0:
        raise ValueError("cell_size must be positive")
    lo, _ = mesh.aabb()
    bins: dict[tuple[int, int, int], list[int]] = {}
    for index, vertex in enumerate(mesh.vertices):
        cell = (
            math.floor((vertex.x - lo.x) / cell_size),
            math.floor((vertex.y - lo.y) / cell_size),
            math.floor((vertex.z - lo.z) / cell_size),
        )
        bins.setdefault(cell, []).append(index)
    cell_centroid: dict[tuple[int, int, int], Vec] = {}
    for cell, indices in bins.items():
        total = Vec(0, 0, 0)
        for i in indices:
            total = total + mesh.vertices[i]
        cell_centroid[cell] = total / len(indices)
    vertex_cell = {}
    for cell, indices in bins.items():
        for i in indices:
            vertex_cell[i] = cell
    remap: dict[int, int] = {}
    cell_remap: dict[tuple[int, int, int], int] = {}
    vertices: list[Vec] = []
    for index in range(mesh.vertex_count):
        cell = vertex_cell[index]
        if cell not in cell_remap:
            cell_remap[cell] = len(vertices)
            vertices.append(cell_centroid[cell])
        remap[index] = cell_remap[cell]
    faces: list[tuple[int, int, int]] = []
    for face in mesh.faces:
        mapped = (remap[i] for i in face)
        tri = tuple(sorted(mapped))
        if len(set(tri)) == 3:
            faces.append((tri[0], tri[1], tri[2]))
    return Mesh(vertices, faces, name)


__all__ = ["quadric_simplify", "simplify_cluster"]
