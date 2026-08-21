"""Tests for cadgenesis.cad.mesh (mesh, io, repair, simplify)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from cadgenesis.cad.mesh.io import (
    read_obj,
    read_ply,
    read_stl,
    write_obj,
    write_ply,
    write_stl,
)
from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.cad.mesh.repair import (
    diagnose,
    fill_holes,
    orient_faces,
    remove_duplicate_vertices,
    remove_unreferenced_vertices,
)
from cadgenesis.cad.mesh.simplify import quadric_simplify, simplify_cluster


class TestMesh:
    def test_box(self) -> None:
        mesh = Mesh.box(10, 5, 3)
        assert mesh.face_count == 12
        assert len(mesh.vertices) == 8

    def test_uv_sphere(self) -> None:
        mesh = Mesh.uv_sphere(radius=1.0, segments=16, rings=8)
        assert mesh.face_count == 16 * 8 * 2

    def test_cylinder(self) -> None:
        mesh = Mesh.cylinder(radius=1.0, height=10, segments=16)
        assert mesh.face_count == 16 * 4

    def test_face_count(self) -> None:
        mesh = Mesh.box()
        assert mesh.face_count == 12

    def test_vertex_count(self) -> None:
        mesh = Mesh.box()
        assert mesh.vertex_count == 8

    def test_invalid_face(self) -> None:
        with pytest.raises(ValueError):
            Mesh(vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], faces=[(0, 1, 2), (0, 1, 5)])


class TestRepair:
    def test_remove_duplicate_vertices(self) -> None:
        mesh = Mesh.box()
        mesh.vertices.append(mesh.vertices[0])  # duplicate vertex
        repaired = remove_duplicate_vertices(mesh)
        assert len(repaired.vertices) == 8

    def test_remove_unreferenced(self) -> None:
        mesh = Mesh.box()
        mesh.vertices.append((100, 100, 100))  # orphan
        repaired = remove_unreferenced_vertices(mesh)
        assert len(repaired.vertices) == 8

    def test_orient_faces_box(self) -> None:
        mesh = Mesh.box()
        oriented = orient_faces(mesh)
        assert oriented.face_count == mesh.face_count

    def test_diagnose(self) -> None:
        mesh = Mesh.box()
        report = diagnose(mesh)
        assert isinstance(report, dict)
        assert report["face_count"] == 12
        assert report["watertight"] is True

    def test_fill_holes_watertight_is_noop(self) -> None:
        mesh = Mesh.box()
        repaired = fill_holes(mesh)
        assert repaired.face_count == mesh.face_count

    def test_fill_holes_closes_boundary(self) -> None:
        mesh = Mesh.box()
        mesh.faces = mesh.faces[:11]  # remove one face -> open box
        assert not mesh.is_watertight()
        repaired = fill_holes(mesh)
        assert repaired.is_watertight()
        assert repaired.face_count == 12


class TestSimplify:
    def test_quadric_reduces_faces(self) -> None:
        mesh = Mesh.uv_sphere(radius=1.0, segments=16, rings=8)
        base = mesh.face_count
        simplified = quadric_simplify(mesh, target_faces=max(10, base // 2))
        assert simplified.face_count <= base

    def test_cluster_reduces_faces(self) -> None:
        mesh = Mesh.uv_sphere(radius=1.0, segments=16, rings=8)
        clustered = simplify_cluster(mesh, cell_size=1.0)
        assert clustered.face_count <= mesh.face_count


class TestIO:
    def _roundtrip(
        self,
        reader,
        writer,
        binary: bool | None = None,
        check_verts: bool = True,
    ) -> Mesh:
        mesh = Mesh.box(4, 2, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mesh"
            if binary is not None:
                writer(mesh, path, binary=binary)
            else:
                writer(mesh, path)
            restored = reader(path)
        assert restored.face_count == mesh.face_count
        if check_verts:
            assert len(restored.vertices) == len(mesh.vertices)
        return restored

    def test_stl_binary(self) -> None:
        # STL expands vertices per-face (36 verts for a triangulated box)
        self._roundtrip(read_stl, write_stl, binary=True, check_verts=False)

    def test_stl_ascii(self) -> None:
        self._roundtrip(read_stl, write_stl, binary=False, check_verts=False)

    def test_obj(self) -> None:
        self._roundtrip(read_obj, write_obj)

    def test_ply_binary(self) -> None:
        self._roundtrip(read_ply, write_ply, binary=True)

    def test_ply_ascii(self) -> None:
        self._roundtrip(read_ply, write_ply, binary=False)
