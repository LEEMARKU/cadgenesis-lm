"""Tests for the Pillar 8 export engine."""

from __future__ import annotations

import pytest

from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.execution import ALL_FORMATS, ExportEngine


@pytest.fixture
def box() -> Mesh:
    return Mesh.box()


class TestExport:
    @pytest.mark.parametrize("fmt", ALL_FORMATS)
    def test_export_all_formats(self, box: Mesh, tmp_path, fmt: str) -> None:
        path = tmp_path / f"out.{fmt}"
        written = ExportEngine().export(box, path, fmt)
        assert written == str(path)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_export_adds_extension(self, box: Mesh, tmp_path) -> None:
        written = ExportEngine().export(box, str(tmp_path / "bare"), "stl")
        assert written.endswith("bare.stl")

    def test_export_creates_directories(self, box: Mesh, tmp_path) -> None:
        nested = tmp_path / "a" / "b" / "c.stl"
        ExportEngine().export(box, nested, "stl")
        assert nested.exists()

    def test_unknown_format_raises(self, box: Mesh, tmp_path) -> None:
        with pytest.raises(ValueError):
            ExportEngine().export(box, tmp_path / "x.xyz", "xyz")

    def test_accepts_mesh_dict(self, tmp_path) -> None:
        path = tmp_path / "d.stl"
        ExportEngine().export(Mesh.box().to_dict(), path, "stl")
        assert path.exists()

    def test_invalid_mesh_rejected(self, tmp_path) -> None:
        with pytest.raises(TypeError):
            ExportEngine().export("not a mesh", tmp_path / "x.stl", "stl")

    def test_to_text_formats(self, box: Mesh) -> None:
        for fmt in ("gltf", "dxf", "step", "iges", "parasolid", "openscad"):
            assert ExportEngine().to_text(box, fmt).strip()

    def test_read_roundtrip(self, box: Mesh, tmp_path) -> None:
        path = tmp_path / "rt.stl"
        engine = ExportEngine()
        engine.export(box, path, "stl")
        loaded = engine.read(path, "stl")
        assert loaded.face_count == box.face_count
        assert loaded.volume() == pytest.approx(box.volume())

    def test_supported(self) -> None:
        assert set(ExportEngine().supported()) == set(ALL_FORMATS)
        assert len(ALL_FORMATS) >= 12
