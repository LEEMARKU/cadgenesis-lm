"""Tests for the Pillar 8 CAD execution engine pipeline."""

from __future__ import annotations

import os

import pytest

from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.execution import CADExecutionEngine, CADExecutionResult
from cadgenesis.execution.execution_engine import _DesignView

BOX_DESIGN = {
    "name": "box",
    "processes": ["machining"],
    "material": {
        "name": "steel",
        "yield_strength_pa": 250e6,
        "density_kg_m3": 7800.0,
    },
    "volume_m3": 150e-9,
    "analysis": {"type": "structural", "load": {"force_n": 1000.0}},
}


@pytest.fixture
def engine() -> CADExecutionEngine:
    return CADExecutionEngine()


class TestExecute:
    def test_full_pipeline_on_box(self, engine: CADExecutionEngine) -> None:
        result = engine.execute(design={**BOX_DESIGN, "mesh": Mesh.box().to_dict()})
        assert result.is_valid_geometry
        assert result.is_manufacturable
        assert result.safety_factor == pytest.approx(25.0)
        assert result.simulation_report["passed"] is True
        assert 0.5 <= result.confidence_score <= 1.0

    def test_export_writes_file(self, engine: CADExecutionEngine, tmp_path) -> None:
        path = tmp_path / "out.stl"
        result = engine.execute(
            design={**BOX_DESIGN, "mesh": Mesh.box().to_dict()},
            export_fmt="stl",
            export_path=str(path),
        )
        assert result.exports == [str(path)]
        assert os.path.exists(path)

    def test_export_requires_mesh(self, engine: CADExecutionEngine) -> None:
        result = engine.execute(design=dict(BOX_DESIGN), export_fmt="stl")
        assert result.exports == []

    def test_empty_design_vacuous(self, engine: CADExecutionEngine) -> None:
        result = engine.execute(design={})
        assert result.is_valid_geometry is True
        assert result.exports == []
        assert result.cost_breakdown.get("total") == 0.0

    def test_memory_persistence(self, engine: CADExecutionEngine) -> None:
        class FakeMemory:
            def __init__(self) -> None:
                self.entries: list[tuple[str, str, dict]] = []

            def remember(self, pool: str, key: str, content: dict) -> None:
                self.entries.append((pool, key, content))

        memory = FakeMemory()
        engine.execute(design={**BOX_DESIGN, "mesh": Mesh.box().to_dict()}, memory=memory)
        assert any(pool == "project" and key == "exec:latest" for pool, key, _ in memory.entries)

    def test_memory_key_respected(self, engine: CADExecutionEngine) -> None:
        class FakeMemory:
            def __init__(self) -> None:
                self.entries: list[tuple[str, str, dict]] = []

            def remember(self, pool: str, key: str, content: dict) -> None:
                self.entries.append((pool, key, content))

        memory = FakeMemory()
        engine.execute(
            design={**BOX_DESIGN, "mesh": Mesh.box().to_dict()},
            memory=memory,
            memory_key="exec:part_1",
        )
        assert any(key == "exec:part_1" for _, key, _ in memory.entries)

    def test_repair_report_written(self, engine: CADExecutionEngine) -> None:
        mesh = Mesh.box()
        mesh.faces.pop()
        result = engine.execute(
            design={
                "mesh": mesh.to_dict(),
                "name": "broken",
                "processes": ["injection_molding"],
                "has_undercut": True,
            }
        )
        assert result.repair_report["attempted"] is True
        assert result.is_valid_geometry is True


class TestLegacyCompat:
    def test_execute_and_evaluate_unchanged(self, engine: CADExecutionEngine) -> None:
        result = engine.execute_and_evaluate(["PRIM_BOX", "LEN_100"])
        assert result.is_valid_geometry
        assert result.parametric_json["primitive"] == "PRIM_BOX"

    def test_empty_tokens_invalid(self, engine: CADExecutionEngine) -> None:
        result = engine.execute_and_evaluate([])
        assert not result.is_valid_geometry
        assert result.errors

    def test_execute_with_program(self, engine: CADExecutionEngine) -> None:
        result = engine.execute(program=["PRIM_BOX"])
        assert result.is_valid_geometry
        assert result.parametric_json.get("program") == ["PRIM_BOX"]


class TestAssemblyAndMechanism:
    def test_execute_assembly(self, engine: CADExecutionEngine) -> None:
        report = engine.execute_assembly(
            {"parts": ["base", "bracket"]},
            [
                {
                    "name": "m1",
                    "mate_type": "COINCIDENT",
                    "reference_a": {"component": "base", "entity": "top"},
                    "reference_b": {"component": "bracket", "entity": "bottom"},
                },
            ],
        )
        assert report["parts"] == 2
        assert report["mates"] == 1

    def test_execute_assembly_empty(self, engine: CADExecutionEngine) -> None:
        report = engine.execute_assembly({"parts": []})
        assert report["parts"] == 0

    def test_simulate_mechanism(self, engine: CADExecutionEngine) -> None:
        from cadgenesis.cad.mechanisms.joints import Joint, Mechanism

        mechanism = Mechanism("arm")
        mechanism.add_link("l0")
        mechanism.add_link("l1")
        mechanism.add_joint(Joint("j1", "REVOLUTE", "l0", "l1"))
        summary = engine.simulate_mechanism(mechanism, {"j1": 0.5})
        assert summary["passed"] is True

    def test_analyze_strength(self, engine: CADExecutionEngine) -> None:
        summary = engine.analyze_strength(dict(BOX_DESIGN), force_n=1000.0)
        assert summary["analysis_type"] == "structural"


class TestDesignView:
    def test_mesh_faces_only_with_mesh(self) -> None:
        view = _DesignView({"mesh": {}}, Mesh.box())
        assert view.faces is None
        assert view.vertices is None

    def test_faces_vertices_from_part(self) -> None:
        view = _DesignView(
            {"faces": [[0, 1, 2]], "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]},
            None,
        )
        assert view.faces == [[0, 1, 2]]
        assert view.vertices == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]

    def test_result_contract(self) -> None:
        result = CADExecutionResult()
        assert result.is_valid_geometry is True
        assert result.estimated_cost_usd == 45.0
        assert result.summary()["estimated_cost_usd"] == 45.0
        assert result.errors == []
