"""Tests for cadgenesis.cad.integration (pipeline + bridges)."""

from __future__ import annotations

import pytest

from cadgenesis.cad.integration.execution_bridge import ExecutionBridge
from cadgenesis.cad.integration.pipeline import CADIntelligencePipeline, PipelineResult
from cadgenesis.cad.integration.simulation_bridge import SimulationBridge, SimulationSetup
from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.tokenizer.cad_tokenizer import AutonomousCADTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return AutonomousCADTokenizer.build_mini()


class TestPipeline:
    def test_run_tokenizes_and_stores(self, tokenizer) -> None:
        pipeline = CADIntelligencePipeline(tokenizer=tokenizer)
        result = pipeline.run(
            {
                "material": "AISI 1045",
                "features": [{"type": "FILLET", "params": {"radius": 2.0}}],
            },
            name="test_block",
            text="make a steel block",
        )
        assert result.tokens
        assert result.sequence is not None
        assert result.sequence.is_valid
        assert result.memory_key is not None

    def test_pipeline_result_summary(self) -> None:
        result = PipelineResult(name="x", tokens=["a", "b"])
        summary = result.summary()
        assert summary["token_count"] == 2
        assert summary["name"] == "x"

    def test_run_batch(self, tokenizer) -> None:
        pipeline = CADIntelligencePipeline(tokenizer=tokenizer)
        results = pipeline.run_batch(
            [{"material": "ABS"}, {"material": "AISI 1045"}],
            texts=["one", "two"],
        )
        assert len(results) == 2
        assert all(r.memory_key is not None for r in results)

    def test_run_without_tokenizer(self) -> None:
        pipeline = CADIntelligencePipeline()
        result = pipeline.run({"material": "ABS"}, name="plain")
        assert result.tokens == []
        assert result.sequence is None
        assert result.validation is not None


class TestExecutionBridge:
    def test_run_tokens(self) -> None:
        bridge = ExecutionBridge()
        result = bridge.run_tokens(["PRIM_BOX", "LEN_100"])
        assert result.is_valid_geometry
        assert result.parametric_json["primitive"] == "PRIM_BOX"

    def test_empty_tokens_flagged(self) -> None:
        bridge = ExecutionBridge()
        result = bridge.run_tokens([])
        assert not result.is_valid_geometry
        assert result.errors

    def test_summary(self) -> None:
        bridge = ExecutionBridge()
        summary = bridge.summary(bridge.run_tokens(["PRIM_BOX"]))
        assert summary["is_valid_geometry"] is True
        assert summary["estimated_cost_usd"] > 0


class TestSimulationBridge:
    def test_setup_and_record(self) -> None:
        bridge = SimulationBridge()
        setup = bridge.setup("part_x", "structural", "service", 2.0)
        assert isinstance(setup, SimulationSetup)
        entry = bridge.record(setup, {"max_von_mises": 40.0})
        assert entry.key.startswith("sim:structural:")

    def test_recall_filters(self) -> None:
        bridge = SimulationBridge()
        setup = bridge.setup("part_y", "structural")
        bridge.record(setup, {"value": 1.0})
        hits = bridge.recall("part_y", analysis_type="thermal")
        assert hits == []

    def test_invalid_analysis_type(self) -> None:
        with pytest.raises(ValueError):
            SimulationBridge().setup("z", "quantum")

    def test_mesh_readiness(self) -> None:
        bridge = SimulationBridge()
        info = bridge.mesh_readiness(Mesh.box(10, 10, 10))
        assert info["mesh_present"] is True
        assert info["watertight"] is True
        assert info["readiness"] == 1.0

    def test_mesh_readiness_none(self) -> None:
        info = SimulationBridge().mesh_readiness(None)
        assert info["mesh_present"] is False
