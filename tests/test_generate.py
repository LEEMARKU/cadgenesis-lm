"""
tests/test_generate.py
=======================
Tests for cadgenesis.inference (production inference engine).
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.config import CADConfig
from cadgenesis.inference import CADInferenceEngine, GenerationResult
from cadgenesis.tokenizer import AutonomousCADTokenizer
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer
from cadgenesis.transformer.self_designing import SelfDesigningTransformer


@pytest.fixture(scope="module")
def mini_tok() -> AutonomousCADTokenizer:
    tok = AutonomousCADTokenizer.build_mini()
    tok.build_lang_vocab(["create a steel box", "make a cylinder", "add a sphere"])
    return tok


@pytest.fixture
def mini_engine(mini_tok) -> CADInferenceEngine:
    torch.manual_seed(0)
    return CADInferenceEngine(GeometryAwareTransformer(CADConfig.mini()), mini_tok)


class TestGreedy:
    def test_returns_result(self, mini_engine):
        res = mini_engine.greedy("make a box", max_len=12)
        assert isinstance(res, GenerationResult)
        assert res.text == "make a box"
        assert len(res.tokens) > 0
        assert len(res.ids) == len(res.tokens)

    def test_ids_are_valid(self, mini_engine):
        res = mini_engine.greedy("make a box", max_len=12)
        out_dim = mini_engine.model.out_proj.out_features
        for tok_id in res.ids:
            assert 0 <= tok_id < out_dim

    def test_confidence_in_unit_range(self, mini_engine):
        res = mini_engine.greedy("make a box", max_len=12)
        assert 0.0 <= res.confidence <= 1.0

    def test_max_len_respected(self, mini_engine):
        res = mini_engine.greedy("make a box", max_len=8)
        assert len(res.ids) <= 8


class TestBeam:
    def test_beam_returns_result(self, mini_engine):
        res = mini_engine.beam("make a box", beam_width=3, max_len=12)
        assert isinstance(res, GenerationResult)
        assert len(res.tokens) > 0

    def test_beam_width_must_be_positive(self, mini_engine):
        with pytest.raises(ValueError):
            mini_engine.beam("make a box", beam_width=0, max_len=4)


class TestBatch:
    def test_batch_generate_greedy(self, mini_engine):
        results = mini_engine.batch_generate(["a", "b"], max_len=8)
        assert len(results) == 2
        assert all(isinstance(r, GenerationResult) for r in results)

    def test_batch_generate_beam(self, mini_engine):
        results = mini_engine.batch_generate(["a", "b"], max_len=8, beam_width=2)
        assert len(results) == 2


class TestToonOutput:
    def test_result_toon_round_trip(self, mini_engine):
        res = mini_engine.greedy("make a box", max_len=8)
        seq2 = mini_engine.tokenizer.deserialize_from_toon(res.toon)
        assert seq2.cad_ids == res.ids

    def test_sequence_uses_bos_internal(self, mini_engine):
        # internal beam starts with BOS but reported tokens exclude it
        res = mini_engine.greedy("make a box", max_len=8)
        assert mini_engine.tokenizer.bos_id not in res.ids


class TestTelemetry:
    def test_self_design_report_on_wrapper(self):
        torch.manual_seed(0)
        model = SelfDesigningTransformer(CADConfig.mini())
        tok = AutonomousCADTokenizer.build_mini()
        engine = CADInferenceEngine(model, tok)
        report = engine.self_design_report()
        assert report is not None
        assert "encoder_layers" in report

    def test_self_design_report_none_on_backbone(self, mini_engine):
        assert mini_engine.self_design_report() is None
