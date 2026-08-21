from __future__ import annotations

import importlib.util
import math
import sys
import types

import pytest

from cadgenesis.config import CADConfig
from cadgenesis.evaluation.hf_eval import benchmark_suite, cad_perplexity, run_lm_eval
from cadgenesis.tokenizer import AutonomousCADTokenizer
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


@pytest.fixture(scope="module")
def model_and_tokenizer():
    cfg = CADConfig.mini()
    model = GeometryAwareTransformer(cfg)
    model.eval()
    tokenizer = AutonomousCADTokenizer.build_mini()
    tokenizer.build_lang_vocab(["sketch", "rect", "extrude", "box", "cylinder"])
    return model, tokenizer


def _cad_ids(tokenizer, tokens: list[str]) -> list[int]:
    return [tokenizer.vocab[t] for t in tokens]


class TestCadPerplexity:
    def test_returns_finite_positive_perplexity(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        sequences = [
            _cad_ids(tokenizer, ["SKETCH_RECT", "EXTRUDE"]),
            _cad_ids(tokenizer, ["BOX", "CYLINDER", "NUM_1", "NUM_2"]),
        ]
        ppl = cad_perplexity(model, tokenizer, sequences, batch_size=2)
        assert math.isfinite(ppl)
        assert ppl > 0.0
        assert ppl > 1.0

    def test_single_sequence_batch(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        seq = _cad_ids(tokenizer, ["SKETCH_RECT", "EXTRUDE"])
        ppl = cad_perplexity(model, tokenizer, [seq], batch_size=1)
        assert math.isfinite(ppl)
        assert ppl > 1.0

    def test_empty_list_raises(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        with pytest.raises(ValueError):
            cad_perplexity(model, tokenizer, [])


class TestRunLmEval:
    @pytest.mark.skipif(
        importlib.util.find_spec("lm_eval") is not None,
        reason="lm_eval is installed in this environment",
    )
    def test_raises_importerror_when_lm_eval_missing(self):
        with pytest.raises(ImportError) as excinfo:
            run_lm_eval(None, None)
        assert "lm-eval" in str(excinfo.value)

    def test_returns_status_dict_when_available(self, monkeypatch):
        fake = types.ModuleType("lm_eval")
        monkeypatch.setitem(sys.modules, "lm_eval", fake)
        result = run_lm_eval(None, None, tasks=["mmlu"])
        assert result["tasks"] == ["mmlu"]
        assert result["status"] == "requires lm_eval integration adapter"

    def test_default_tasks(self, monkeypatch):
        fake = types.ModuleType("lm_eval")
        monkeypatch.setitem(sys.modules, "lm_eval", fake)
        result = run_lm_eval(None, None)
        assert result["tasks"] == ["mmlu"]


class TestBenchmarkSuite:
    def test_returns_expected_keys(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        seq = _cad_ids(tokenizer, ["SKETCH_RECT", "EXTRUDE"])
        report = benchmark_suite(model, tokenizer, [seq])
        assert set(report.keys()) == {"perplexity", "n_sequences"}
        assert report["n_sequences"] == 1
        assert math.isfinite(report["perplexity"])
        assert report["perplexity"] > 1.0
