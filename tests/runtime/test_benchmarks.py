"""
tests/runtime/test_benchmarks.py
================================
Live benchmarks (v6.2): forward timing + decode timing on a mini model.
"""

from __future__ import annotations

import torch

from cadgenesis.config import CADConfig
from cadgenesis.inference.engine import CADInferenceEngine
from cadgenesis.runtime.benchmarks import benchmark_decode, benchmark_forward
from cadgenesis.tokenizer import AutonomousCADTokenizer
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer


def _mini_env():
    torch.manual_seed(0)
    cfg = CADConfig.mini()
    model = GeometryAwareTransformer(cfg)
    tok = AutonomousCADTokenizer.build_mini()
    tok.build_lang_vocab(["create a steel box"])
    engine = CADInferenceEngine(model, tok, device="cpu")
    return model, tok, engine


def test_benchmark_forward_reports_timing():
    model, _tok, _ = _mini_env()
    res = benchmark_forward(
        model,
        batch_size=1,
        seq_len=16,
        vocab_size=model.cad_vocab_size,
        device="cpu",
        steps=2,
    )
    assert res.elapsed_s > 0
    assert res.tokens_per_s > 0
    assert res.backend == "GeometryAwareTransformer"
    assert res.batch_size == 1
    assert res.seq_len == 16


def test_benchmark_decode_reports_per_step():
    _model, _tok, engine = _mini_env()
    res = benchmark_decode(
        engine, engine.greedy, text="create a steel box", max_len=8, device="cpu", steps=2
    )
    assert res.total_s > 0
    assert res.per_step_ms > 0
    assert res.steps == 8


def test_decoder_is_faster_than_forward_per_step():
    """Sanity: one decode step must not be slower than a full 64-token forward."""
    model, _tok, engine = _mini_env()
    fwd = benchmark_forward(
        model, batch_size=1, seq_len=64, vocab_size=model.cad_vocab_size, device="cpu", steps=2
    )
    dec = benchmark_decode(
        engine, engine.greedy, text="create a steel box", max_len=16, device="cpu", steps=1
    )
    assert dec.per_step_ms < fwd.elapsed_s * 1000.0 * 1.5