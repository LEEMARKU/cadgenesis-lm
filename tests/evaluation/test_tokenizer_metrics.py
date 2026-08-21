"""tests/evaluation/test_tokenizer_metrics.py
=============================================
Unit tests for tokenizer quality metrics against the real tokenizer.
"""

from __future__ import annotations

import pytest

from cadgenesis.evaluation.tokenizer_metrics import TokenizerMetrics
from cadgenesis.tokenizer import AutonomousCADTokenizer

_TEXTS = [
    "create a steel box",
    "design a cylinder",
    "make a sphere of radius 25",
]


@pytest.fixture(scope="module")
def mini_tok() -> AutonomousCADTokenizer:
    tok = AutonomousCADTokenizer.build_mini()
    tok.build_lang_vocab(_TEXTS)
    return tok


def test_oov_rate(mini_tok):
    assert TokenizerMetrics.oov_rate(["create a steel box"], mini_tok) == 0.0
    assert TokenizerMetrics.oov_rate(["quark zap"], mini_tok) == 1.0
    assert TokenizerMetrics.oov_rate([], mini_tok) == 0.0


def test_compression_ratio(mini_tok):
    ratio = TokenizerMetrics.compression_ratio(["create a steel box"], mini_tok)
    assert ratio > 1.0
    assert TokenizerMetrics.compression_ratio([], mini_tok) == 0.0
    assert TokenizerMetrics.compression_ratio([""], mini_tok) == 0.0


def test_round_trip_fidelity(mini_tok):
    assert TokenizerMetrics.round_trip_fidelity(_TEXTS, mini_tok) == 1.0
    assert TokenizerMetrics.round_trip_fidelity([], mini_tok) == 0.0


def test_vocabulary_coverage(mini_tok):
    coverage = TokenizerMetrics.vocabulary_coverage(mini_tok)
    total = mini_tok.vocab_size
    assert coverage["PAD"] == pytest.approx(1.0 / total)
    assert coverage["BOX"] == pytest.approx(1.0 / total)
    assert coverage["SKETCH_RECT"] == pytest.approx(1.0 / total)
    # The mini tokenizer now registers BOTH numeric conventions: padded
    # canonical bins (NUM_000..NUM_019) and legacy raw-mm tokens (NUM_0..NUM_19).
    assert coverage["NUM_0..NUM_19"] == pytest.approx(40.0 / total)
    assert sum(coverage.values()) == pytest.approx(49.0 / total)


def test_vocabulary_coverage_full_tokenizer():
    full_tok = AutonomousCADTokenizer.build()
    coverage = TokenizerMetrics.vocabulary_coverage(full_tok)
    total = full_tok.vocab_size
    assert coverage["BOX"] == pytest.approx(1.0 / total)
    # The full tokenizer registers BOTH numeric conventions: padded canonical
    # bins (NUM_000..NUM_019) and legacy raw-mm tokens (NUM_0..NUM_19).
    assert coverage["NUM_0..NUM_19"] == pytest.approx(40.0 / total)
