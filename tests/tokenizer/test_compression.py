"""tests/tokenizer/test_compression.py
=====================================
Unit tests for cadgenesis.tokenizer.compression.
"""

from __future__ import annotations

import pytest

from cadgenesis.tokenizer.cad_tokenizer import AutonomousCADTokenizer
from cadgenesis.tokenizer.compression import (
    compress_tokens,
    compression_ratio,
    expand_tokens,
    remap_tokens,
    roundtrip_preserves,
)
from cadgenesis.tokenizer.vocabulary import TokenFamily


@pytest.fixture(scope="module")
def tokenizer() -> AutonomousCADTokenizer:
    tok = AutonomousCADTokenizer.build()
    vocab = tok.vocab
    if "PRIM_BOX_NUM_025" not in vocab:
        vocab.merge_tokens(
            ["PRIM_BOX", "NUM_025"],
            "PRIM_BOX_NUM_025",
            TokenFamily.GEOMETRY,
            "Box + 25mm length composite",
        )
    return tok


class TestCompressTokens:
    def test_merges_known_pair(self, tokenizer):
        tokens = ["PRIM_BOX", "NUM_025", "CURVE_LINE"]
        compressed, ratio = compress_tokens(tokens, tokenizer.vocab)
        assert compressed == ["PRIM_BOX_NUM_025", "CURVE_LINE"]
        assert ratio == pytest.approx(1 / 3)

    def test_no_composite_unchanged(self, tokenizer):
        tokens = ["CURVE_LINE", "CURVE_ARC"]
        compressed, ratio = compress_tokens(tokens, tokenizer.vocab)
        assert compressed == tokens
        assert ratio == 0.0

    def test_empty(self, tokenizer):
        assert compress_tokens([], tokenizer.vocab) == ([], 0.0)


class TestExpandTokens:
    def test_expands_composite(self, tokenizer):
        assert expand_tokens(["PRIM_BOX_NUM_025"], tokenizer.vocab) == [
            "PRIM_BOX",
            "NUM_025",
        ]

    def test_non_composite_passthrough(self, tokenizer):
        assert expand_tokens(["PRIM_BOX"], tokenizer.vocab) == ["PRIM_BOX"]

    def test_roundtrip_preserves(self, tokenizer):
        tokens = ["PRIM_BOX", "NUM_025", "CURVE_LINE", "CURVE_ARC"]
        assert roundtrip_preserves(tokens, tokenizer.vocab)


class TestRemap:
    def test_remap_consistent(self, tokenizer):
        tokens = ["PRIM_BOX", "NUM_025"]
        assert remap_tokens(tokens, tokenizer.vocab) == ["PRIM_BOX_NUM_025"]

    def test_compression_ratio(self):
        assert compression_ratio([1, 2, 3], [1, 2]) == pytest.approx(1 / 3)
        assert compression_ratio([], []) == 0.0
