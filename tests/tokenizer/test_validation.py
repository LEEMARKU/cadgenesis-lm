"""tests/tokenizer/test_validation.py
====================================
Unit tests for cadgenesis.tokenizer.validation.
"""

from __future__ import annotations

import pytest

from cadgenesis.tokenizer.cad_tokenizer import (
    AutonomousCADTokenizer,
    CADTokenSequence,
)
from cadgenesis.tokenizer.validation import (
    sequence_is_valid,
    unknown_tokens,
    validate_cad_sequence,
    validate_token,
    validate_with_reason,
)


@pytest.fixture(scope="module")
def tokenizer() -> AutonomousCADTokenizer:
    return AutonomousCADTokenizer.build()


class TestValidateToken:
    def test_registered(self, tokenizer):
        ok, msg = validate_token("PRIM_BOX", tokenizer.vocab)
        assert ok
        assert msg == "OK"

    def test_unregistered(self, tokenizer):
        ok, msg = validate_token("NOT_A_REAL_TOKEN_XYZ", tokenizer.vocab)
        assert not ok
        assert "not registered" in msg


class TestValidateCadSequence:
    def test_valid(self, tokenizer):
        ok, _ = validate_cad_sequence(["PRIM_BOX", "NUM_025", "NUM_010"], tokenizer.vocab)
        assert ok

    def test_empty(self, tokenizer):
        ok, _ = validate_cad_sequence([], tokenizer.vocab)
        assert not ok

    def test_unknown(self, tokenizer):
        ok, msg = validate_cad_sequence(["PRIM_BOX", "NOPE"], tokenizer.vocab)
        assert not ok
        assert "Unknown tokens" in msg

    def test_first_must_be_geometry_or_feature(self, tokenizer):
        ok, _ = validate_cad_sequence(["NUM_025", "PRIM_BOX"], tokenizer.vocab)
        assert not ok

    def test_only_specials(self, tokenizer):
        ok, _ = validate_cad_sequence(["<bos>", "<eos>"], tokenizer.vocab)
        assert not ok


class TestSequenceIsValid:
    def test_valid_sequence(self):
        seq = CADTokenSequence(cad_ids=[1, 2], type_ids=[0, 1], attention_mask=[1, 1])
        assert sequence_is_valid(seq)

    def test_misaligned(self):
        seq = CADTokenSequence(cad_ids=[1, 2], type_ids=[0], attention_mask=[1, 1])
        assert not sequence_is_valid(seq)


class TestUnknownTokens:
    def test_finds_unknown(self, tokenizer):
        unk = unknown_tokens(["PRIM_BOX", "GHOST"], tokenizer.vocab)
        assert unk == ["GHOST"]

    def test_none_unknown(self, tokenizer):
        assert unknown_tokens(["PRIM_BOX"], tokenizer.vocab) == []


class TestValidateWithReason:
    def test_ok(self, tokenizer):
        ok, _, unk = validate_with_reason(["PRIM_BOX", "NUM_025"], tokenizer.vocab)
        assert ok and unk == []

    def test_unknown(self, tokenizer):
        ok, _, unk = validate_with_reason(["GHOST"], tokenizer.vocab)
        assert not ok
        assert unk == ["GHOST"]
