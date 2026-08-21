"""tests/tokenizer/test_serialization.py
=======================================
Unit tests for cadgenesis.tokenizer.serialization.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from cadgenesis.tokenizer.cad_tokenizer import AutonomousCADTokenizer
from cadgenesis.tokenizer.serialization import (
    deserialize_from_toon,
    load_sequences,
    save_sequences,
    sequence_from_json,
    sequence_to_json,
    serialize_to_toon,
)


@pytest.fixture(scope="module")
def tokenizer() -> AutonomousCADTokenizer:
    tok = AutonomousCADTokenizer.build()
    tok.build_lang_vocab(["create", "a", "box"])
    return tok


@pytest.fixture(scope="module")
def sequence(tokenizer):
    return tokenizer.encode_multimodal("create a box", ["PRIM_BOX", "NUM_025", "CURVE_LINE"])


class TestJsonRoundTrip:
    def test_round_trip(self, sequence):
        data = sequence_to_json(sequence)
        rebuilt = sequence_from_json(data)
        assert rebuilt.cad_ids == sequence.cad_ids
        assert rebuilt.type_ids == sequence.type_ids
        assert rebuilt.raw_text == sequence.raw_text

    def test_data_is_serialisable(self, sequence):
        import json

        json.dumps(sequence_to_json(sequence))


class TestToon:
    def test_round_trip(self, tokenizer, sequence):
        text = serialize_to_toon(sequence, tokenizer.vocab)
        rebuilt = deserialize_from_toon(text, tokenizer.vocab)
        assert rebuilt.cad_ids == sequence.cad_ids
        assert rebuilt.type_ids == sequence.type_ids


class TestFileIO:
    def test_save_load_round_trip(self, sequence):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "seq.jsonl"
            save_sequences([sequence, sequence], path)
            loaded = load_sequences(path)
            assert len(loaded) == 2
            assert loaded[0].cad_ids == sequence.cad_ids

    def test_load_empty_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "empty.jsonl"
            path.write_text("", encoding="utf-8")
            assert load_sequences(path) == []
