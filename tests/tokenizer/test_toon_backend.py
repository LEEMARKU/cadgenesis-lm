"""
tests/tokenizer/test_toon_backend.py
====================================
Tests for cadgenesis.tokenizer.toon_backend (TOON serialization backend).
"""

from __future__ import annotations

import pytest

from cadgenesis.tokenizer import AutonomousCADTokenizer, ToonBackend


@pytest.fixture
def mini_tok() -> AutonomousCADTokenizer:
    return AutonomousCADTokenizer.build_mini()


class TestToonBackendSequences:
    def test_sequence_round_trip(self, mini_tok):
        seq = mini_tok.encode_cad_sequence(["BOX", "NUM_1", "EXTRUDE"])
        text = mini_tok.serialize_to_toon(seq)
        seq2 = mini_tok.deserialize_from_toon(text)
        assert seq2.cad_ids == seq.cad_ids
        assert seq2.type_ids == seq.type_ids
        assert seq2.attention_mask == seq.attention_mask

    def test_round_trip_without_vocab(self):
        backend = ToonBackend()  # no vocab
        seq = AutonomousCADTokenizer.build_mini().encode_cad_sequence(["BOX"])
        text = backend.serialize_sequence(seq)
        seq2 = backend.deserialize_sequence(text)
        assert seq2.cad_ids == seq.cad_ids

    def test_sequence_to_text(self, mini_tok):
        backend = mini_tok.toon_backend
        seq = mini_tok.encode_cad_sequence(["BOX", "EXTRUDE"])
        text = backend.sequence_to_text(seq)
        tokens = backend.text_to_tokens(text)
        assert tokens == ["<bos>", "BOX", "EXTRUDE", "<eos>"]

    def test_multimodal_round_trip(self, mini_tok):
        seq = mini_tok.encode_multimodal("make a box", ["BOX", "EXTRUDE"])
        text = mini_tok.serialize_to_toon(seq)
        seq2 = mini_tok.deserialize_from_toon(text)
        assert seq2.cad_ids == seq.cad_ids


class TestToonBackendVocabulary:
    def test_state_round_trip_custom_slots(self, mini_tok):
        state = mini_tok.toon_backend.serialize_vocabulary_state()
        assert set(state.keys()) == {"slots", "tokens"}
        vocab2 = mini_tok.toon_backend.deserialize_vocabulary_state(state)
        assert len(vocab2) == len(mini_tok.vocab)
        # ids preserved verbatim
        for r in mini_tok.vocab:
            assert vocab2[r.token_id] == r.token_str
        # slot layout preserved
        assert vocab2.slot_capacities() == mini_tok.vocab.slot_capacities()

    def test_state_round_trip_default(self):
        tok = AutonomousCADTokenizer.build()
        state = tok.toon_backend.serialize_vocabulary_state()
        vocab2 = tok.toon_backend.deserialize_vocabulary_state(state)
        assert len(vocab2) == len(tok.vocab)
        assert vocab2["PRIM_BOX"] == tok.vocab["PRIM_BOX"]

    def test_default_slots_deserialize(self):
        # single-string form assumes the default layout; it must round-trip a
        # default-layout vocabulary exactly
        tok = AutonomousCADTokenizer.build()
        text = tok.toon_backend.serialize_vocabulary()
        vocab2 = ToonBackend().deserialize_vocabulary(text)
        assert len(vocab2) == len(tok.vocab)
        assert vocab2["PRIM_BOX"] == tok.vocab["PRIM_BOX"]

    def test_estimate_tokens(self, mini_tok):
        n = mini_tok.toon_backend.estimate_tokens("create a steel box 50mm")
        assert isinstance(n, int) and n > 0
