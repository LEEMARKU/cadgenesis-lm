"""
tests/tokenizer/test_statistics.py
====================================
Unit tests for cadgenesis.tokenizer.statistics and the tokenizer-level
capabilities layered on top of it (unknown handling, validation,
compression, migration integration).

Coverage:
    - compute_statistics over str/id/CADTokenSequence inputs
    - per-family counts and relative shares
    - unknown-rate detection
    - sequence-length summary
    - compression-ratio measurement
    - AutonomousCADTokenizer.token_statistics integration
    - validate_token / is_unknown_token / register_new_token
    - compress_sequence / expand_sequence lossless round-trip
    - tokenizer migrate_vocabulary + remap_ids_to_vocab integration
"""

from __future__ import annotations

import pytest

from cadgenesis.tokenizer.cad_tokenizer import AutonomousCADTokenizer
from cadgenesis.tokenizer.statistics import compute_statistics
from cadgenesis.tokenizer.vocabulary import TokenFamily


@pytest.fixture
def mini_tok() -> AutonomousCADTokenizer:
    return AutonomousCADTokenizer.build_mini()


@pytest.fixture
def corpus(mini_tok):
    """Three small CAD sequences sharing frequent BOX pairs."""
    return [
        mini_tok.encode_cad_sequence(["BOX", "BOX", "CYLINDER"]),
        mini_tok.encode_cad_sequence(["BOX", "BOX", "SPHERE"]),
        mini_tok.encode_cad_sequence(["BOX", "CYLINDER"]),
    ]


class TestComputeStatistics:
    def test_string_sequences(self, mini_tok):
        stats = compute_statistics([["BOX", "NUM_0"], ["BOX", "BOX"]], mini_tok.vocab)
        assert stats.num_sequences == 2
        assert stats.total_tokens == 4
        assert stats.unique_tokens == 2
        assert stats.min_seq_len == 2 and stats.max_seq_len == 2

    def test_id_sequences(self, mini_tok):
        ids = [mini_tok.vocab["BOX"]] * 3
        stats = compute_statistics([ids], mini_tok.vocab)
        assert stats.total_tokens == 3
        assert stats.per_family_counts.get("GEOMETRY") == 3

    def test_cad_token_sequence_input(self, mini_tok, corpus):
        stats = compute_statistics(corpus, mini_tok.vocab)
        # Each example gained BOS/EOS specials
        assert stats.num_sequences == 3
        assert stats.per_family_counts["SPECIAL"] > 0

    def test_per_family_relative(self, mini_tok):
        stats = compute_statistics([["BOX"]], mini_tok.vocab)
        assert stats.per_family_relative["GEOMETRY"] == pytest.approx(1.0)

    def test_unknown_rate(self, mini_tok):
        stats = compute_statistics([["BOX", "MADE_UP_TOKEN"]], mini_tok.vocab)
        assert stats.unknown_tokens == 1
        assert stats.unknown_rate == pytest.approx(0.5)

    def test_empty_corpus_raises(self, mini_tok):
        with pytest.raises(ValueError):
            compute_statistics([], mini_tok.vocab)

    def test_compression_ratio_with_fn(self, mini_tok, corpus):
        # No composite tokens registered → ratio 0
        stats = compute_statistics(corpus, mini_tok.vocab, compress_fn=lambda t: t)
        assert stats.compression_ratio == 0.0

    def test_compression_ratio_with_tuple_returning_fn(self, mini_tok):
        # compress_sequence returns (tokens, ratio); compute_statistics must
        # tolerate that shape and measure the actual length reduction
        stats = compute_statistics([["BOX", "BOX"]], mini_tok.vocab, compress_fn=lambda t: (t, 0.0))
        assert stats.compression_ratio == 0.0


class TestUnknownHandling:
    def test_is_unknown_token(self, mini_tok):
        assert mini_tok.is_unknown_token("NOT_A_TOKEN")
        assert not mini_tok.is_unknown_token("BOX")

    def test_validate_token_ok(self, mini_tok):
        ok, msg = mini_tok.validate_token("BOX")
        assert ok and msg == "OK"

    def test_validate_token_unknown(self, mini_tok):
        ok, msg = mini_tok.validate_token("NOPE")
        assert not ok
        assert "not registered" in msg

    def test_validate_numeric_decodes(self, mini_tok):
        ok, msg = mini_tok.validate_token("NUM_0")
        assert ok, msg

    def test_register_new_token(self, mini_tok):
        before = mini_tok.vocab_size
        tid = mini_tok.register_new_token("MY_FEATURE", TokenFamily.GEOMETRY)
        assert mini_tok.vocab_size == before + 1
        assert mini_tok.vocab[tid] == "MY_FEATURE"

    def test_register_new_token_guesses_family(self, mini_tok):
        mini_tok.register_new_token("EXTRUDE_A")
        assert mini_tok.family_of("EXTRUDE_A") is not None

    def test_register_existing_raises(self, mini_tok):
        with pytest.raises(KeyError):
            mini_tok.register_new_token("BOX")

    def test_unknown_rate(self, mini_tok, corpus):
        assert mini_tok.unknown_rate(corpus) == 0.0


class TestCompression:
    def test_compress_and_expand_roundtrip(self, mini_tok):
        from cadgenesis.tokenizer.evolution import VocabularyEvolution

        # Register a composite token for a frequent pair
        engine = VocabularyEvolution(vocab=mini_tok.vocab)
        engine.vocab.register("BOX_BOX", TokenFamily.GEOMETRY, "merged", parts=("BOX", "BOX"))

        seq = ["BOX", "BOX", "CYLINDER"]
        compressed, ratio = mini_tok.compress_sequence(seq)
        assert len(compressed) < len(seq)
        assert ratio > 0.0
        assert mini_tok.expand_sequence(compressed) == seq

    def test_no_composite_tokens_no_compression(self, mini_tok):
        seq = ["BOX", "CYLINDER"]
        compressed, ratio = mini_tok.compress_sequence(seq)
        assert compressed == seq
        assert ratio == 0.0

    def test_expand_passthrough(self, mini_tok):
        assert mini_tok.expand_sequence(["BOX"]) == ["BOX"]

    def test_expand_nested_merges(self, mini_tok):
        v = mini_tok.vocab
        v.register("A_B", TokenFamily.GEOMETRY, parts=("A", "B"))
        v.register("X", TokenFamily.GEOMETRY)
        v.register("A_B_X", TokenFamily.GEOMETRY, parts=("A_B", "X"))
        assert v.expand_token("A_B_X") == ["A", "B", "X"]


class TestTokenizerStatisticsIntegration:
    def test_token_statistics(self, mini_tok, corpus):
        stats = mini_tok.token_statistics(corpus)
        assert stats.num_sequences == 3
        assert stats.compression_ratio >= 0.0
        assert set(stats.per_family_counts) >= {"SPECIAL", "GEOMETRY"}

    def test_token_statistics_with_unknowns(self, mini_tok):
        seq = mini_tok.encode_cad_sequence(["BOX", "UNSEEN_ZZ"])
        seq.cad_ids[-1] = mini_tok._unk_id
        stats = mini_tok.token_statistics([seq])
        assert stats.unknown_tokens == 0  # unk id IS in vocabulary
        # encode_cad_sequence maps unknown to <unk>; nothing raw is OOV
        assert mini_tok.is_unknown_token("UNSEEN_ZZ")


class TestMigrationIntegration:
    def test_migrate_and_remap_ids(self, mini_tok):
        new_slots = dict(mini_tok.vocab.slot_capacities())
        result = mini_tok.migrate_vocabulary(new_slots, target_version="3.0.0")
        assert result.vocab is not None
        assert result.target_version == "3.0.0"

        # A legacy sequence of ids maps cleanly into the migrated vocab
        legacy_ids = [mini_tok.vocab["BOX"], mini_tok.vocab["CYLINDER"]]
        migrated_ids = mini_tok.remap_ids_to_vocab(legacy_ids, mapping=result.id_mapping)
        assert migrated_ids == legacy_ids  # same layout → same ids
        assert all(result.vocab[i] == mini_tok.vocab[i] for i in legacy_ids)

    def test_migrate_shrinks_and_remaps(self, mini_tok):
        new_slots = dict(mini_tok.vocab.slot_capacities())
        new_slots[TokenFamily.GEOMETRY] = 4
        result = mini_tok.migrate_vocabulary(new_slots)
        # 5 primitives don't fit in 4 slots → at least one dropped or remapped
        assert result.dropped_tokens + result.remapped_ids > 0
