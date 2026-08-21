"""
tests/tokenizer/test_vocabulary.py
====================================
Unit tests for cadgenesis.tokenizer.vocabulary.CADVocabulary.

Coverage:
    - TokenFamily enum properties
    - Token registration (single, batch)
    - Bidirectional lookup (str→id, id→str)
    - Family range allocation
    - Overflow detection
    - Duplicate registration rejection
    - Thread-safety of concurrent registration
    - Serialization (save → load round-trip)
    - Legacy compat helpers (to_tok2id / to_id2tok)
    - Special token constants
    - build_default() factory
"""

from __future__ import annotations

import concurrent.futures
import json
import tempfile
from pathlib import Path

import pytest

from cadgenesis.tokenizer.vocabulary import (
    BOS_TOKEN,
    CAD_END_TOKEN,
    CAD_START_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    CADVocabulary,
    TokenFamily,
    TokenRecord,
    _register_special_tokens,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_vocab() -> CADVocabulary:
    """A vocabulary with only the default slot capacities, no tokens."""
    return CADVocabulary()


@pytest.fixture
def minimal_vocab() -> CADVocabulary:
    """A vocabulary with only SPECIAL + GEOMETRY slots, specials pre-populated."""
    v = CADVocabulary(
        slots={
            TokenFamily.SPECIAL: 64,
            TokenFamily.NUMERIC: 32,
            TokenFamily.GEOMETRY: 32,
            TokenFamily.FEATURE: 0,
            TokenFamily.CONSTRAINT: 0,
            TokenFamily.MATERIAL: 0,
            TokenFamily.ASSEMBLY: 0,
            TokenFamily.MANUFACTURING: 0,
            TokenFamily.SIMULATION: 0,
            TokenFamily.LANGUAGE: 64,
        }
    )
    _register_special_tokens(v)
    return v


# ---------------------------------------------------------------------------
# TokenFamily tests
# ---------------------------------------------------------------------------


class TestTokenFamily:
    def test_family_count(self):
        assert len(TokenFamily) == 10

    def test_family_values_are_unique(self):
        values = [f.value for f in TokenFamily]
        assert len(values) == len(set(values))

    def test_special_is_zero(self):
        assert TokenFamily.SPECIAL.value == 0

    def test_language_is_last(self):
        # LANGUAGE must have the highest value (its range comes last)
        assert TokenFamily.LANGUAGE.value == max(f.value for f in TokenFamily)


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_single_token(self, empty_vocab):
        rec = empty_vocab.register("TEST_TOK", TokenFamily.GEOMETRY, "test")
        assert isinstance(rec, TokenRecord)
        assert rec.token_str == "TEST_TOK"
        assert rec.family == TokenFamily.GEOMETRY

    def test_token_id_is_positive(self, empty_vocab):
        rec = empty_vocab.register("MY_GEOM", TokenFamily.GEOMETRY)
        # ID must be within the GEOMETRY slot range
        start, end = empty_vocab.family_range(TokenFamily.GEOMETRY)
        assert start <= rec.token_id < end

    def test_duplicate_raises_key_error(self, empty_vocab):
        empty_vocab.register("DUP", TokenFamily.GEOMETRY)
        with pytest.raises(KeyError, match="DUP"):
            empty_vocab.register("DUP", TokenFamily.GEOMETRY)

    def test_overflow_raises(self):
        # Create a vocab with capacity 2 in GEOMETRY
        v = CADVocabulary(slots={fam: 0 for fam in TokenFamily})
        v._ranges[TokenFamily.GEOMETRY].capacity = 2
        v.register("A", TokenFamily.GEOMETRY)
        v.register("B", TokenFamily.GEOMETRY)
        with pytest.raises(OverflowError):
            v.register("C", TokenFamily.GEOMETRY)

    def test_register_many(self, empty_vocab):
        tokens = [("G1", TokenFamily.GEOMETRY), ("G2", TokenFamily.GEOMETRY)]
        records = empty_vocab.register_many(tokens)
        assert len(records) == 2
        assert records[0].token_str == "G1"
        assert records[1].token_str == "G2"

    def test_ids_are_contiguous_within_family(self, empty_vocab):
        toks = ["T1", "T2", "T3"]
        recs = [empty_vocab.register(t, TokenFamily.FEATURE) for t in toks]
        ids = [r.token_id for r in recs]
        assert ids == sorted(ids)
        assert ids[1] - ids[0] == 1
        assert ids[2] - ids[1] == 1


# ---------------------------------------------------------------------------
# Lookup tests
# ---------------------------------------------------------------------------


class TestLookup:
    def test_str_lookup(self, empty_vocab):
        rec = empty_vocab.register("LOOK", TokenFamily.GEOMETRY)
        assert empty_vocab["LOOK"] == rec.token_id

    def test_int_lookup(self, empty_vocab):
        rec = empty_vocab.register("LOOK2", TokenFamily.GEOMETRY)
        assert empty_vocab[rec.token_id] == "LOOK2"

    def test_contains_str(self, empty_vocab):
        empty_vocab.register("X", TokenFamily.GEOMETRY)
        assert "X" in empty_vocab
        assert "NOTHERE" not in empty_vocab

    def test_contains_int(self, empty_vocab):
        rec = empty_vocab.register("Y", TokenFamily.GEOMETRY)
        assert rec.token_id in empty_vocab
        assert 999_999 not in empty_vocab

    def test_bad_key_type_raises(self, empty_vocab):
        with pytest.raises(TypeError):
            _ = empty_vocab[3.14]

    def test_family_of_str(self, empty_vocab):
        empty_vocab.register("GEO", TokenFamily.GEOMETRY)
        assert empty_vocab.family_of("GEO") == TokenFamily.GEOMETRY

    def test_type_id_equals_family_value(self, empty_vocab):
        empty_vocab.register("FEAT", TokenFamily.FEATURE)
        assert empty_vocab.type_id_of("FEAT") == TokenFamily.FEATURE.value

    def test_tokens_in_family(self, empty_vocab):
        empty_vocab.register("A", TokenFamily.CONSTRAINT)
        empty_vocab.register("B", TokenFamily.CONSTRAINT)
        empty_vocab.register("C", TokenFamily.GEOMETRY)
        recs = empty_vocab.tokens_in_family(TokenFamily.CONSTRAINT)
        assert len(recs) == 2
        assert all(r.family == TokenFamily.CONSTRAINT for r in recs)


# ---------------------------------------------------------------------------
# Family range tests
# ---------------------------------------------------------------------------


class TestFamilyRange:
    def test_ranges_are_non_overlapping(self):
        """No two families should share any token ID."""
        v = CADVocabulary()
        ranges = [v.family_range(fam) for fam in TokenFamily]
        # Sort by start and check no overlaps
        ranges.sort()
        for i in range(len(ranges) - 1):
            assert ranges[i][1] <= ranges[i + 1][0], (
                f"Family range overlap: {ranges[i]} and {ranges[i + 1]}"
            )

    def test_special_range_starts_at_zero(self):
        v = CADVocabulary()
        start, _ = v.family_range(TokenFamily.SPECIAL)
        assert start == 0

    def test_zero_capacity_family_has_no_range(self):
        v = CADVocabulary(slots={fam: 0 for fam in TokenFamily})
        start, end = v.family_range(TokenFamily.GEOMETRY)
        assert start == end  # zero-width range


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_includes_all_families(self, empty_vocab):
        stats = empty_vocab.stats()
        for fam in TokenFamily:
            assert fam.name in stats

    def test_stats_used_count(self, empty_vocab):
        empty_vocab.register("S1", TokenFamily.SIMULATION)
        empty_vocab.register("S2", TokenFamily.SIMULATION)
        stats = empty_vocab.stats()
        assert stats["SIMULATION"]["used"] == 2

    def test_stats_total_used(self, empty_vocab):
        empty_vocab.register("A", TokenFamily.GEOMETRY)
        empty_vocab.register("B", TokenFamily.FEATURE)
        stats = empty_vocab.stats()
        assert stats["TOTAL"]["used"] == 2


# ---------------------------------------------------------------------------
# Special token tests
# ---------------------------------------------------------------------------


class TestSpecialTokens:
    def test_pad_is_first_token(self, minimal_vocab):
        """PAD must have id=0 — many models hard-code this."""
        assert minimal_vocab[PAD_TOKEN] == 0

    def test_special_tokens_are_registered(self, minimal_vocab):
        for tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, CAD_START_TOKEN, CAD_END_TOKEN):
            assert tok in minimal_vocab

    def test_special_tokens_have_correct_family(self, minimal_vocab):
        for tok in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
            assert minimal_vocab.family_of(tok) == TokenFamily.SPECIAL


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_save_load_round_trip(self, minimal_vocab):
        minimal_vocab.register("TEST_SAVE", TokenFamily.GEOMETRY)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab.json"
            minimal_vocab.save(path)
            loaded = CADVocabulary.load(path)

        assert "TEST_SAVE" in loaded
        assert loaded["TEST_SAVE"] == minimal_vocab["TEST_SAVE"]

    def test_save_produces_valid_json(self, minimal_vocab):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab.json"
            minimal_vocab.save(path)
            with path.open() as fh:
                data = json.load(fh)
        assert "tokens" in data
        assert "version" in data

    def test_loaded_vocab_has_same_size(self, minimal_vocab):
        minimal_vocab.register("Z", TokenFamily.GEOMETRY)
        n = len(minimal_vocab)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vocab.json"
            minimal_vocab.save(path)
            loaded = CADVocabulary.load(path)
        assert len(loaded) == n


# ---------------------------------------------------------------------------
# Thread-safety tests
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_registration(self, empty_vocab):
        """Many threads registering tokens simultaneously should not corrupt state."""
        errors = []
        n_threads = 8
        n_per_thread = 20

        def register_batch(thread_id: int):
            try:
                for i in range(n_per_thread):
                    tok = f"THREAD_{thread_id}_{i}"
                    empty_vocab.register(tok, TokenFamily.GEOMETRY)
            except Exception as exc:
                errors.append(exc)

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = [ex.submit(register_batch, tid) for tid in range(n_threads)]
            concurrent.futures.wait(futures)

        # Only overflow errors are acceptable (geometry slot may fill up)
        [e for e in errors if isinstance(e, OverflowError)]
        other_errors = [e for e in errors if not isinstance(e, OverflowError)]
        assert not other_errors, f"Non-overflow errors: {other_errors}"

        # All successfully registered tokens should be findable
        registered_count = len(empty_vocab.tokens_in_family(TokenFamily.GEOMETRY))
        assert registered_count > 0


# ---------------------------------------------------------------------------
# Legacy compat tests
# ---------------------------------------------------------------------------


class TestLegacyCompat:
    def test_to_tok2id(self, minimal_vocab):
        d = minimal_vocab.to_tok2id()
        assert isinstance(d, dict)
        assert PAD_TOKEN in d
        assert d[PAD_TOKEN] == 0

    def test_to_id2tok(self, minimal_vocab):
        d = minimal_vocab.to_id2tok()
        assert isinstance(d, dict)
        assert 0 in d
        assert d[0] == PAD_TOKEN

    def test_iteration_is_id_sorted(self, minimal_vocab):
        ids = [r.token_id for r in minimal_vocab]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# build_default factory test (integration-ish)
# ---------------------------------------------------------------------------


class TestBuildDefault:
    def test_build_default_creates_vocab(self):
        vocab = CADVocabulary.build_default()
        assert len(vocab) > 100  # should have hundreds of tokens

    def test_build_default_has_all_families(self):
        vocab = CADVocabulary.build_default()
        stats = vocab.stats()
        # All domain families should have > 0 registered tokens
        for fam in [
            TokenFamily.GEOMETRY,
            TokenFamily.FEATURE,
            TokenFamily.CONSTRAINT,
            TokenFamily.MATERIAL,
            TokenFamily.ASSEMBLY,
            TokenFamily.MANUFACTURING,
            TokenFamily.SIMULATION,
            TokenFamily.NUMERIC,
            TokenFamily.SPECIAL,
        ]:
            assert stats[fam.name]["used"] > 0, f"Family {fam.name} has zero registered tokens"

    def test_build_default_pad_is_zero(self):
        vocab = CADVocabulary.build_default()
        assert vocab[PAD_TOKEN] == 0

    def test_build_default_no_id_collisions(self):
        vocab = CADVocabulary.build_default()
        all_ids = [r.token_id for r in vocab]
        assert len(all_ids) == len(set(all_ids)), "Duplicate token IDs detected!"
