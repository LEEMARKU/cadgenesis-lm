"""
tests/tokenizer/test_cad_tokenizer.py
=======================================
Unit + integration tests for AutonomousCADTokenizer.

Coverage:
    - build() factory
    - build_mini() factory (legacy compat)
    - encode_text
    - encode_cad_sequence (BOS/EOS, truncation, type ids)
    - encode_multimodal
    - decode_text / decode_cad_sequence
    - Numeric encode/decode helpers
    - validate_cad_sequence (valid and invalid cases)
    - collate / batching (padding, shapes)
    - vocab_stats / vocab_size properties
    - save / load round-trip
    - as_legacy_data_py compatibility
    - Legacy shim: LangTokenizer, build_dataset, cad_token_type, PAD_ID, BOS_ID, EOS_ID
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from cadgenesis.tokenizer.cad_tokenizer import (
    AutonomousCADTokenizer,
    MultiModalBatch,
)
from cadgenesis.tokenizer.vocabulary import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    TokenFamily,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_tok() -> AutonomousCADTokenizer:
    """Full tokenizer with complete vocabulary (built once per module)."""
    tok = AutonomousCADTokenizer.build()
    texts = [
        "Create a box 50mm wide.",
        "Design a cylinder with radius 10.",
        "Make a sphere of radius 25.",
    ]
    tok.build_lang_vocab(texts)
    return tok


@pytest.fixture(scope="module")
def mini_tok() -> AutonomousCADTokenizer:
    """Mini tokenizer (legacy 20-bin mode)."""
    return AutonomousCADTokenizer.build_mini()


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFactories:
    def test_build_returns_instance(self):
        tok = AutonomousCADTokenizer.build()
        assert isinstance(tok, AutonomousCADTokenizer)

    def test_build_mini_returns_instance(self):
        tok = AutonomousCADTokenizer.build_mini()
        assert isinstance(tok, AutonomousCADTokenizer)

    def test_full_vocab_is_larger_than_mini(self, full_tok, mini_tok):
        assert full_tok.vocab_size > mini_tok.vocab_size

    def test_build_mini_has_legacy_primitives(self, mini_tok):
        for prim in ["BOX", "CYLINDER", "SPHERE", "SKETCH_RECT", "EXTRUDE"]:
            assert prim in mini_tok.vocab

    def test_build_mini_has_legacy_numeric(self, mini_tok):
        for i in range(20):
            assert f"NUM_{i}" in mini_tok.vocab

    def test_repr(self, full_tok):
        r = repr(full_tok)
        assert "AutonomousCADTokenizer" in r
        assert "vocab_size" in r


# ---------------------------------------------------------------------------
# Text encoding tests
# ---------------------------------------------------------------------------


class TestEncodeText:
    def test_encode_text_returns_list(self, full_tok):
        ids = full_tok.encode_text("Create a box")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_empty_string(self, full_tok):
        ids = full_tok.encode_text("")
        assert isinstance(ids, list)  # may be empty or [unk]

    def test_encode_respects_max_text_len(self):
        tok = AutonomousCADTokenizer.build(max_text_len=5)
        tok.build_lang_vocab(["a b c d e f g h i j k"])
        ids = tok.encode_text("a b c d e f g h i j k")
        assert len(ids) <= 5

    def test_encode_text_unk_for_unseen(self, mini_tok):
        # mini tokenizer has no lang vocab built — all words are unk
        mini_tok.encode_text("unprecedented word zxqy")
        # Should not raise, may return unk ids


# ---------------------------------------------------------------------------
# CAD sequence encoding tests
# ---------------------------------------------------------------------------


class TestEncodeCADSequence:
    def test_encode_adds_bos_eos(self, full_tok):
        seq = full_tok.encode_cad_sequence(["PRIM_BOX"])
        toks = seq.raw_cad_tokens
        assert toks[0] == BOS_TOKEN
        assert toks[-1] == EOS_TOKEN

    def test_encode_no_bos(self, full_tok):
        seq = full_tok.encode_cad_sequence(["PRIM_BOX"], add_bos=False)
        assert seq.raw_cad_tokens[0] != BOS_TOKEN

    def test_encode_no_eos(self, full_tok):
        seq = full_tok.encode_cad_sequence(["PRIM_BOX"], add_eos=False)
        assert seq.raw_cad_tokens[-1] != EOS_TOKEN

    def test_type_ids_match_family_values(self, full_tok):
        seq = full_tok.encode_cad_sequence(["PRIM_BOX"])
        for tok_id, type_id in zip(seq.cad_ids, seq.type_ids, strict=False):
            if tok_id in full_tok.vocab:
                expected = full_tok.vocab.type_id_of(tok_id)
                assert type_id == expected

    def test_attention_mask_all_ones(self, full_tok):
        seq = full_tok.encode_cad_sequence(["PRIM_BOX", "PRIM_CYLINDER"])
        assert all(m == 1 for m in seq.attention_mask)

    def test_encode_truncates_at_max_cad_len(self):
        tok = AutonomousCADTokenizer.build(max_cad_len=4)
        long_seq = ["PRIM_BOX"] * 20
        seq = tok.encode_cad_sequence(long_seq)
        assert len(seq.cad_ids) <= 4

    def test_is_valid_after_encode(self, full_tok):
        seq = full_tok.encode_cad_sequence(["PRIM_BOX", "PRIM_CYLINDER"])
        assert seq.is_valid()

    def test_sequence_len_consistent(self, full_tok):
        seq = full_tok.encode_cad_sequence(["PRIM_BOX", "FEAT_EXTRUDE"])
        n = len(seq.cad_ids)
        assert len(seq.type_ids) == n
        assert len(seq.attention_mask) == n


# ---------------------------------------------------------------------------
# Multimodal encoding tests
# ---------------------------------------------------------------------------


class TestEncodeMultimodal:
    def test_encode_multimodal_populates_both_sides(self, full_tok):
        seq = full_tok.encode_multimodal(
            "Create a box",
            ["PRIM_BOX", "FEAT_EXTRUDE"],
        )
        assert len(seq.text_ids) > 0
        assert len(seq.cad_ids) > 0

    def test_raw_text_preserved(self, full_tok):
        text = "Create a sphere with radius 10."
        seq = full_tok.encode_multimodal(text, ["PRIM_SPHERE"])
        assert seq.raw_text == text


# ---------------------------------------------------------------------------
# Decode tests
# ---------------------------------------------------------------------------


class TestDecode:
    def test_decode_cad_removes_pad(self, full_tok):
        pad_id = full_tok.pad_id
        ids = [full_tok.vocab[BOS_TOKEN], pad_id, pad_id]
        decoded = full_tok.decode_cad_sequence(ids)
        assert PAD_TOKEN not in decoded

    def test_decode_cad_round_trip(self, full_tok):
        tokens = ["PRIM_BOX", "FEAT_FILLET"]
        seq = full_tok.encode_cad_sequence(tokens, add_bos=False, add_eos=False)
        decoded = full_tok.decode_cad_sequence(seq.cad_ids)
        assert decoded == tokens


# ---------------------------------------------------------------------------
# Numeric helper tests
# ---------------------------------------------------------------------------


class TestNumericHelpers:
    def test_encode_length_returns_tuple(self, full_tok):
        idx, tok = full_tok.encode_length(50.0)
        assert isinstance(idx, int)
        assert isinstance(tok, str)

    def test_decode_length_round_trip(self, full_tok):
        _, tok = full_tok.encode_length(100.0)
        decoded = full_tok.decode_length(tok)
        assert decoded is not None
        assert abs(decoded - 100.0) <= 1000.0 / 256

    def test_encode_angle_round_trip(self, full_tok):
        _, tok = full_tok.encode_angle(90.0)
        decoded = full_tok.decode_angle(tok)
        assert decoded is not None
        assert abs(decoded - 90.0) <= 1.0


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_geometry_sequence(self, full_tok):
        ok, msg = full_tok.validate_cad_sequence(["PRIM_BOX"])
        assert ok, msg

    def test_valid_feature_sequence(self, full_tok):
        ok, msg = full_tok.validate_cad_sequence(["FEAT_EXTRUDE"])
        assert ok, msg

    def test_empty_sequence_invalid(self, full_tok):
        ok, _msg = full_tok.validate_cad_sequence([])
        assert not ok

    def test_only_specials_invalid(self, full_tok):
        ok, _msg = full_tok.validate_cad_sequence([BOS_TOKEN, EOS_TOKEN])
        assert not ok

    def test_unknown_tokens_invalid(self, full_tok):
        ok, msg = full_tok.validate_cad_sequence(["TOTALLY_UNKNOWN_TOKEN"])
        assert not ok
        assert "Unknown" in msg

    def test_numeric_first_is_invalid(self, full_tok):
        ok, _msg = full_tok.validate_cad_sequence(["NUM_001"])
        assert not ok


# ---------------------------------------------------------------------------
# Collation / batching tests
# ---------------------------------------------------------------------------


class TestCollation:
    @pytest.fixture
    def three_seqs(self, full_tok):
        return [
            full_tok.encode_cad_sequence(["PRIM_BOX"]),
            full_tok.encode_cad_sequence(["PRIM_CYLINDER", "FEAT_EXTRUDE"]),
            full_tok.encode_cad_sequence(["PRIM_SPHERE"]),
        ]

    def test_collate_returns_batch(self, full_tok, three_seqs):
        batch = full_tok.collate(three_seqs)
        assert isinstance(batch, MultiModalBatch)
        assert batch.batch_size == 3

    def test_collate_shapes_consistent(self, full_tok, three_seqs):
        batch = full_tok.collate(three_seqs)
        T = batch.max_tgt_len
        for row in batch.cad_ids:
            assert len(row) == T
        for row in batch.type_ids:
            assert len(row) == T
        for row in batch.attention_mask:
            assert len(row) == T

    def test_collate_padding_with_pad_id(self, full_tok, three_seqs):
        batch = full_tok.collate(three_seqs)
        pad_id = full_tok.pad_id
        for row, attn in zip(batch.cad_ids, batch.attention_mask, strict=False):
            for tok_id, mask in zip(row, attn, strict=False):
                if mask == 0:
                    assert tok_id == pad_id

    def test_collate_empty_raises(self, full_tok):
        with pytest.raises(ValueError, match="empty"):
            full_tok.collate([])

    def test_collate_to_torch(self, full_tok, three_seqs):
        pytest.importorskip("torch")
        import torch

        batch = full_tok.collate(three_seqs)
        tensors = batch.to_torch()
        assert "cad_ids" in tensors
        assert tensors["cad_ids"].shape == (3, batch.max_tgt_len)
        assert tensors["type_ids"].dtype == torch.long


# ---------------------------------------------------------------------------
# Properties and stats tests
# ---------------------------------------------------------------------------


class TestProperties:
    def test_vocab_size_positive(self, full_tok):
        assert full_tok.vocab_size > 0

    def test_pad_bos_eos_ids(self, full_tok):
        assert full_tok.pad_id == full_tok.vocab[PAD_TOKEN]
        assert full_tok.bos_id == full_tok.vocab[BOS_TOKEN]
        assert full_tok.eos_id == full_tok.vocab[EOS_TOKEN]

    def test_vocab_stats_returns_dict(self, full_tok):
        stats = full_tok.vocab_stats()
        assert isinstance(stats, dict)
        assert "TOTAL" in stats

    def test_family_of_known_token(self, full_tok):
        fam = full_tok.family_of("PRIM_BOX")
        assert fam == TokenFamily.GEOMETRY

    def test_family_of_unknown_returns_none(self, full_tok):
        fam = full_tok.family_of("DOES_NOT_EXIST")
        assert fam is None

    def test_type_id_of_known_token(self, full_tok):
        tid = full_tok.type_id_of("PRIM_BOX")
        assert tid == TokenFamily.GEOMETRY.value


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_save_load_round_trip(self, full_tok):
        with tempfile.TemporaryDirectory() as tmpdir:
            full_tok.save(tmpdir)
            loaded = AutonomousCADTokenizer.load(tmpdir)

        assert loaded.vocab_size == full_tok.vocab_size
        assert loaded.max_text_len == full_tok.max_text_len
        assert loaded.max_cad_len == full_tok.max_cad_len

    def test_save_creates_expected_files(self, full_tok):
        with tempfile.TemporaryDirectory() as tmpdir:
            full_tok.save(tmpdir)
            p = Path(tmpdir)
            assert (p / "vocabulary.json").exists()
            assert (p / "tokenizer_state.json").exists()

    def test_loaded_tokenizer_encodes_same(self, full_tok):
        with tempfile.TemporaryDirectory() as tmpdir:
            full_tok.save(tmpdir)
            loaded = AutonomousCADTokenizer.load(tmpdir)

        original = full_tok.encode_cad_sequence(["PRIM_BOX"])
        reloaded = loaded.encode_cad_sequence(["PRIM_BOX"])
        assert original.cad_ids == reloaded.cad_ids


# ---------------------------------------------------------------------------
# Legacy shim integration tests
# ---------------------------------------------------------------------------


class TestLegacyShim:
    def test_shim_pad_bos_eos_ids(self):
        from cadgenesis.tokenizer.legacy_shim import BOS_ID, EOS_ID, PAD_ID

        assert PAD_ID == 0
        assert BOS_ID > PAD_ID
        assert EOS_ID > PAD_ID

    def test_shim_cad_tok2id_has_box(self):
        from cadgenesis.tokenizer.legacy_shim import CAD_TOK2ID

        assert "BOX" in CAD_TOK2ID

    def test_shim_cad_id2tok_inverse(self):
        from cadgenesis.tokenizer.legacy_shim import CAD_ID2TOK, CAD_TOK2ID

        for tok, tid in list(CAD_TOK2ID.items())[:10]:
            assert CAD_ID2TOK[tid] == tok

    def test_shim_cad_token_type(self):
        from cadgenesis.tokenizer.legacy_shim import BOS_ID, cad_token_type

        # BOS is a SPECIAL token → type_id = 0
        assert cad_token_type(BOS_ID) == 0

    def test_shim_build_dataset(self):
        from cadgenesis.tokenizer.legacy_shim import LangTokenizer, build_dataset

        lang_tok = LangTokenizer()
        pairs = build_dataset(10, lang_tok=lang_tok)
        assert len(pairs) == 10
        for text, ids in pairs:
            assert isinstance(text, str)
            assert isinstance(ids, list)
            assert all(isinstance(i, int) for i in ids)

    def test_shim_lang_tokenizer_encode_decode(self):
        from cadgenesis.tokenizer.legacy_shim import LangTokenizer

        lt = LangTokenizer()
        lt.build_vocab(["create a box width 5.0 height 3.0"])
        ids = lt.encode("create a box")
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_shim_num_bins(self):
        from cadgenesis.tokenizer.legacy_shim import NUM_BINS

        assert len(NUM_BINS) == 20
        assert NUM_BINS[0] == pytest.approx(0.5)
        assert NUM_BINS[-1] == pytest.approx(10.0)

    def test_shim_value_to_bin(self):
        from cadgenesis.tokenizer.legacy_shim import NUM_BINS, value_to_bin

        for i, v in enumerate(NUM_BINS):
            assert value_to_bin(v) == i
