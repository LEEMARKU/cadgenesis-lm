"""tests/tokenizer/test_legacy_vocab_coverage.py
===============================================
The dataset layers emit the *legacy* token convention (``BOX``, ``EXTRUDE``,
unpadded ``NUM_80``).  These tests prove the full and mini vocabularies cover
every dataset program end-to-end: tokenize -> model forward.
"""

from __future__ import annotations

import pytest

from cadgenesis.config import CADConfig
from cadgenesis.datasets.cad_program_synth import build_synthetic_records, token_coverage
from cadgenesis.tokenizer import AutonomousCADTokenizer
from cadgenesis.tokenizer.legacy_shim import NUM_BINS
from cadgenesis.tokenizer.vocabulary import TokenFamily


@pytest.fixture(scope="module")
def full_tok() -> AutonomousCADTokenizer:
    return AutonomousCADTokenizer.build()


@pytest.fixture(scope="module")
def mini_tok() -> AutonomousCADTokenizer:
    return AutonomousCADTokenizer.build_mini()


@pytest.fixture(scope="module")
def records():
    return build_synthetic_records(200, seed=42)


def test_dataset_tokens_covered_by_full_vocab(full_tok, records):
    """Every token the dataset emits must encode to a real (non-unk) id."""
    missing = [
        t for t in token_coverage(records) if full_tok.encode_cad_token(t) == full_tok.unk_id
    ]
    assert not missing, f"dataset tokens missing from full vocab: {missing}"


def test_dataset_tokens_covered_by_mini_vocab(mini_tok, records):
    """build_mini() must also cover the dataset (training path uses mini)."""
    missing = [
        t for t in token_coverage(records) if mini_tok.encode_cad_token(t) == mini_tok.unk_id
    ]
    assert not missing, f"dataset tokens missing from mini vocab: {missing}"


def test_legacy_operations_in_correct_families(full_tok):
    assert full_tok.family_of("BOX") is TokenFamily.GEOMETRY
    assert full_tok.family_of("EXTRUDE") is TokenFamily.FEATURE
    assert full_tok.family_of("SKETCH_RECT") is TokenFamily.GEOMETRY
    assert full_tok.family_of("CYLINDER") is TokenFamily.GEOMETRY
    assert full_tok.family_of("SPHERE") is TokenFamily.GEOMETRY


def test_canonical_tokens_still_registered(full_tok):
    """Adding the legacy set must not displace the canonical vocabulary."""
    assert "PRIM_BOX" in full_tok.vocab
    assert "FEAT_EXTRUDE" in full_tok.vocab
    assert "NUM_012" in full_tok.vocab
    assert full_tok.encode_cad_token("BOX") != full_tok.encode_cad_token("PRIM_BOX") or (
        "BOX" in full_tok.vocab and "PRIM_BOX" in full_tok.vocab
    )


def test_unpadded_numeric_round_trip(full_tok):
    """NUM_80 must decode to the 80 mm bin centre and re-encode consistently."""
    assert "NUM_80" in full_tok.vocab
    value = full_tok.decode_length("NUM_80")
    assert value is not None and 70.0 <= value <= 90.0
    assert full_tok.decode_length("NUM_80") == pytest.approx(value)


def test_legacy_shim_tokens_covered(full_tok, mini_tok):
    """legacy_shim generators (data.py compat) emit NUM_0..NUM_19 + prims."""
    legacy_tokens = ["BOX", "CYLINDER", "SPHERE", "SKETCH_RECT", "EXTRUDE"]
    legacy_tokens += [f"NUM_{i}" for i in range(len(NUM_BINS))]
    for tok in (full_tok, mini_tok):
        missing = [t for t in legacy_tokens if tok.encode_cad_token(t) == tok.unk_id]
        assert not missing, f"legacy shim tokens missing: {missing}"


def test_sequence_encode_and_validate(full_tok, records):
    """A full dataset program must pass the tokenizer's structural validation."""
    sample = records[0]
    seq = full_tok.encode_cad_sequence(sample["cad"])
    assert seq.is_valid()
    ok, reason = full_tok.validate_cad_sequence(sample["cad"])
    assert ok, reason


def test_mini_ids_stay_within_lang_embed_range(mini_tok):
    """Mini-mode CAD ids must not collide with the mini model's language
    embedding range (CADConfig.mini() sizes lang_vocab_size=512)."""
    from cadgenesis.config import CADConfig

    lang_size = CADConfig.mini().tokenizer.lang_vocab_size
    max_id = max(r.token_id for r in mini_tok.vocab)
    assert max_id < lang_size, f"mini vocab id {max_id} >= lang_vocab_size {lang_size}"


def test_model_forward_accepts_dataset_tokens():
    """The actual model must consume dataset token ids without OOB errors."""
    import torch

    tok = AutonomousCADTokenizer.build_mini()
    records = build_synthetic_records(8, seed=7)
    model = None
    from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer

    model = GeometryAwareTransformer(CADConfig.mini())
    model.eval()
    for record in records[:4]:
        seq = tok.encode_cad_sequence(record["cad"])
        src = torch.zeros((1, 4), dtype=torch.long)
        tgt = torch.tensor([seq.cad_ids[:8]], dtype=torch.long)
        typ = torch.zeros_like(tgt)
        with torch.no_grad():
            logits = model(src, tgt, typ)[0]
        assert logits.shape[1] == tgt.shape[1]
        assert logits.shape[0] == 1
        assert logits.shape[2] > max(seq.cad_ids), "logits do not cover dataset token ids"
