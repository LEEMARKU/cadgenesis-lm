"""Tests for the synthetic CAD program dataset generator + JSONL round-trip."""

from cadgenesis.datasets.cad_jsonl import CADJsonlDataset
from cadgenesis.datasets.cad_program_synth import (
    build_synthetic_records,
    token_coverage,
    write_synthetic_jsonl,
)
from cadgenesis.tokenizer import AutonomousCADTokenizer


def test_records_are_deterministic_per_seed():
    a = build_synthetic_records(50, seed=7)
    b = build_synthetic_records(50, seed=7)
    c = build_synthetic_records(50, seed=8)
    assert a == b
    assert a != c


def test_records_have_text_and_cad():
    for record in build_synthetic_records(20, seed=1):
        assert record["text"]
        assert record["cad"]
        assert all(isinstance(t, str) for t in record["cad"])


def test_all_tokens_are_registerable_in_mini_vocab():
    records = build_synthetic_records(100, seed=3)
    tok = AutonomousCADTokenizer.build_mini()
    covered = token_coverage(records)
    tok2id = tok.vocab.to_tok2id()
    missing = [t for t in covered if tok2id.get(t) is None]
    assert not missing, f"mini vocab must already cover the dataset; missing: {missing}"


def test_jsonl_roundtrip_via_cad_jsonl_dataset(tmp_path):
    path = write_synthetic_jsonl(tmp_path / "progs.jsonl", n=30, seed=5)
    tok = AutonomousCADTokenizer.build_mini()
    from cadgenesis.datasets.cad_program_synth import token_coverage
    from cadgenesis.tokenizer import TokenFamily

    covered = token_coverage(build_synthetic_records(30, seed=5))
    tok2id = tok.vocab.to_tok2id()
    tok.vocab.register_many(
        [
            (t, TokenFamily.NUMERIC if t.startswith("NUM_") else TokenFamily.FEATURE)
            for t in covered
            if tok2id.get(t) is None
        ]
    )
    ds = CADJsonlDataset(path, tok)
    assert len(ds) == 30
    src, tgt = ds[0]
    assert src and tgt
    # every tgt id maps back to a registered token
    id2tok = tok.vocab.to_id2tok()
    assert all(i in id2tok for i in tgt)


def test_token_coverage_collects_all_tokens():
    records = build_synthetic_records(100, seed=9)
    coverage = token_coverage(records)
    for record in records:
        for tok in record["cad"]:
            assert tok in coverage
    assert any(t.startswith("NUM_") for t in coverage)
