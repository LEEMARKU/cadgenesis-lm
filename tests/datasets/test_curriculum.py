"""
tests/datasets/test_curriculum.py
=================================
M3 curriculum dataset: categories, quality filter, adversarial rejection,
leakage-free splits, JSONL + manifest output.
"""

from __future__ import annotations

import json

import pytest

from cadgenesis.datasets.curriculum import (
    RECORD_TYPES,
    adversarial_records,
    build_curriculum_records,
    make_splits,
    quality_filter,
    write_curriculum_jsonl,
)


def test_records_are_deterministic_per_seed():
    a = build_curriculum_records(200, seed=7)
    b = build_curriculum_records(200, seed=7)
    c = build_curriculum_records(200, seed=8)
    assert a == b
    assert a != c


def test_all_nine_categories_are_present():
    records = build_curriculum_records(500, seed=7)
    present = {r["type"] for r in records}
    assert present == set(RECORD_TYPES)


def test_records_are_balanced_across_categories():
    records = build_curriculum_records(450, seed=7)
    counts = {kind: sum(1 for r in records if r["type"] == kind) for kind in RECORD_TYPES}
    # Dedup (content-hash + MinHash) removes repeats per type, so the spread
    # reflects the dedup rate, not generator bias.
    assert max(counts.values()) - min(counts.values()) <= 15
    assert sum(counts.values()) >= 300


def test_every_kept_record_passes_quality_filter():
    records = build_curriculum_records(300, seed=7)
    assert all(r["score"] >= 0.85 for r in records)
    assert all(all(r["quality"].values()) for r in records)


def test_quality_filter_rejects_all_adversarial_records():
    base = build_curriculum_records(120, seed=7)
    bad = adversarial_records(base, seed=3)
    kept, rejected = quality_filter(bad)
    assert not kept
    assert len(rejected) == len(bad)


def test_quality_filter_keeps_valid_records():
    base = build_curriculum_records(100, seed=7)
    kept, rejected = quality_filter(base)
    assert len(kept) == len(base)
    assert not rejected


def test_program_ids_are_unique_after_dedup():
    records = build_curriculum_records(400, seed=7)
    ids = [r["program_id"] for r in records]
    assert len(set(ids)) == len(ids)


def test_splits_are_leakage_free_and_stratified():
    records = build_curriculum_records(300, seed=7)
    train, val, test = make_splits(records, seed=42)
    all_ids = (
        {r["program_id"] for r in train}
        | {r["program_id"] for r in val}
        | {r["program_id"] for r in test}
    )
    assert len(all_ids) == len(records), "program leaked across splits"
    assert len(train) + len(val) + len(test) == len(records)
    for kind in RECORD_TYPES:
        assert any(r["type"] == kind for r in train)
        assert any(r["type"] == kind for r in val)
        assert any(r["type"] == kind for r in test)


def test_splits_are_deterministic():
    records = build_curriculum_records(200, seed=7)
    a = make_splits(records, seed=42)
    b = make_splits(records, seed=42)
    assert a == b


def test_invalid_split_fractions_raise():
    records = build_curriculum_records(50, seed=7)
    with pytest.raises(ValueError):
        make_splits(records, train_fraction=1.5)
    with pytest.raises(ValueError):
        make_splits(records, train_fraction=0.8, val_fraction=0.5)


def test_write_curriculum_jsonl_writes_files_and_manifest(tmp_path):
    manifest = write_curriculum_jsonl(tmp_path, n=150, seed=11, progress=False)
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "val.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
    total = manifest["train"]["count"] + manifest["val"]["count"] + manifest["test"]["count"]
    assert manifest["requested"] == 150
    assert 0 < total <= 150, "dedup may reduce the count but never increase it"
    assert manifest["val"]["digest"] and manifest["test"]["digest"]
    assert sum(manifest["types"].values()) == total
    lines = sum(1 for _ in (tmp_path / "train.jsonl").open(encoding="utf-8"))
    assert lines == manifest["train"]["count"]


def test_manifest_digest_is_stable_per_seed(tmp_path):
    write_curriculum_jsonl(tmp_path, n=100, seed=5, progress=False)
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    write_curriculum_jsonl(tmp_path, n=100, seed=5, progress=False)
    manifest2 = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["train"]["digest"] == manifest2["train"]["digest"]


def test_curriculum_files_load_with_existing_pipeline(tmp_path):
    from cadgenesis.datasets.cad_jsonl import CADJsonlDataset
    from cadgenesis.tokenizer import AutonomousCADTokenizer

    write_curriculum_jsonl(tmp_path, n=120, seed=9, progress=False)
    tok = AutonomousCADTokenizer.build_mini()
    ds = CADJsonlDataset(tmp_path / "train.jsonl", tok)
    assert len(ds) > 0
    src, tgt = ds[0]
    assert src and tgt


def test_curriculum_tokens_register_in_mini_vocab():
    from cadgenesis.datasets.cad_program_synth import token_coverage
    from cadgenesis.tokenizer import AutonomousCADTokenizer

    records = build_curriculum_records(200, seed=7)
    tok = AutonomousCADTokenizer.build_mini()
    tok2id = tok.vocab.to_tok2id()
    missing = [t for t in token_coverage(records) if tok2id.get(t) is None]
    assert not missing, f"mini vocab must cover curriculum; missing: {missing}"
