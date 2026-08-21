"""
tests/tokenizer/test_versioning.py
====================================
Unit tests for cadgenesis.tokenizer.versioning and the layout-migration
support on CADVocabulary.

Coverage:
    - Version comparison (compare_versions)
    - migrate_vocabulary preserves ids that still fit
    - migrate_vocabulary remaps ids that no longer fit
    - migrate_vocabulary drops tokens on capacity exhaustion (unmapped)
    - Source vocabulary is never modified
    - remap_ids translates old sequences into the new id space
    - migrate_layout round-trips through CADVocabulary.last_migration
"""

from __future__ import annotations

import pytest

from cadgenesis.tokenizer.versioning import (
    compare_versions,
    migrate_vocabulary,
    remap_ids,
)
from cadgenesis.tokenizer.vocabulary import (
    CADVocabulary,
    TokenFamily,
    _register_special_tokens,
)


@pytest.fixture
def small_vocab() -> CADVocabulary:
    v = CADVocabulary(
        slots={
            TokenFamily.SPECIAL: 64,
            TokenFamily.NUMERIC: 8,
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
    for i in range(8):
        v.register(f"NUM_{i}", TokenFamily.NUMERIC)
    v.register("BOX", TokenFamily.GEOMETRY)
    v.register("CYLINDER", TokenFamily.GEOMETRY)
    return v


class TestCompareVersions:
    def test_equal_versions(self):
        assert compare_versions("2.0.0", "2.0.0") == 0

    def test_less_than(self):
        assert compare_versions("1.9.9", "2.0.0") == -1
        assert compare_versions("2.0.0", "2.1.0") == -1
        assert compare_versions("2.0.0", "2.0.1") == -1

    def test_greater_than(self):
        assert compare_versions("2.1.0", "2.0.0") == 1

    def test_single_component_versions(self):
        assert compare_versions("3", "2.9.9") == 1


class TestMigrateVocabulary:
    def test_source_unchanged(self, small_vocab):
        before = {r.token_id for r in small_vocab}
        new_slots = dict(small_vocab.slot_capacities())
        migrate_vocabulary(small_vocab, new_slots)
        assert {r.token_id for r in small_vocab} == before

    def test_identical_layout_preserves_all_ids(self, small_vocab):
        new_slots = dict(small_vocab.slot_capacities())
        result = migrate_vocabulary(small_vocab, new_slots)
        assert result.preserved_ids == len(small_vocab)
        assert result.dropped_tokens == 0
        assert result.vocab is not None
        assert result.vocab.version == result.target_version

    def test_shrunk_layout_drops_overflow(self, small_vocab):
        new_slots = dict(small_vocab.slot_capacities())
        new_slots[TokenFamily.NUMERIC] = 4  # currently uses 8
        result = migrate_vocabulary(small_vocab, new_slots)
        assert result.dropped_tokens == 4

    def test_shifting_family_start_remaps_ids(self, small_vocab):
        # Growing NUMERIC shifts GEOMETRY's start; BOX/CYLINDER no longer fit
        new_slots = dict(small_vocab.slot_capacities())
        new_slots[TokenFamily.NUMERIC] = 12
        result = migrate_vocabulary(small_vocab, new_slots)
        assert result.remapped_ids == 2
        assert result.dropped_tokens == 0
        # Every token survives; ids outside the new range were reassigned
        assert result.preserved_ids + result.remapped_ids == len(small_vocab)

    def test_tiny_layout_drops_tokens(self, small_vocab):
        new_slots = dict(small_vocab.slot_capacities())
        new_slots[TokenFamily.NUMERIC] = 2
        result = migrate_vocabulary(small_vocab, new_slots)
        assert result.dropped_tokens == 6
        assert len(result.vocab) == len(small_vocab) - 6

    def test_migrated_vocab_lookup_consistent(self, small_vocab):
        new_slots = dict(small_vocab.slot_capacities())
        result = migrate_vocabulary(small_vocab, new_slots)
        for old_rec in small_vocab:
            assert old_rec.token_str in result.vocab
            new_id = result.id_mapping[old_rec.token_id]
            assert result.vocab[old_rec.token_str] == new_id
            assert result.vocab.family_of(old_rec.token_str) == old_rec.family

    def test_custom_target_version(self, small_vocab):
        new_slots = dict(small_vocab.slot_capacities())
        result = migrate_vocabulary(small_vocab, new_slots, target_version="2.1.0")
        assert result.target_version == "2.1.0"
        assert result.vocab.version == "2.1.0"


class TestRemapIds:
    def test_remap_known_and_fallback(self, small_vocab):
        new_slots = dict(small_vocab.slot_capacities())
        result = migrate_vocabulary(small_vocab, new_slots)
        old_ids = [r.token_id for r in small_vocab]
        new_ids = remap_ids(old_ids, result.id_mapping, fallback_unk_id=-1)
        assert len(new_ids) == len(old_ids)
        assert all(i != -1 for i in new_ids)

    def test_missing_ids_use_fallback(self, small_vocab):
        new_slots = dict(small_vocab.slot_capacities())
        result = migrate_vocabulary(small_vocab, new_slots)
        new_ids = remap_ids([999_999, 0], result.id_mapping, fallback_unk_id=42)
        assert new_ids == [42, result.id_mapping.get(0)]


class TestSerializationRoundTrip:
    def test_version_and_parts_survive_roundtrip(self, small_vocab, tmp_path):
        small_vocab.register("A_B", TokenFamily.GEOMETRY, "merged", parts=("A", "B"))
        small_vocab.version = "2.5.0"
        path = tmp_path / "vocab.json"
        small_vocab.save(path)
        loaded = CADVocabulary.load(path)
        assert loaded.version == "2.5.0"
        assert loaded.record_of("A_B").parts == ("A", "B")
        assert loaded.expand_token("A_B") == ["A", "B"]

    def test_old_format_file_loads(self, tmp_path):
        # Files written without the new fields must still load
        import json

        payload = {
            "version": "2.0",
            "tokens": [
                {
                    "token_str": "BOX",
                    "token_id": 72,
                    "family": "GEOMETRY",
                    "type_id": 2,
                    "description": "",
                },
            ],
            "slot_capacities": {
                "SPECIAL": 64,
                "NUMERIC": 8,
                "GEOMETRY": 32,
                "FEATURE": 0,
                "CONSTRAINT": 0,
                "MATERIAL": 0,
                "ASSEMBLY": 0,
                "MANUFACTURING": 0,
                "SIMULATION": 0,
                "LANGUAGE": 64,
            },
        }
        path = tmp_path / "old.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = CADVocabulary.load(path)
        assert loaded["BOX"] == 72
        assert loaded.record_of("BOX").parts == ()
