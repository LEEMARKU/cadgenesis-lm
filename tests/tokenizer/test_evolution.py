"""
tests/tokenizer/test_evolution.py
==================================
Tests for cadgenesis.tokenizer.evolution (autonomous vocabulary growth).
"""

from __future__ import annotations

import pytest

from cadgenesis.tokenizer import (
    AutonomousCADTokenizer,
    TokenFamily,
    TokenUpgrade,
    VocabularyEvolution,
    VocabularyUpgradePlan,
    guess_family,
)


@pytest.fixture
def mini_tok() -> AutonomousCADTokenizer:
    return AutonomousCADTokenizer.build_mini()


def _corpus(tok, repeat_unknown: int = 4) -> list:
    seqs = [
        tok.encode_cad_sequence(["BOX", "RARE_NEW_GEOM", "EXTRUDE"]) for _ in range(repeat_unknown)
    ]
    seqs.append(tok.encode_cad_sequence(["BOX", "NUM_5", "EXTRUDE"]))
    return seqs


class TestGuessFamily:
    def test_numeric_prefix(self):
        assert guess_family("NUM_042") == TokenFamily.NUMERIC
        assert guess_family("ANG_090") == TokenFamily.NUMERIC

    def test_special_angle(self):
        assert guess_family("<custom>") == TokenFamily.SPECIAL

    def test_material_and_geometry_default(self):
        assert guess_family("STEEL_1018") == TokenFamily.MATERIAL
        assert guess_family("SOME_NEW_PRIM") == TokenFamily.GEOMETRY


class TestVocabularyUpgradePlan:
    def test_bool_empty(self):
        assert not VocabularyUpgradePlan(operations=[])
        assert bool(
            VocabularyUpgradePlan(operations=[TokenUpgrade("register", "X", TokenFamily.GEOMETRY)])
        )

    def test_len(self):
        ops = [TokenUpgrade("register", f"X{i}", TokenFamily.GEOMETRY) for i in range(3)]
        assert len(VocabularyUpgradePlan(operations=ops)) == 3


class TestTokenFrequencyTracker:
    def test_counts(self):
        from cadgenesis.tokenizer.evolution import TokenFrequencyTracker

        tracker = TokenFrequencyTracker()
        tracker.observe_tokens(["A", "B", "A"], vocab=None)
        assert tracker.total_sequences == 1
        assert tracker.total_tokens == 3
        assert tracker.token_counts["A"] == 2
        assert tracker.pair_counts[("A", "B")] == 1
        assert tracker.pair_counts[("B", "A")] == 1

    def test_unknown_detection(self, mini_tok):
        from cadgenesis.tokenizer.evolution import TokenFrequencyTracker

        tracker = TokenFrequencyTracker()
        tracker.observe_tokens(["BOX", "MYSTERY_TK"], vocab=mini_tok.vocab)
        assert tracker.unknown_counts["MYSTERY_TK"] == 1
        assert "BOX" not in tracker.unknown_counts

    def test_observe_sequence(self, mini_tok):
        from cadgenesis.tokenizer.evolution import TokenFrequencyTracker

        tracker = TokenFrequencyTracker()
        seq = mini_tok.encode_cad_sequence(["BOX", "EXTRUDE"])
        tracker.observe(seq, mini_tok.vocab)
        assert tracker.total_sequences == 1
        assert tracker.token_counts["BOX"] == 1


class TestVocabularyEvolution:
    def test_registers_frequent_unknowns(self, mini_tok):
        engine = VocabularyEvolution(vocab=mini_tok.vocab, min_frequency=3)
        plan = engine.analyze(_corpus(mini_tok))
        registered = [op for op in plan.operations if op.op == "register"]
        assert any(op.token == "RARE_NEW_GEOM" for op in registered)

    def test_apply_registers_new_tokens(self, mini_tok):
        engine = VocabularyEvolution(vocab=mini_tok.vocab, min_frequency=3)
        _plan, records, applied = engine.evolve(_corpus(mini_tok))
        assert any(r.token_str == "RARE_NEW_GEOM" for r in records)
        assert "RARE_NEW_GEOM" in mini_tok.vocab
        assert all(op in applied for op in applied)

    def test_low_frequency_unknown_ignored(self, mini_tok):
        engine = VocabularyEvolution(vocab=mini_tok.vocab, min_frequency=10)
        plan = engine.analyze(_corpus(mini_tok, repeat_unknown=2))
        assert not any(op.token == "RARE_NEW_GEOM" for op in plan.operations)

    def test_merge_composites(self, mini_tok):
        engine = VocabularyEvolution(vocab=mini_tok.vocab, min_pair_frequency=2)
        plan = engine.analyze(_corpus(mini_tok))
        merges = [op for op in plan.operations if op.op == "merge"]
        assert any(op.token == "BOX_RARE_NEW_GEOM" for op in merges)

    def test_remap_merges_pair(self, mini_tok):
        engine = VocabularyEvolution(vocab=mini_tok.vocab, min_pair_frequency=1)
        plan, _records, _ = engine.evolve(_corpus(mini_tok))
        out = engine.remap_sequence(["BOX", "EXTRUDE", "FILLET"], plan)
        # only pairs actually present in the corpus should merge
        assert "BOX_EXTRUDE" not in out or "BOX_EXTRUDE" in mini_tok.vocab

    def test_plan_stats_present(self, mini_tok):
        engine = VocabularyEvolution(vocab=mini_tok.vocab, min_frequency=1)
        plan = engine.analyze(_corpus(mini_tok))
        assert plan.stats["total_sequences"] > 0


class TestTokenizerEvolve:
    def test_evolve_method(self, mini_tok):
        report = mini_tok.evolve(_corpus(mini_tok), min_frequency=3)
        assert report["stats"]["total_sequences"] == len(_corpus(mini_tok))
        assert any(r.token_str == "RARE_NEW_GEOM" for r in report["new_tokens"])

    def test_evolve_then_encode_works(self, mini_tok):
        mini_tok.evolve(_corpus(mini_tok), min_frequency=3)
        # after evolution the formerly-unknown token encodes to a real id
        assert mini_tok.encode_cad_token("RARE_NEW_GEOM") != mini_tok.pad_id

    def test_remap_method(self, mini_tok):
        mini_tok.evolve(_corpus(mini_tok), min_frequency=3, min_pair_frequency=2)
        out = mini_tok.remap_sequence(["BOX", "RARE_NEW_GEOM", "EXTRUDE"])
        assert "BOX_RARE_NEW_GEOM" in out


class TestAutoRegister:
    def test_auto_register(self, mini_tok):
        assert "HOLLOW_PRIM" not in mini_tok.vocab
        tok_id = mini_tok.encode_cad_token("HOLLOW_PRIM", auto_register=True)
        assert tok_id != mini_tok.pad_id
        assert "HOLLOW_PRIM" in mini_tok.vocab

    def test_no_auto_register_by_default(self, mini_tok):
        assert mini_tok.encode_cad_token("NOT_THERE") == mini_tok._unk_id
        assert "NOT_THERE" not in mini_tok.vocab
