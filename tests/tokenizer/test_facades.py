"""tests/tokenizer/test_facades.py
=================================
Unit tests for the tokenizer facade modules.
"""

from __future__ import annotations

from cadgenesis.tokenizer.cad_tokenizer import AutonomousCADTokenizer
from cadgenesis.tokenizer.language_tokens import (
    BPETokenizer,
    LanguageTokenizerBase,
    LegacyWordTokenizer,
    build_language_token_table,
)
from cadgenesis.tokenizer.numeric_tokens import (
    NumericTokenizer,
    build_numeric_token_table,
)
from cadgenesis.tokenizer.token_evolution import (
    TokenFrequencyTracker,
    TokenUpgrade,
    VocabularyEvolution,
    VocabularyUpgradePlan,
    guess_family,
)
from cadgenesis.tokenizer.tokenizer import (
    CADTokenSequence,
    MultiModalBatch,
    sequence_to_json,
    serialize_to_toon,
)
from cadgenesis.tokenizer.vocabulary_manager import (
    CADVocabulary,
    TokenFamily,
    compare_versions,
    migrate_vocabulary,
    remap_ids,
)


class TestTokenizerFacade:
    def test_exports(self):
        assert AutonomousCADTokenizer is not None
        assert CADTokenSequence is not None
        assert MultiModalBatch is not None

    def test_build_via_facade(self):
        from cadgenesis.tokenizer.tokenizer import AutonomousCADTokenizer

        tok = AutonomousCADTokenizer.build_mini()
        assert tok.vocab_size > 0


class TestTokenEvolutionFacade:
    def test_exports(self):
        for cls in [
            TokenFrequencyTracker,
            TokenUpgrade,
            VocabularyEvolution,
            VocabularyUpgradePlan,
        ]:
            assert callable(cls)
        assert callable(guess_family)

    def test_guess_family(self):
        assert guess_family("PRIM_BOX") == TokenFamily.GEOMETRY


class TestVocabularyManagerFacade:
    def test_exports(self):
        assert CADVocabulary is not None
        assert callable(compare_versions)
        assert callable(migrate_vocabulary)
        assert callable(remap_ids)

    def test_compare_versions(self):
        assert compare_versions("1.0.0", "2.0.0") < 0


class TestNumericTokens:
    def test_table(self):
        table = build_numeric_token_table()
        assert len(table) > 100
        names = {t for t, _ in table}
        assert any(n.startswith("NUM_") for n in names)
        assert any(n.startswith("ANG_") for n in names)

    def test_numeric_tokenizer_alias(self):
        assert NumericTokenizer.encode_length(25.0)[1].startswith("NUM_")


class TestLanguageTokens:
    def test_legacy_build_from_vocab(self):
        tok = AutonomousCADTokenizer.build_mini()
        table = build_language_token_table(tok.vocab)
        assert isinstance(table, list)
        assert all(isinstance(pair, tuple) for pair in table)

    def test_none_vocab_empty(self):
        assert build_language_token_table() == []

    def test_aliases(self):
        assert LegacyWordTokenizer is not None
        assert BPETokenizer is not None
        assert LanguageTokenizerBase is not None


class TestFacadeSerialization:
    def test_sequence_to_json_via_facade(self):
        tok = AutonomousCADTokenizer.build_mini()
        seq = tok.encode_cad_sequence(["BOX", "NUM_0"])
        data = sequence_to_json(seq)
        assert data["cad_ids"] == seq.cad_ids

    def test_serialize_to_toon_via_facade(self):
        tok = AutonomousCADTokenizer.build_mini()
        seq = tok.encode_cad_sequence(["BOX", "NUM_0"])
        text = serialize_to_toon(seq, tok.vocab)
        assert isinstance(text, str) and text
