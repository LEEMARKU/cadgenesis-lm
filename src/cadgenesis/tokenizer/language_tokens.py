"""cadgenesis.tokenizer.language_tokens
=====================================
Language token definitions.

Language tokens are corpus-learned rather than statically enumerated, so this
module derives the language token table from a populated vocabulary's
LANGUAGE family when one is supplied.
"""

from cadgenesis.tokenizer.language import (
    BPETokenizer,
    LanguageTokenizerBase,
    LegacyWordTokenizer,
)


def build_language_token_table(vocab=None) -> list[tuple[str, str]]:
    """Derive ``(token_string, description)`` for all LANGUAGE-family tokens.

    When ``vocab`` is a :class:`cadgenesis.tokenizer.vocabulary.CADVocabulary`,
    the table is built from its registered LANGUAGE tokens.  Language tokens
    are corpus-learned, so an empty vocabulary yields an empty table.
    """
    if vocab is None:
        return []
    try:
        from cadgenesis.tokenizer.vocabulary import TokenFamily

        tokens = vocab.tokens_in_family(TokenFamily.LANGUAGE)
    except (AttributeError, TypeError, KeyError):
        return []
    return [(token, "Language token") for token in tokens]


__all__ = [
    "BPETokenizer",
    "LanguageTokenizerBase",
    "LegacyWordTokenizer",
    "build_language_token_table",
]
