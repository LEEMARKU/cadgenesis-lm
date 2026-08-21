"""cadgenesis.evaluation.tokenizer_metrics
=======================================
Tokenizer quality metrics.

Metrics designed for the real ``AutonomousCADTokenizer`` (or any tokenizer
exposing ``encode_text``/``decode_text`` and a ``vocab``): OOV rate,
compression ratio, round-trip fidelity, and coverage of the canonical
CADGenesis vocabulary families.
"""

from __future__ import annotations

import re
from contextlib import suppress
from typing import Any

_WORD_RE = re.compile(r"[a-zA-Z]+|\d+\.\d+|\d+")
_NUMERIC_RE = re.compile(r"NUM_(\d+)")


class TokenizerMetrics:
    """Tokenizer quality metrics over text corpora."""

    _CANONICAL_TOKENS: dict[str, tuple[str, ...]] = {
        "PAD": ("<pad>",),
        "BOS": ("<bos>",),
        "EOS": ("<eos>",),
        "UNK": ("<unk>",),
        "BOX": ("BOX",),
        "CYLINDER": ("CYLINDER",),
        "SPHERE": ("SPHERE",),
        "SKETCH_RECT": ("SKETCH_RECT",),
        "EXTRUDE": ("EXTRUDE",),
    }

    @staticmethod
    def oov_rate(texts: list[str], tokenizer: Any) -> float:
        """Fraction of words not covered by the tokenizer vocabulary.

        A word is known when it is present in the CAD ``vocab`` or the
        language tokenizer's ``tok2id`` mapping.
        """
        total = 0
        oov = 0
        for text in texts:
            for word in _WORD_RE.findall(text.lower()):
                total += 1
                if not TokenizerMetrics._is_known(word, tokenizer):
                    oov += 1
        return oov / total if total else 0.0

    @staticmethod
    def compression_ratio(texts: list[str], tokenizer: Any) -> float:
        """Mean characters-per-token ratio over texts with >=1 token."""
        ratios: list[float] = []
        for text in texts:
            ids = TokenizerMetrics._encode(tokenizer, text)
            if not ids:
                continue
            ratios.append(len(text) / len(ids))
        return sum(ratios) / len(ratios) if ratios else 0.0

    @staticmethod
    def round_trip_fidelity(texts: list[str], tokenizer: Any) -> float:
        """Fraction of texts preserved by encode -> decode round trips."""
        preserved = 0
        checked = 0
        for text in texts:
            ids = TokenizerMetrics._encode(tokenizer, text)
            if ids is None:
                continue
            decoded = TokenizerMetrics._decode(tokenizer, ids)
            if decoded is None:
                continue
            checked += 1
            if decoded == text:
                preserved += 1
        return preserved / checked if checked else 0.0

    @staticmethod
    def vocabulary_coverage(tokenizer: Any) -> dict[str, float]:
        """Fraction of the vocabulary occupied by each canonical family.

        Canonical families: PAD/BOS/EOS/UNK specials, the five legacy
        primitives, and the legacy 20-bin numeric tokens ``NUM_0..NUM_19``.
        """
        entries: list[str] = []
        vocab = getattr(tokenizer, "vocab", None)
        if vocab is not None:
            for item in vocab:
                if isinstance(item, str):
                    entries.append(item)
                elif hasattr(item, "token_str"):
                    entries.append(item.token_str)
        total = len(entries)
        if total == 0:
            return {
                **{name: 0.0 for name in TokenizerMetrics._CANONICAL_TOKENS},
                "NUM_0..NUM_19": 0.0,
            }
        coverage = {
            name: sum(1 for entry in entries if entry in group) / total
            for name, group in TokenizerMetrics._CANONICAL_TOKENS.items()
        }
        legacy_numeric = sum(
            1
            for entry in entries
            if (match := _NUMERIC_RE.fullmatch(entry)) is not None and int(match.group(1)) <= 19
        )
        coverage["NUM_0..NUM_19"] = legacy_numeric / total
        return coverage

    @staticmethod
    def _is_known(word: str, tokenizer: Any) -> bool:
        vocab = getattr(tokenizer, "vocab", None)
        if vocab is not None:
            with suppress(TypeError):
                if word in vocab:
                    return True
        lang_tok = getattr(tokenizer, "lang_tok", None)
        tok2id = getattr(lang_tok, "tok2id", None)
        if isinstance(tok2id, dict):
            return word in tok2id
        return False

    @staticmethod
    def _encode(tokenizer: Any, text: str) -> list[int] | None:
        encode_text = getattr(tokenizer, "encode_text", None)
        if callable(encode_text):
            return list(encode_text(text))
        encode = getattr(tokenizer, "encode", None)
        if callable(encode):
            return list(encode(text))
        return None

    @staticmethod
    def _decode(tokenizer: Any, ids: list[int]) -> str | None:
        decode_text = getattr(tokenizer, "decode_text", None)
        if callable(decode_text):
            return str(decode_text(ids))
        lang_tok = getattr(tokenizer, "lang_tok", None)
        decode = getattr(lang_tok, "decode", None)
        if callable(decode):
            return str(decode(ids))
        return None


__all__ = ["TokenizerMetrics"]
