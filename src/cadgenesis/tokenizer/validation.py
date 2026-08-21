"""cadgenesis.tokenizer.validation
================================
Structural validation helpers for CAD token sequences.

These are standalone versions of the tokenizer's lightweight validation
checks, usable without instantiating the full tokenizer.  Deep geometric
validation is the job of the CAD Execution Intelligence Engine.
"""

from __future__ import annotations

from cadgenesis.tokenizer.cad_tokenizer import CADTokenSequence
from cadgenesis.tokenizer.vocabulary import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN


def validate_token(token_str: str, vocab) -> tuple[bool, str]:
    """Validate a single CAD token string against ``vocab``."""
    if token_str not in vocab:
        return False, f"Token {token_str!r} is not registered."
    return True, "OK"


def validate_cad_sequence(tokens: list[str], vocab) -> tuple[bool, str]:
    """Lightweight structural validation mirroring
    ``AutonomousCADTokenizer.validate_cad_sequence``.

    Checks registration, presence of non-special content, and that the first
    content token is a GEOMETRY or FEATURE token.
    """
    from cadgenesis.tokenizer.vocabulary import TokenFamily

    if not tokens:
        return False, "Empty token sequence."

    unknown = [t for t in tokens if t not in vocab]
    if unknown:
        return False, f"Unknown tokens: {unknown[:5]}"

    content = [t for t in tokens if t not in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)]
    if not content:
        return False, "Sequence contains only special tokens."

    first = content[0]
    family = vocab.family_of(first)
    if family not in (TokenFamily.GEOMETRY, TokenFamily.FEATURE):
        return (
            False,
            f"Expected first CAD token to be GEOMETRY or FEATURE, "
            f"got {family.name if family else 'UNKNOWN'} ({first!r}).",
        )

    return True, "OK"


def sequence_is_valid(seq: CADTokenSequence) -> bool:
    """Structural consistency of a ``CADTokenSequence`` (aligned lists)."""
    n = len(seq.cad_ids)
    return len(seq.type_ids) == n and len(seq.attention_mask) == n and n > 0


def unknown_tokens(tokens: list[str], vocab) -> list[str]:
    """Return the subset of ``tokens`` that is not registered in ``vocab``."""
    return [t for t in tokens if t not in vocab]


def validate_with_reason(
    tokens: list[str],
    vocab,
) -> tuple[bool, str, list[str]]:
    """Extended validation returning ``(ok, message, unknown_tokens)``."""
    unknown = unknown_tokens(tokens, vocab)
    if unknown:
        return False, f"Unknown tokens: {unknown[:5]}", unknown
    ok, msg = validate_cad_sequence(tokens, vocab)
    return ok, msg, []


__all__ = [
    "sequence_is_valid",
    "unknown_tokens",
    "validate_cad_sequence",
    "validate_token",
    "validate_with_reason",
]
