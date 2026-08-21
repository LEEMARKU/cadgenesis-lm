"""cadgenesis.tokenizer.compression
=================================
Lossless CAD token-sequence compression and expansion helpers.

Compression merges adjacent token pairs whose composite token is registered in
the vocabulary (composite names are the underscore-joined pair, e.g.
``PRIM_BOX_NUM_025``).  Every composite token expands back to its components
via ``CADVocabulary.expand_token``, so the process is fully reversible.
"""

from __future__ import annotations


def remap_tokens(tokens: list[str], vocab) -> list[str]:
    """Merge adjacent pairs whose underscore-joined composite is registered."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if i + 1 < len(tokens):
            merged = f"{token}_{tokens[i + 1]}"
            if merged in vocab:
                out.append(merged)
                i += 2
                continue
        out.append(token)
        i += 1
    return out


def compress_tokens(
    tokens: list[str],
    vocab,
) -> tuple[list[str], float]:
    """Greedily compress ``tokens``; returns ``(compressed, ratio)``.

    ``ratio = 1 - |compressed| / |tokens|``.  Lossless — the output expands
    back to the original sequence via :func:`expand_tokens`.
    """
    if not tokens:
        return [], 0.0
    merged = remap_tokens(tokens, vocab)
    ratio = 1.0 - (len(merged) / len(tokens))
    return merged, ratio


def expand_tokens(tokens: list[str], vocab) -> list[str]:
    """Losslessly expand composite tokens back to their components."""
    flattened: list[str] = []
    for tok in tokens:
        if tok in vocab:
            flattened.extend(vocab.expand_token(tok))
        else:
            flattened.append(tok)
    return flattened


def compression_ratio(original: list[str], compressed: list[str]) -> float:
    """Fractional length reduction ``1 - |compressed| / |original|``."""
    if not original:
        return 0.0
    return 1.0 - (len(compressed) / len(original))


def roundtrip_preserves(tokens: list[str], vocab) -> bool:
    """True when compress→expand reproduces the original token list."""
    compressed, _ = compress_tokens(tokens, vocab)
    return expand_tokens(compressed, vocab) == tokens


__all__ = [
    "compress_tokens",
    "compression_ratio",
    "expand_tokens",
    "remap_tokens",
    "roundtrip_preserves",
]
