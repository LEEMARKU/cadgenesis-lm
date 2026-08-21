"""cadgenesis.tokenizer.assembly_tokens
=====================================
Assembly relationship token definitions.

Canonical public surface over :mod:`cadgenesis.tokenizer.assembly`
exposing the complete assembly token table used by the CAD vocabulary.
"""

from cadgenesis.tokenizer.assembly import (
    _ALL_ASSEMBLY_TOKENS,
    AssemblyTokenizer,
)

ALL_ASSEMBLY_TOKENS = _ALL_ASSEMBLY_TOKENS

__all__ = ["ALL_ASSEMBLY_TOKENS", "AssemblyTokenizer"]
