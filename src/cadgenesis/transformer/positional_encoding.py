"""cadgenesis.transformer.positional_encoding
===========================================
Backward-compatible alias module for positional encodings.

Re-exports the canonical implementations from :mod:`cadgenesis.transformer.positional`.
"""

from cadgenesis.transformer.positional import (
    ALiBiBias,
    GeometryPositionalEncoding,
    RotaryEmbedding,
    SinusoidalPositionalEncoding,
)

__all__ = [
    "ALiBiBias",
    "GeometryPositionalEncoding",
    "RotaryEmbedding",
    "SinusoidalPositionalEncoding",
]
