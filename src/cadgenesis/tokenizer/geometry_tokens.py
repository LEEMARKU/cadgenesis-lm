"""cadgenesis.tokenizer.geometry_tokens
====================================
Geometry primitive and B-Rep token definitions.

Canonical public surface over :mod:`cadgenesis.tokenizer.geometry` exposing
the complete geometry token table used by the CAD vocabulary.
"""

from cadgenesis.tokenizer.geometry import (
    _ALL_GEOMETRY_TOKENS,
    GeometryTokenizer,
)

ALL_GEOMETRY_TOKENS = _ALL_GEOMETRY_TOKENS

__all__ = ["ALL_GEOMETRY_TOKENS", "GeometryTokenizer"]
