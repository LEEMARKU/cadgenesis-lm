"""
CADGenesis SDK
==============
Standalone SDK modules that ship with CADGenesis-LM but are not part of the
core ``cadgenesis`` package.

- ``toon`` — the TOON serialization format (compact, escaped pipe-delimited
  text) used as the interchange backend for CAD token sequences and
  vocabularies.
- ``toon_extended`` — schema-aware TOON, streaming / chunking, and LLM-prompt
  helpers built on top of ``toon``.
"""

from sdk.toon import (
    estimate_tokens,
    from_toon,
    to_toon,
)
from sdk.toon_extended import (
    build_prompt_for_llm,
    chunk_toon,
    stream_toon,
)
from sdk.toon_extended import (
    from_toon as from_toon_extended,
)
from sdk.toon_extended import (
    to_toon as to_toon_extended,
)

__all__ = [
    "build_prompt_for_llm",
    "chunk_toon",
    "estimate_tokens",
    "from_toon",
    "from_toon_extended",
    "stream_toon",
    "to_toon",
    "to_toon_extended",
]
