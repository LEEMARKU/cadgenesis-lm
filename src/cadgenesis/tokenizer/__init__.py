"""
cadgenesis.tokenizer
====================
Autonomous CAD Tokenizer — Phase 1 of CADGenesis-LM v2.0.

Sub-modules:
    vocabulary      — Extensible multi-modal vocabulary registry
    geometry        — Geometric primitive + B-Rep tokens
    feature         — CAD feature operation tokens (extrude, fillet, …)
    constraint      — Parametric constraint tokens
    material        — Material + property tokens
    assembly        — Assembly relationship tokens
    manufacturing   — Manufacturing process tokens
    simulation      — Simulation + physics tokens
    language        — Text tokenizer (BPE-compatible)
    evolution       — Autonomous vocabulary growth engine
    toon_backend    — TOON serialization backend adapter
    versioning      — Vocabulary versioning + layout migration
    statistics      — Corpus token statistics + compression metrics
    cad_tokenizer   — Unified AutonomousCADTokenizer orchestrator
"""

from cadgenesis.tokenizer.cad_tokenizer import (
    AutonomousCADTokenizer,
    CADTokenSequence,
    MultiModalBatch,
    restore_vocab_tokens,
    vocab_tokens,
)
from cadgenesis.tokenizer.evolution import (
    TokenFrequencyTracker,
    TokenUpgrade,
    VocabularyEvolution,
    VocabularyUpgradePlan,
    guess_family,
)
from cadgenesis.tokenizer.statistics import (
    CorpusStatistics,
    compute_statistics,
)
from cadgenesis.tokenizer.toon_backend import ToonBackend
from cadgenesis.tokenizer.versioning import (
    MigrationResult,
    compare_versions,
    migrate_vocabulary,
    remap_ids,
)
from cadgenesis.tokenizer.vocabulary import (
    DEFAULT_VOCAB_VERSION,
    CADVocabulary,
    TokenFamily,
    TokenRecord,
)

__all__ = [
    "DEFAULT_VOCAB_VERSION",
    "AutonomousCADTokenizer",
    "CADTokenSequence",
    "CADVocabulary",
    "CorpusStatistics",
    "MigrationResult",
    "MultiModalBatch",
    "TokenFamily",
    "TokenFrequencyTracker",
    "TokenRecord",
    "TokenUpgrade",
    "ToonBackend",
    "VocabularyEvolution",
    "VocabularyUpgradePlan",
    "compare_versions",
    "compute_statistics",
    "guess_family",
    "migrate_vocabulary",
    "remap_ids",
    "restore_vocab_tokens",
    "vocab_tokens",
]
