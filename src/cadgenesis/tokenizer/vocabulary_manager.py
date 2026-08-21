"""cadgenesis.tokenizer.vocabulary_manager
=========================================
Facade for the CAD vocabulary registry and lifecycle helpers.

Re-exports the canonical vocabulary API from :mod:`cadgenesis.tokenizer.vocabulary`
and the migration helpers from :mod:`cadgenesis.tokenizer.versioning`.
"""

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
    "CADVocabulary",
    "MigrationResult",
    "TokenFamily",
    "TokenRecord",
    "compare_versions",
    "migrate_vocabulary",
    "remap_ids",
]
