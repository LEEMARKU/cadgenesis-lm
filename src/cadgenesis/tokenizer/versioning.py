"""
cadgenesis.tokenizer.versioning
================================
Vocabulary version management and layout migration for the CAD tokenizer.

Purpose
-------
Vocabularies evolve: token sets grow, slot layouts change, and persisted
vocabulary files outlive the code that wrote them.  This module provides:

* a versioned migration path between slot layouts (``migrate_vocabulary``);
* an id remapping helper so existing token sequences can be translated to a
  new vocabulary's id space (``remap_ids``);
* version comparisons for vocabulary files (``compare_versions``).

The low-level layout rebuild is implemented by
``CADVocabulary.migrate_layout``; this module adds the version metadata and
the id-remapping utilities used by preprocessing and inference.

Interfaces
----------
    result = migrate_vocabulary(vocab, new_slots, target_version="2.1.0")
    ids2    = remap_ids(ids, result.id_mapping, fallback_unk_id)
    compare_versions("2.0.0", "2.1.0")     # → -1, 0 or 1
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from cadgenesis.tokenizer.vocabulary import (
    DEFAULT_VOCAB_VERSION,
    CADVocabulary,
    TokenFamily,
)


@dataclass
class MigrationResult:
    """Report of a vocabulary layout migration."""

    source_version: str
    target_version: str
    source_layout: dict[str, int]
    target_layout: dict[str, int]
    id_mapping: dict[int, int] = field(default_factory=dict)
    unmapped_ids: list[int] = field(default_factory=list)
    vocab: CADVocabulary | None = None

    @property
    def preserved_ids(self) -> int:
        """Number of tokens whose ID survived the migration unchanged."""
        return sum(1 for old, new in self.id_mapping.items() if old == new)

    @property
    def remapped_ids(self) -> int:
        """Number of tokens whose ID changed during the migration."""
        return sum(1 for old, new in self.id_mapping.items() if old != new)

    @property
    def dropped_tokens(self) -> int:
        """Number of tokens that could not be carried into the new layout."""
        return len(self.unmapped_ids)


def compare_versions(a: str, b: str) -> int:
    """
    Compare two semantic version strings ``MAJOR.MINOR.PATCH``.

    Returns -1 if ``a < b``, 0 if equal, 1 if ``a > b``.  Non-numeric
    segments compare lexicographically as a tie-breaker.
    """

    def _segments(version: str) -> list[object]:
        parts: list[object] = []
        for chunk in version.split("."):
            try:
                parts.append(int(chunk))
            except ValueError:
                parts.append(chunk)
        return parts

    sa, sb = _segments(a), _segments(b)
    for x, y in zip(sa, sb, strict=False):
        if x == y:
            continue
        if isinstance(x, int) and isinstance(y, int):
            return -1 if x < y else 1
        assert isinstance(x, str) and isinstance(y, str)  # non-numeric segments compare as str
        return -1 if x < y else 1
    return (len(sa) > len(sb)) - (len(sa) < len(sb))


def migrate_vocabulary(
    vocab: CADVocabulary,
    new_slots: dict[TokenFamily, int],
    target_version: str | None = None,
) -> MigrationResult:
    """
    Migrate ``vocab`` onto ``new_slots``, preserving as many token IDs as
    possible and producing the old→new id mapping.

    The source vocabulary is never modified; the migrated copy is available
    as ``result.vocab``.
    """
    if target_version is None:
        target_version = DEFAULT_VOCAB_VERSION

    migrated = vocab.migrate_layout(new_slots, new_version=target_version)
    mapping, unmapped = migrated.last_migration or ({}, [])

    return MigrationResult(
        source_version=vocab.version,
        target_version=target_version,
        source_layout={fam.name: cap for fam, cap in vocab.slot_capacities().items()},
        target_layout={fam.name: cap for fam, cap in new_slots.items()},
        id_mapping=mapping,
        unmapped_ids=unmapped,
        vocab=migrated,
    )


def remap_ids(
    ids: Sequence[int],
    id_mapping: dict[int, int],
    fallback_unk_id: int,
) -> list[int]:
    """
    Translate an existing sequence of token ids into a migrated vocabulary's
    id space.  Ids absent from ``id_mapping`` (e.g. special ids that were
    dropped) map to ``fallback_unk_id``.
    """
    return [id_mapping.get(i, fallback_unk_id) for i in ids]
