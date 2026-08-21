"""
cadgenesis.tokenizer.statistics
================================
Corpus-level token statistics and compression metrics.

Purpose
-------
Provides a single ``compute_statistics`` entry point that aggregates token
usage across a corpus of token sequences (str sequences, id sequences, or
``CADTokenSequence`` objects) and reports:

* totals, sequence-length summary, unique-token count;
* per-family token counts and relative shares;
* unknown (out-of-vocabulary) rate;
* optional compression ratio when a compression function is supplied.

Used by preprocessing (corpus analysis / dataset filtering), training
(vocabulary evolution planning and logging) and reporting.

Interfaces
----------
    stats = compute_statistics(sequences, vocab, compress_fn=None)
    stats.to_dict()          # → JSON-serializable dict
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from cadgenesis.tokenizer.vocabulary import CADVocabulary

# Accepted token-sequence types: str ids, str tokens, or objects with a
# `.token_ids` / `.tokens` attribute (CADTokenSequence).  The final Any
# covers duck-typed token containers without a Sequence interface.
_SequenceLike = Sequence[int] | Sequence[str] | Any


def _iter_tokens(seq: _SequenceLike) -> list[object]:
    """Normalize any accepted sequence shape into a list of raw tokens."""
    if hasattr(seq, "cad_ids"):  # CADTokenSequence
        return list(seq.cad_ids)
    if hasattr(seq, "token_ids"):  # generic token container
        return list(seq.token_ids)
    return list(seq)


def _token_to_str(token, vocab: CADVocabulary) -> str:
    if isinstance(token, str):
        return token
    try:
        return vocab.record_of(token).token_str  # id → token string
    except (KeyError, TypeError):
        return f"<id:{token}>"


@dataclass
class CorpusStatistics:
    """Aggregate token statistics over a corpus."""

    num_sequences: int
    total_tokens: int
    unique_tokens: int
    mean_seq_len: float
    min_seq_len: int
    max_seq_len: int
    per_family_counts: dict[str, int]
    per_family_relative: dict[str, float]
    unknown_tokens: int = 0
    unknown_rate: float = 0.0
    top_tokens: list[tuple[str, int]] = field(default_factory=list)
    compression_ratio: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "num_sequences": self.num_sequences,
            "total_tokens": self.total_tokens,
            "unique_tokens": self.unique_tokens,
            "mean_seq_len": round(self.mean_seq_len, 4),
            "min_seq_len": self.min_seq_len,
            "max_seq_len": self.max_seq_len,
            "per_family_counts": self.per_family_counts,
            "per_family_relative": {k: round(v, 6) for k, v in self.per_family_relative.items()},
            "unknown_tokens": self.unknown_tokens,
            "unknown_rate": round(self.unknown_rate, 6),
            "top_tokens": self.top_tokens[:20],
            "compression_ratio": round(self.compression_ratio, 6),
        }


def compute_statistics(
    sequences: Sequence[_SequenceLike],
    vocab: CADVocabulary,
    compress_fn: Callable[[Sequence[str]], Sequence[str]] | None = None,
    top_k: int = 10,
) -> CorpusStatistics:
    """
    Compute corpus statistics for a list of token sequences.

    ``compress_fn``, when given, is applied to each sequence (as token
    strings) to measure the achieved compression ratio (1 - len(compressed) /
    len(original)); it should be e.g. ``tokenizer.compress_sequence``.
    It may return either the bare compressed sequence or a
    ``(compressed, ratio)`` tuple (as ``compress_sequence`` does).

    Raises ValueError if ``sequences`` is empty.
    """
    if not sequences:
        raise ValueError("Cannot compute statistics over an empty corpus.")

    lengths: list[int] = []
    family_counts: Counter = Counter()
    total_counts: Counter = Counter()
    unknown = 0
    raw_total = 0
    compressed_total = 0

    for seq in sequences:
        tokens = _iter_tokens(seq)
        raw_total += len(tokens)
        lengths.append(len(tokens))
        for token in tokens:
            tok_str = _token_to_str(token, vocab)
            total_counts[tok_str] += 1
            if (isinstance(token, str) and token not in vocab) or (
                isinstance(token, int) and token not in vocab
            ):
                unknown += 1
            if tok_str in vocab:
                family_counts[vocab.family_of(tok_str).name] += 1
            elif tok_str == "<unk>":
                family_counts["SPECIAL"] += 1

        if compress_fn is not None:
            raw_tokens = [_token_to_str(t, vocab) for t in tokens]
            compressed = compress_fn(raw_tokens)
            if (
                isinstance(compressed, tuple)
                and len(compressed) == 2
                and not isinstance(compressed[0], str)
            ):
                compressed = compressed[0]
            compressed_total += len(compressed)

    compression_ratio = 1.0
    if compress_fn is not None and raw_total > 0:
        compression_ratio = 1.0 - (compressed_total / raw_total)

    per_family_relative = {}
    for family in family_counts:
        per_family_relative[family] = family_counts[family] / raw_total

    return CorpusStatistics(
        num_sequences=len(sequences),
        total_tokens=raw_total,
        unique_tokens=len(total_counts),
        mean_seq_len=mean(lengths),
        min_seq_len=min(lengths),
        max_seq_len=max(lengths),
        per_family_counts=dict(family_counts),
        per_family_relative=per_family_relative,
        unknown_tokens=unknown,
        unknown_rate=unknown / raw_total if raw_total else 0.0,
        top_tokens=total_counts.most_common(top_k),
        compression_ratio=compression_ratio,
    )
