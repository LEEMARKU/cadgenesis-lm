"""
cadgenesis.tokenizer.evolution
==============================
Vocabulary Evolution for the Autonomous CAD Tokenizer.

This module implements the *autonomous* half of the tokenizer: it watches how
the model's vocabulary is actually used on a corpus and proposes how the
vocabulary should grow.

    * New tokens          — frequent unknown CAD tokens are registered.
    * Composite merges    — frequent adjacent pairs are merged into a single
                             composite token (compression / faster decoding).
    * Remapping           — existing sequences are re-encoded against the
                             upgraded vocabulary.

The evolution loop is deliberately cheap (pure Python + the existing
``CADVocabulary``) and deterministic given a corpus.  Nothing is rebuilt: the
model reuses its embedding tables; growing the vocabulary simply extends the
token space the model can emit / consume.

Complexity
----------
    analyze():  O(N)   over the tokenised corpus (single pass counters)
    apply():    O(ops · V)  with ops = planned operations
    remap():    O(N)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from cadgenesis.tokenizer.vocabulary import CADVocabulary, TokenFamily

# ---------------------------------------------------------------------------
# TokenUpgrade / VocabularyUpgradePlan
# ---------------------------------------------------------------------------


@dataclass
class TokenUpgrade:
    """A single vocabulary mutation proposed by the evolution engine."""

    op: str  # "register" | "merge" | "split"
    token: str
    family: TokenFamily
    args: dict = field(default_factory=dict)
    reason: str = ""
    score: float = 0.0

    def __repr__(self) -> str:
        return f"TokenUpgrade({self.op!r}, {self.token!r} → {self.family.name}, s={self.score:.1f})"


@dataclass
class VocabularyUpgradePlan:
    """An ordered list of vocabulary mutations plus corpus statistics."""

    operations: list[TokenUpgrade]
    stats: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.operations)

    def __len__(self) -> int:
        return len(self.operations)


# ---------------------------------------------------------------------------
# TokenFrequencyTracker — single-pass corpus statistics
# ---------------------------------------------------------------------------


class TokenFrequencyTracker:
    """Counts token usage, unknown-token occurrences and adjacent pairs."""

    def __init__(self) -> None:
        self.token_counts: Counter = Counter()
        self.pair_counts: Counter = Counter()
        self.unknown_counts: Counter = Counter()
        self.total_tokens = 0
        self.total_sequences = 0

    def observe_tokens(self, tokens: list[str], vocab: CADVocabulary | None = None) -> None:
        self.total_sequences += 1
        for i, token in enumerate(tokens):
            self.token_counts[token] += 1
            self.total_tokens += 1
            if vocab is not None and token not in vocab:
                self.unknown_counts[token] += 1
            if i > 0:
                self.pair_counts[(tokens[i - 1], token)] += 1

    def observe(self, seq, vocab: CADVocabulary | None = None) -> None:
        """
        Observe a ``CADTokenSequence``.  Uses ``raw_cad_tokens`` when present,
        otherwise falls back to decoding ids through ``vocab``.
        """
        if seq.raw_cad_tokens:
            self.observe_tokens(list(seq.raw_cad_tokens), vocab)
        elif vocab is not None:
            self.observe_tokens([vocab.record_of(i).token_str for i in seq.cad_ids], vocab)

    def observe_many(self, sequences, vocab: CADVocabulary | None = None) -> None:
        for seq in sequences:
            self.observe(seq, vocab)

    def report(self, top_k: int = 20) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "total_sequences": self.total_sequences,
            "unique_tokens": len(self.token_counts),
            "top_tokens": self.token_counts.most_common(top_k),
            "top_pairs": self.pair_counts.most_common(top_k),
            "top_unknowns": self.unknown_counts.most_common(top_k),
        }


# ---------------------------------------------------------------------------
# Family guessing for unknown tokens
# ---------------------------------------------------------------------------


def guess_family(token_str: str) -> TokenFamily:
    """
    Heuristic family assignment for unknown CAD token strings.

    Explicit prefixes win; otherwise uppercase/underscore names are treated as
    geometry primitives.  Override by passing a custom ``family_guesser``.
    """
    if token_str.startswith("NUM_") or token_str.startswith("ANG_"):
        return TokenFamily.NUMERIC
    if token_str.startswith("<") and token_str.endswith(">"):
        return TokenFamily.SPECIAL
    if token_str.startswith(("MAT_", "AL_", "STEEL", "ALUM")):
        return TokenFamily.MATERIAL
    if token_str.startswith(("MATE_", "COINCIDENT", "CONCENTRIC")):
        return TokenFamily.CONSTRAINT
    if token_str.startswith(("SIM_", "FEA_", "STRESS")):
        return TokenFamily.SIMULATION
    return TokenFamily.GEOMETRY


# ---------------------------------------------------------------------------
# VocabularyEvolution — the autonomous growth engine
# ---------------------------------------------------------------------------


class VocabularyEvolution:
    """
    Monitors token usage and grows the vocabulary to match observed usage.

    Parameters
    ----------
    vocab : CADVocabulary
        Vocabulary to evolve (mutated in place by ``apply``/``evolve``).
    min_frequency : int
        An unknown token must appear at least this many times to be proposed.
    min_pair_frequency : int
        An adjacent pair must co-occur at least this many times to be merged.
    max_ops : int
        Cap on proposed operations per analysis.
    """

    def __init__(
        self,
        vocab: CADVocabulary,
        min_frequency: int = 3,
        min_pair_frequency: int = 5,
        max_ops: int = 64,
    ) -> None:
        self.vocab = vocab
        self.min_frequency = min_frequency
        self.min_pair_frequency = min_pair_frequency
        self.max_ops = max_ops

    # ------------------------------------------------------------ analysis

    def analyze(
        self,
        sequences,
        family_guesser: Callable[[str], TokenFamily] | None = None,
    ) -> VocabularyUpgradePlan:
        """Produce a plan of vocabulary upgrades from a corpus of sequences."""
        guesser = family_guesser or guess_family
        tracker = TokenFrequencyTracker()
        for seq in sequences:
            tracker.observe(seq, self.vocab)

        ops: list[TokenUpgrade] = []

        # 1) Frequent unknown tokens → register them.
        for token, count in tracker.unknown_counts.most_common():
            if len(ops) >= self.max_ops:
                break
            if count < self.min_frequency:
                break
            if token in self.vocab:
                continue
            family = guesser(token)
            if self.vocab.remaining_slots(family) <= 0:
                continue
            ops.append(
                TokenUpgrade(
                    op="register",
                    token=token,
                    family=family,
                    args={"count": count},
                    reason=f"frequent unknown (x{count})",
                    score=float(count),
                )
            )

        # 2) Frequent adjacent pairs → composite merge token.
        for (a, b), count in tracker.pair_counts.most_common():
            if len(ops) >= self.max_ops:
                break
            if count < self.min_pair_frequency:
                break
            merged = f"{a}_{b}"
            if merged in self.vocab:
                continue
            family = self.vocab.family_of(a) if a in self.vocab else guesser(a)
            if self.vocab.remaining_slots(family) <= 0:
                continue
            ops.append(
                TokenUpgrade(
                    op="merge",
                    token=merged,
                    family=family,
                    args={"parts": [a, b], "count": count},
                    reason=f"frequent pair (x{count})",
                    score=float(count),
                )
            )

        return VocabularyUpgradePlan(operations=ops, stats=tracker.report())

    # -------------------------------------------------------------- apply

    def apply(self, plan: VocabularyUpgradePlan) -> tuple[list, list]:
        """
        Execute a plan against the vocabulary.  Returns
        ``(new_records, applied_upgrades)``.  Ops that would overflow or
        duplicate are skipped, never raised.
        """
        records: list = []
        applied: list[TokenUpgrade] = []
        for op in plan.operations:
            if op.op == "register":
                if op.token in self.vocab or self.vocab.remaining_slots(op.family) <= 0:
                    continue
                records.append(self.vocab.register(op.token, op.family, op.reason))
                applied.append(op)
            elif op.op == "merge":
                if op.token in self.vocab or self.vocab.remaining_slots(op.family) <= 0:
                    continue
                try:
                    records.append(
                        self.vocab.merge_tokens(op.args["parts"], op.token, op.family, op.reason)
                    )
                    applied.append(op)
                except KeyError:
                    continue
            elif op.op == "split":
                try:
                    recs = self.vocab.split_token(op.args["token"], op.args["parts"], op.family)
                    records.extend(recs)
                    applied.append(op)
                except KeyError:
                    continue
        return records, applied

    # ------------------------------------------------------------- evolve

    def evolve(
        self,
        sequences,
        family_guesser: Callable[[str], TokenFamily] | None = None,
    ) -> tuple[VocabularyUpgradePlan, list, list]:
        """Analyse a corpus and apply the resulting plan in one call."""
        plan = self.analyze(sequences, family_guesser)
        records, applied = self.apply(plan)
        return plan, records, applied

    # -------------------------------------------------------------- remap

    def remap_sequence(
        self,
        cad_tokens: list[str],
        plan: VocabularyUpgradePlan | None = None,
    ) -> list[str]:
        """
        Re-encode a CAD token list against the (evolved) vocabulary:
        merges adjacent pairs whose composite token now exists and registers
        unknowns (plan or on-the-fly).  Returns the remapped token list.
        """
        out: list[str] = []
        i = 0
        while i < len(cad_tokens):
            token = cad_tokens[i]
            if i + 1 < len(cad_tokens):
                merged = f"{token}_{cad_tokens[i + 1]}"
                if merged in self.vocab:
                    out.append(merged)
                    i += 2
                    continue
            out.append(token)
            i += 1
        return out

    def remap_sequences(
        self,
        sequences,
        plan: VocabularyUpgradePlan | None = None,
    ) -> list[list[str]]:
        """Remap a batch of CAD token lists after an evolution step."""
        return [
            self.remap_sequence(list(s.raw_cad_tokens) if s.raw_cad_tokens else s, plan)
            for s in sequences
        ]
