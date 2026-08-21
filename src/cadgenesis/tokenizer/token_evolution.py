"""cadgenesis.tokenizer.token_evolution
=====================================
Facade for the autonomous vocabulary-growth engine.

Re-exports the canonical evolution API from :mod:`cadgenesis.tokenizer.evolution`.
"""

from cadgenesis.tokenizer.evolution import (
    TokenFrequencyTracker,
    TokenUpgrade,
    VocabularyEvolution,
    VocabularyUpgradePlan,
    guess_family,
)

__all__ = [
    "TokenFrequencyTracker",
    "TokenUpgrade",
    "VocabularyEvolution",
    "VocabularyUpgradePlan",
    "guess_family",
]
