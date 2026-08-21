"""cadgenesis.knowledge_network
==============================
Global engineering knowledge network (v6.0, Pillar 7 / Pillar 18 seed).

Aggregates multiple knowledge sources (knowledge graphs, standards libraries,
semantic memory mirrors) behind one query surface: registered sources are
searched in parallel, results merged by score, and the union can be rendered
into a single :class:`~cadgenesis.reasoning.knowledge_graph.KnowledgeGraph`
for symbolic traversal.
"""

from cadgenesis.knowledge_network.network import (
    KnowledgeHit,
    KnowledgeNetwork,
    KnowledgeSource,
)
from cadgenesis.knowledge_network.sources import (
    KnowledgeGraphSource,
    StandardsSource,
)

__all__ = [
    "KnowledgeGraphSource",
    "KnowledgeHit",
    "KnowledgeNetwork",
    "KnowledgeSource",
    "StandardsSource",
]
