"""
cadgenesis.memory
=================
Layer-Integrated Memory System for CADGenesis-LM v6.0 (Pillar 6 complete).

Two complementary memory layers are exposed:

* **Neural bank** — :class:`MemoryPool` / :class:`LayerIntegratedMemorySystem`
  (torch slot vectors used by layer-integrated memory attention).
* **Semantic bank** — the eight domain pools (:class:`WorkingMemory`, ...) plus
  the opt-in :class:`LongTermMemory` ninth store, cross-pool
  :class:`MemoryRetrieval` (lexical, graph, symbolic, temporal, hybrid),
  :class:`MemoryRouter` (semantic + contextual routing),
  :class:`MemoryPruner` / compression tools, and v1+v2 :class:`MemoryPersistence`
  (snapshots, rollback, append log, file locks), composed behind the
  :class:`MemorySystem` facade.  :class:`SemanticMemoryBridge` renders semantic
  hits into neural slot vectors, and :mod:`cadgenesis.memory.augmentation`
  provides the memory-augmented transformer helpers.
"""

from cadgenesis.memory.augmentation import (
    ContextExpansion,
    MemoryAugmentedDecoding,
    MemoryRetrievalLayer,
    PersistentContextCache,
)
from cadgenesis.memory.bridge import SemanticMemoryBridge
from cadgenesis.memory.cad_memory import CADMemory
from cadgenesis.memory.compression import (
    AdaptivePruner,
    CompressionReport,
    EmbeddingCompressor,
    MemoryConsolidator,
    MemorySummarizer,
)
from cadgenesis.memory.engineering_memory import EngineeringMemory
from cadgenesis.memory.long_term_memory import LONG_TERM_POOL, LongTermMemory
from cadgenesis.memory.manufacturing_memory import ManufacturingMemory
from cadgenesis.memory.memory_common import (
    MemoryEntry,
    MemoryStore,
    SearchResult,
)
from cadgenesis.memory.memory_pools import (
    LayerIntegratedMemorySystem,
    MemoryPool,
)
from cadgenesis.memory.memory_router import MemoryRouter, RoutingDecision
from cadgenesis.memory.memory_system import MemorySystem
from cadgenesis.memory.persistence import MemoryPersistence
from cadgenesis.memory.project_memory import ProjectMemory
from cadgenesis.memory.pruning import MemoryPruner, PruningReport
from cadgenesis.memory.retrieval import MemoryRetrieval, RetrievalHit, RetrievalResult
from cadgenesis.memory.session_memory import SessionMemory
from cadgenesis.memory.simulation_memory import SimulationMemory
from cadgenesis.memory.user_memory import UserMemory
from cadgenesis.memory.working_memory import WorkingMemory

__all__ = [
    "LONG_TERM_POOL",
    "AdaptivePruner",
    "CADMemory",
    "CompressionReport",
    "ContextExpansion",
    "EmbeddingCompressor",
    "EngineeringMemory",
    "LayerIntegratedMemorySystem",
    "LongTermMemory",
    "ManufacturingMemory",
    "MemoryAugmentedDecoding",
    "MemoryConsolidator",
    "MemoryEntry",
    "MemoryPersistence",
    "MemoryPool",
    "MemoryPruner",
    "MemoryRetrieval",
    "MemoryRetrievalLayer",
    "MemoryRouter",
    "MemoryStore",
    "MemorySummarizer",
    "MemorySystem",
    "PersistentContextCache",
    "ProjectMemory",
    "PruningReport",
    "RetrievalHit",
    "RetrievalResult",
    "RoutingDecision",
    "SearchResult",
    "SemanticMemoryBridge",
    "SessionMemory",
    "SimulationMemory",
    "UserMemory",
    "WorkingMemory",
]
