# Pillar 6 — Layer-Integrated Memory: Repository Audit

Audit performed before implementation (v6.0 roadmap). The neural
layer-integrated machinery (8 pools / 288 slots, `MemoryAttention`, per-layer
`refine`) and the semantic layer (8 stores, router, retriever, pruner,
persistence) are both complete and well-tested (74 memory unit tests). The P6
gaps are **connectivity and breadth**; everything below is additive behind
feature flags with zero impact on the existing 8-pool contract.

## 1. Existing memory modules

- `memory_common.py` — `MemoryEntry(key, content, pool, importance, metadata,
  created_at, last_access, access_count)` (+`touch/text/to_dict/from_dict`),
  `SearchResult`, `MemoryStore` (bounded keyed store, keyword+recency+importance
  scoring, capacity eviction, search/top/summary/serialization).
- `memory_pools.py` — neural bank: `MemoryPool(num_slots, d_model)` slot
  vectors; `LayerIntegratedMemorySystem(d_model, slots)` — 8 pools / 288 slots,
  `get_combined_memory_bank`, `retrieve` (cosine top-k), `refine` (differentiable
  per-layer write-back into the working region).
- `memory_system.py` — `MemorySystem` facade: `POOLS` (8), `remember/recall/
  retrieve/route/forget/prune/save/load/summary`, composed of `MemoryRetrieval`,
  `MemoryRouter`, `MemoryPruner`, `MemoryPersistence`.
- 8 domain stores (all subclass `MemoryStore`): `WorkingMemory`, `SessionMemory`,
  `UserMemory`, `ProjectMemory`, `CADMemory`, `EngineeringMemory`,
  `ManufacturingMemory`, `SimulationMemory`.
- `retrieval.py` — `MemoryRetrieval.retrieve` (lexical cross-pool merge),
  `retrieve_multi`, `RetrievalHit`, `RetrievalResult(.top, .by_pool())`.
- `memory_router.py` — `MemoryRouter.route(query)` (affinity keywords +
  content overlap only).
- `pruning.py` — `MemoryPruner` (`capacity/staleness/importance/combined`).
- `persistence.py` — atomic full-JSON `save/load/save_many/load_many`, v1.

## 2. Memory types implemented

8 semantic stores + 8 neural pools. **Missing:** long-term store and agent
memory (P6 spec has 9 stores; agent memory is handled by Pillar 5's
`LayeredSharedMemory`).

## 3. Transformer integration status

- `MemoryAttention(d_model, num_heads)` in `transformer/attention.py`,
  re-exported by `transformer/memory_attention.py`; wired into
  `MultiHeadAttentionMixture`, `CADTransformerBlock`, `EncoderStack`,
  `DecoderStack`, `GeometryAwareTransformer`, `hierarchical_transformer`,
  `self_designing` (config `memory_attn_heads`).
- Per-layer differentiable `refine` write-back: exists.
- **Missing:** retrieval layer over the *semantic* `MemorySystem` (slot vectors
  are random unless written), memory-augmented decoding, context expansion,
  persistent context/KV cache across calls.

## 4. Routing / retrieval / compression / persistence gaps

- Routing: only semantic (1 of 5 modes). Missing context/task/confidence/
  agent-aware.
- Retrieval: semantic layer is lexical-only. Missing graph/symbolic/hybrid/
  temporal. (Vector exists only in the torch bank.)
- Compression: only heuristic pruning. Missing summarization, embedding
  compression, hierarchical (working→long-term consolidation), adaptive
  (learned) pruning.
- Persistence: static v1 JSON. Missing versioning/migration, system snapshot/
  rollback, incremental append, sync/locking.

## 5. Integration points (verified call sites)

- `world_model/integration.py` — `store(graph, memory)` uses `"engineering"`
  pool; `conditioned_reason` retrieves priors.
- `world_model/world_model.py` — `WorldModelSystem.persist()` calls
  `remember("world_model", ...)` → **KeyError bug** (pool not registered).
- `multimodal/integration.py` — `MultimodalIntegrator.retrieve()` reads
  `result.items` → **returns [] bug** (`RetrievalResult` exposes `.hits`).
- `agents/coordinator.py` — `AgentCoordinator(memory=SharedMemory)` blackboard;
  never touches `MemorySystem`.
- `cad/integration/memory_bridge.py` — `CADMemoryBridge` wraps `CADMemory`.
- `continual_learning/` — 7 empty stubs; no memory wiring.

## 6. Duplicated logic

- `SharedMemory` vs `MemoryStore` (parallel KV stores, no bridge).
- Three retrieval implementations (per-store `search`, `MemoryRetrieval`,
  `MemoryRouter.retrieve`) with duplicated scoring.
- Two memory systems (neural vs semantic) with **no bridge**.
- Overlapping snapshot/serialization: `WorldModelState`, `ProjectMemory.snapshot`,
  `MemoryPersistence`.

## 7. Architecture plan (backward compatible)

1. `long_term_memory.py` — 9th store (extended mode, off by default so the
   8-pool invariant and all 74 tests stay green).
2. `memory_system.py` — additive `register_store(name, store)`.
3. `memory_router.py` — additive `route_by_context/task/confidence/agent`.
4. `retrieval.py` — additive `graph_search/symbolic_search/temporal_search/
   hybrid_retrieve`.
5. `compression.py` — `MemorySummarizer`, `EmbeddingCompressor`,
   `MemoryConsolidator`, `AdaptivePruner`.
6. `persistence.py` — v2 versioned records (reads v1), `snapshot/rollback`,
   incremental append log, sync lock.
7. `bridge.py` — `SemanticMemoryBridge` renders semantic hits into neural slot
   vectors so `MemoryAttention` attends to stored knowledge.
8. `augmentation.py` — `MemoryRetrievalLayer` (torch), `MemoryAugmentedDecoding`,
   `PersistentContextCache`, `ContextExpansion` — composable over the existing
   transformer without changing existing forward paths.
9. `continual_learning/replay_buffer.py` — real implementation using
   `MemorySystem` as substrate.
10. Fix the two integration bugs (`world_model` pool, `result.hits`).
11. `evaluation/memory_metrics.py` + benchmarks + docs + tests.
