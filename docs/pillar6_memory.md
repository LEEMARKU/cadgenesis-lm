# Pillar 6 — Layer-Integrated Memory

Implementation report for the **Layer-Integrated Memory** pillar of the
CADGenesis-LM v6.0 roadmap (`docs/v6_roadmap.md`). Closes the connectivity and
breadth gaps identified in `docs/pillar6_memory_audit.md`: it adds the ninth
semantic store (long-term), contextual routing, graph/symbolic/temporal/hybrid
retrieval, a compression layer, versioned persistence with snapshot/replay, a
semantic→neural bridge, transformer memory augmentation, a real replay buffer,
memory evaluation metrics and two integration bug fixes — all **additive** and
backward compatible (the 8-pool facade contract is untouched until a caller
opts in via `register_store`).

## 1. Scope (requirements → modules)

| # | Capability | Module |
|---|-----------|--------|
| 1 | Long-term store (9th pool, `LONG_TERM_POOL="long_term"`) | `memory/long_term_memory.py` |
| 2 | Additive store registration on the facade | `memory/memory_system.py` (`register_store`) |
| 3 | Context/task/confidence/agent-aware routing | `memory/memory_router.py` |
| 4 | Graph/symbolic/temporal/hybrid retrieval | `memory/retrieval.py` |
| 5 | Summarization, embedding compression, consolidation, adaptive pruning | `memory/compression.py` |
| 6 | Versioned persistence v2, snapshots/rollback, append log, file lock | `memory/persistence.py` |
| 7 | Semantic → neural bridge (hash-bag embeddings) | `memory/bridge.py` |
| 8 | Retrieval layer, augmented decoding, persistent cache, context expansion | `memory/augmentation.py` |
| 9 | Continual-learning replay buffer on `MemorySystem` | `continual_learning/replay_buffer.py` |
| 10 | Two integration bug fixes (`world_model` pool, `result.hits`) | `world_model/world_model.py`, `multimodal/integration.py` |
| 11 | Memory evaluation metrics + benchmark | `evaluation/memory_metrics.py`, `benchmarks/memory_benchmarks.py` |

## 2. Architecture

```
MemorySystem (facade, 8 default pools)
 ├── MemoryStore ──► LongTermMemory         # 9th store, register_store() opt-in
 ├── MemoryRouter  ──► route / route_by_context / route_by_task /
 │                    route_by_confidence / route_by_agent
 ├── MemoryRetrieval ─► retrieve / graph_search / symbolic_search /
 │                     temporal_search / hybrid_retrieve
 ├── MemoryPersistence ─► v1/v2 dumps, save/load, save_system/load_system,
 │                       snapshot/rollback, append/replay/truncate_log,
 │                       _FileLock (best-effort O_EXCL)
 ├── compression.py ──► MemorySummarizer / EmbeddingCompressor /
 │                     MemoryConsolidator / AdaptivePruner
 └── SemanticMemoryBridge ──► LayerIntegratedMemorySystem (neural bank)
      │                        render → to_vectors / write_pool / combined_bank
      └── augmentation.py ──► MemoryRetrievalLayer / MemoryAugmentedDecoding /
                             PersistentContextCache / ContextExpansion
```

The torch neural bank (`memory_pools.py`) and the `MemoryAttention` transformer
integration are untouched; the bridge and augmentation modules compose **on
top** of them.

## 3. Key APIs

| Component | API |
|-----------|-----|
| `LongTermMemory` | `consolidate(key, content, source, importance, metadata)`, `record_episode(key, summary, project_id)`, `recall(query, top_k)`, `episodes(top_k)` |
| `MemorySystem` | `register_store(name, store, keywords=None)` — registers into the store map, router and retriever |
| `MemoryRouter` | `route(query)`, `route_by_context(dict)`, `route_by_task(type)`, `route_by_confidence(query, confidence, low_pool, high_pool)`, `route_by_agent(role)` |
| `MemoryRetrieval` | `retrieve/retrieve_multi` (existing) + `graph_search(anchor, hop_count)`, `symbolic_search({facet: value})`, `temporal_search(query, since, until)`, `hybrid_retrieve(query, symbolic, temporal, keyword_weight)` |
| `MemorySummarizer` | `summarize(store, keys, summary_key, group, importance)` → merged record |
| `EmbeddingCompressor` | `compress(values, factor)`, `expansion_ratio`, `reconstruction_error` |
| `MemoryConsolidator` | `consolidate(source, target, query, importance_threshold, group)` → `CompressionReport(created_keys, consumed_keys)` |
| `AdaptivePruner` | `prune(store, value_threshold, recency_weight)` |
| `MemoryPersistence` | `dumps(version=None)` (v2 default, reads v1), `save/load`, `save_many/load_many`, `save_system/load_system` (multi-store), `snapshot/rollback`, `append/replay(since_timestamp)/truncate_log`, `_FileLock` |
| `SemanticMemoryBridge` | `embed_text(text)` (blake2b hash-bag, L2-normalised), `to_vectors(result, top_k, batch_size)` → (B,k,d), `write_pool(system, result, pool_name, max_slots)`, `combined_bank(system, result, batch_size, top_k)`, `stats()` |
| `MemoryRetrievalLayer` | `forward(x, context, mask)` split-head attention, `retrieve_and_attend(x, bridge, result, top_k)` |
| `MemoryAugmentedDecoding` | `forward(hidden, context, mode)` with `concat`/`sum`/`mean` |
| `PersistentContextCache` | `push(key, content, vector)`, `get(entry_id)`, `vectors()`, `clear()` |
| `ContextExpansion` | `expand(tokens, context)`, `fitted(used_tokens, context_tokens)` |
| `ReplayBuffer` | `store(experience)`, `store_many`, `sample(batch_size, strategy=uniform\|importance)`, `recall(query, top_k)`, `clear` |
| `evaluation/memory_metrics.py` | `precision_at_k`, `recall_at_k`, `mean_reciprocal_rank`, `routing_accuracy`, `memory_hit_rate`, `consolidation_ratio`, `compression_ratio`, `run_memory_benchmark()` |

## 4. Behavior notes

- **Backward compatibility**: `dumps(version=1)` writes the legacy v1 payload
  and `dumps()` (no arg) defaults to the new v2; `loads` reads both. The 8-pool
  facade is unchanged; `register_store` is purely additive.
- **Routing by confidence** guarantees the preferred pool ranks first by
  flooring its score at `max_other + 0.01`, so a low-confidence fallback to
  `working` is always honoured.
- **Keyword-only filtering** is applied where `MemoryStore.search()` can return
  0.0-score prefix matches: `temporal_search`, `ReplayBuffer.recall` and
  `MemoryConsolidator.consolidate(query=...)` all require token overlap via the
  importable `_tokens` helper.
- **Hybrid retrieval** is O(n) per store: `hit_by_key` is computed once per
  store instead of re-searching per record (fixed a 342 ms → 9.2 ms regression).
- **Bridge determinism**: `embed_text` uses a seeded blake2b hash-bag so the
  same text always yields the same vector; empty results render to a single
  zero slot so the attention path always has a tensor.
- **Persistence v2**: records carry a `version` field, store-level snapshots
  group all stores under one root, `append` writes an incremental JSONL log
  under a `_FileLock`, `replay(since_timestamp)` rebuilds state, and
  `rollback` restores a stored snapshot atomically.

## 5. Integration bug fixes (P6)

- `world_model/world_model.py` — `WorldModelSystem.persist()` now writes to the
  registered `"project"` pool instead of the unregistered `"world_model"`.
- `multimodal/integration.py` — `MultimodalIntegrator.retrieve_from_memory()`
  now returns `list(result.hits)` instead of the nonexistent `result.items`.

## 6. Quality gates

- **70 new tests** (`tests/memory/test_pillar6_*.py` × 6 files,
  `tests/continual_learning/test_replay_buffer.py`,
  `tests/evaluation/test_memory_metrics.py`) — all green; full suite
  **1385 passed** (was 1315, +70), legacy memory tests untouched.
- Ruff clean on all P6 source (remaining `memory_pools.py` / `continual_learning/__init__.py`
  findings are pre-existing legacy and untouched).
- Mypy clean on all P6 source (14 milestone files).
- `benchmarks/memory_benchmarks.py --reps 2`: retrieval lexical 1.4 ms,
  graph 0.01 ms, symbolic 0.13 ms, hybrid 1.7 ms; routing ~1.3–1.8 ms;
  summarize 0.008 ms, consolidate 0.003 ms, embed-compress 0.02 ms;
  v1-save 0.9 ms, v2-snapshot 1.9 ms, append 0.4 ms; bridge render 2.2 ms,
  attention 0.36 ms.

See `docs/pillar6_memory_audit.md` for the pre-implementation requirements
audit; this document records the delivered design and API.
