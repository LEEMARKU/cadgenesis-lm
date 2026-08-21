# M4 — Memory System Completeness

Milestone M4 of the CADGenesis-LM v6.0 Ultimate Architecture roadmap
(`docs/v6_roadmap.md`) completes the World Model (pillar 4) and Layer-Integrated
Memory System (pillar 6) pillars: every stub under `cadgenesis.memory` is
replaced by a tested, documented, production-quality module.

## Scope

The existing torch-based `LayerIntegratedMemorySystem` / `MemoryPool` (neural
slot-vector bank used by memory attention) is preserved unchanged.  M4 adds the
**semantic memory layer**: structured, dependency-free stores for CAD designs,
standards, manufacturing limits, simulation results, user preferences and
session/project context, plus cross-pool retrieval, routing, pruning and
persistence.

## Modules delivered

| Module | Contents |
| --- | --- |
| `memory_common.py` | `MemoryEntry` (record) and `MemoryStore` (bounded keyed store with importance, recency, keyword scoring, capacity enforcement, dict round-trip) |
| `working_memory.py` | `WorkingMemory` — short-term context buffer with remember/recall/context/squash |
| `session_memory.py` | `SessionMemory` — session-scoped records with begin/end, filtering, clear |
| `user_memory.py` | `UserMemory` — named preferences + design-style fingerprints |
| `project_memory.py` | `ProjectMemory` — project-scoped state, snapshots, attach/detach |
| `cad_memory.py` | `CADMemory` — feature trees & B-Rep records, kind filtering |
| `engineering_memory.py` | `EngineeringMemory` — standards (ISO/ASME/DIN) and guidelines |
| `manufacturing_memory.py` | `ManufacturingMemory` — process capabilities / DFM limits |
| `simulation_memory.py` | `SimulationMemory` — FEA/CFD results, analysis-type filtering |
| `retrieval.py` | `MemoryRetrieval` — cross-pool, deduplicated, weighted top-k retrieval (`RetrievalHit` / `RetrievalResult`) |
| `pruning.py` | `MemoryPruner` — capacity / staleness / importance / combined eviction policies (`PruningReport`) |
| `persistence.py` | `MemoryPersistence` — atomic JSON save/load, `save_many`/`load_many` |
| `memory_router.py` | `MemoryRouter` — pool affinity keywords, ranked routing, `best_pool` |
| `memory_system.py` | `MemorySystem` — facade composing all eight pools + retriever + router + pruner + persistence, `from_config(MemoryConfig)` |
| `__init__.py` | Package facade exporting the full public API (22 names) |

## Design notes

- **Pure Python.** The semantic memory layer is dependency-free, so it unit
  tests instantly and runs standalone; the torch neural bank remains available
  for layer-integrated attention.
- **Two complementary banks.** `memory_pools.py` holds differentiable slot
  vectors; M4's stores hold structured records. `MemorySystem` composes the
  semantic side while `LayerIntegratedMemorySystem` stays untouched.
- **Shared substrate.** All eight domain pools subclass `MemoryStore`; scoring,
  capacity and persistence machinery lives once in `memory_common.py`.
- **Composable services.** `MemoryRetrieval` merges ranked hits across pools;
  `MemoryRouter` picks the right pool first; `MemoryPruner` enforces memory
  budgets; `MemoryPersistence` round-trips every pool atomically.
- **Configurable.** `MemorySystem.from_config(MemoryConfig)` maps the
  `*_memory_slots` fields to pool capacities.

## Verification

```text
pytest           742 passed (74 new memory tests)
ruff check       clean for cadgenesis.memory (new modules) and tests/memory
audit_repo.py    181 modules · 298 public APIs · 16 182 LOC · 71 stubs
                 World Model pillar: OK
                 Layer-Integrated Memory pillar: OK
```
