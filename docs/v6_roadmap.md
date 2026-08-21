# CADGenesis-LM v6.0 — Ultimate Architecture Implementation Roadmap

This document is the canonical, actionable roadmap for transforming CADGenesis-LM
into the **v6.0 Ultimate Architecture**. It maps the 20 architectural pillars to
concrete modules under `src/cadgenesis/`, identifies what already exists versus
what is still stubbed, and sequences the work into verifiable milestones.

## 0. Guiding rules

1. **Never remove working features.** Every existing capability is preserved.
2. **Never replace stable modules without justification.** Upgrades must be
   additive and backward compatible.
3. **No placeholder implementations, no TODOs, no mock components.**
4. **SOLID + modular architecture.** Each module exposes a stable, documented API.
5. **Everything configurable** via `CADConfig` (single source of truth).
6. **Tested, documented, integrated, verified** before a pillar is declared done.
7. Milestones are executed **one at a time**; after each milestone a repository
   audit (`python scripts/audit_repo.py`) must pass before starting the next.

## 1. Pillar → Module map

| # | Pillar | Primary modules | Status (baseline) |
|---|--------|-----------------|-------------------|
| 1 | Foundation Model | `transformer/`, `tokenizer/`, `inference/`, `training/` | Core implemented; sub-modules stubbed |
| 2 | CAD Intelligence | `tokenizer/*_tokens`, `reasoning/`, `execution/` | Partial |
| 3 | Multimodal Understanding | `multimodal/`, `datasets/multimodal.py`, `evaluation/multimodal_metrics.py` | Implemented (M19) |
| 4 | World Model | `world_model/`, `evaluation/world_model_metrics.py` | Implemented (M19) |
| 5 | Multi-Agent Intelligence | `agents/` | Implemented (M19) |
| 6 | Layer-Integrated Memory System | `memory/` | Implemented (M19) |
| 7 | Neuro-Symbolic Reasoning | `reasoning/`, `knowledge_network/` | Implemented (M19) |
| 8 | CAD Execution & Validation | `execution/` | Basic engine done; backends/validation stubbed |
| 9 | Learning System | `training/`, `continual_learning/`, `adapters/`, `distillation/` | Trainer done; most stubbed |
| 10 | Reliability & Confidence | `confidence/`, `monitoring/`, `evaluation/` | Core engine done; most stubbed |
| 11 | Production Platform | `serving/`, `cli/`, `optimization/`, `config/`, `telemetry/`, `logging/` | Stubbed |
| 12 | Research Infrastructure | `evaluation/`, `datasets/`, `scripts/`, `experiment registry` | Stubbed |
| 13 | Blockchain Infrastructure (provenance/auditability) | `provenance/`, `ledger/` | New |
| 14 | Research Economy | `research_economy/` (reproducibility registry) | New |
| 15 | Quantum Research Interfaces (optional) | `quantum/` | New (optional backend) |
| 16 | Frontier AI Research Laboratory | `research_lab/frontier/` | New |
| 17 | Autonomous AI Research Laboratory (human-approval gated) | `research_lab/autonomous/` | New |
| 18 | Global Engineering Knowledge Network | `reasoning/knowledge_graph`, `knowledge_network/` | Implemented (M19) |
| 19 | Industrial Digital Twin Integration | `digital_twin/`, `execution/`, `simulation/` | New |
| 20 | Autonomous Engineering Platform | `platform/`, `agents/`, `execution/` | New |

## 2. Baseline audit (v2.0, 300 tests passing)

Implemented and covered by tests:

- `tokenizer/` — vocabulary, evolution, TOON backend, statistics, versioning,
  numeric, language, geometry, feature, material, manufacturing, simulation,
  constraint, assembly, legacy shim.
- `transformer/` — attention mixture, transformer block, MoE, positional
  encodings, efficient attention, interaction, geometry transformer,
  `self_designing/` (NAS, routing, adaptive heads, pruning, rollback).
- `memory/memory_pools.py` — 8 pools / 288 slots.
- `agents/multi_agent_system.py` — internal 8-agent transformer bus.
- `reasoning/neuro_symbolic.py` — neural↔symbolic projection.
- `execution/execution_engine.py` — basic execution pipeline.
- `adapters/lora.py` + `manager.py` — LoRA application + adapter bank.
- `distillation/distillation_engine.py` + `distill_pipeline.py` — multi-teacher.
- `alignment/constitutional_ai.py` — constitutional principles + safety.
- `confidence/confidence_engine.py` — sequence confidence.
- `inference/engine.py` — greedy/beam engine.
- `training/trainer.py` + `cli/train.py` — training loop + CLI.
- `config/cad_config.py` — single source of truth.
- `sdk/` — TOON + extended serialization.

Stubbed (empty `from __future__ import annotations` bodies): ~160 modules across
`adapters/`, `agents/`, `cli/`, `confidence/`, `continual_learning/`,
`distillation/`, `evaluation/`, `execution/`, `logging/`, `memory/`,
`monitoring/`, `optimization/`, `reasoning/`, `serving/`, `telemetry/`,
`training/`, `transformer/`, `utils/`, `tokenizer/` (token defs & facades).

## 3. Milestone plan

Each milestone is a vertical slice: **implement → unit tests → integration tests →
docs → config integration → audit**.

### M0 — Roadmap & audit tooling (this document + `scripts/audit_repo.py`)
Deliverables: roadmap, automated audit script, version bump to 6.0.0.

### M1 — Foundations: utilities & observability
Pillars 11 (partial), 12 (partial). Modules:
`utils/` (decorators, filesystem, hashing, math, time), `logging/` (config,
emitter), `telemetry/` (metrics, tracing, logs), `monitoring/` (health, drift,
alerts).

### M2 — Foundation Model completeness
Pillar 1. `transformer/`: `embeddings.py`, `encoder.py`, `decoder.py`,
`heads.py`, `losses.py`, `transformer.py` facade, `positional_encoding.py`
shim. `tokenizer/`: token-definition modules (`cad_tokens.py`,
`geometry_tokens.py`, ...), `compression.py`, `serialization.py`,
`validation.py`, `tokenizer.py` facade.

### M3 — CAD Intelligence & Neuro-Symbolic Reasoning
Pillars 2, 7. `reasoning/`: `rule_engine.py`, `constraint_solver.py`,
`geometry_reasoner.py`, `knowledge_graph.py`, `manufacturing_rules.py`,
`planner.py`, `symbolic_reasoner.py`, `topology.py`, `validator.py`.

### M4 — Memory System completeness
Pillars 4, 6. `memory/`: `retrieval.py`, `persistence.py`, `pruning.py`,
`memory_router.py`, domain pools (`working`, `session`, `user`, `project`,
`cad`, `engineering`, `manufacturing`, `simulation`).

**Extended (M19) to full P6:** `long_term_memory.py` (9th store),
`memory_system.py` (`register_store`), `memory_router.py`
(`route_by_context/task/confidence/agent`), `retrieval.py`
(`graph_search/symbolic_search/temporal_search/hybrid_retrieve`),
`compression.py` (summarizer, embedding compressor, consolidator, adaptive
pruner), `persistence.py` (v2 records, `save_system`/`load_system`,
`snapshot`/`rollback`, append/replay log, `_FileLock`), `bridge.py`
(`SemanticMemoryBridge`), `augmentation.py` (retrieval layer, augmented
decoding, persistent context cache, context expansion), replay buffer in
`continual_learning/`, memory metrics, two integration bug fixes. See
`docs/pillar6_memory.md`.

### M5 — Multi-Agent Intelligence
Pillar 5. `agents/`: `message_bus.py`, `coordinator.py`, `scheduler.py`,
`consensus.py`, `shared_memory.py`, plus role packages (`planner`, `geometry`,
`constraint`, `manufacturing`, `optimization`, `assembly`, `simulation`,
`validation`).

**Extended (M19) to full P5:** `infrastructure.py`, `versioning.py`,
`registry.py`, `loader.py`, `plugins.py`, `health.py`, `event_bus.py`,
`scheduling.py` (DAG/priority/deadline/dynamic), extended `consensus.py`,
`shared_memory.py` (LayeredSharedMemory), `pipeline.py` (task planning),
`orchestrator.py` (AgentPlatform), `integration.py` (platform adapters),
`fleet.py` (18-agent fleet) and 10 specialized role agents. See
`docs/pillar5_multi_agent.md`.

### M6 — CAD Execution & Validation
Pillar 8. `execution/`: `exporter.py`, `geometry_validation.py`,
`manufacturing.py`, `cost_estimation.py`, `topology_analysis.py`,
`optimization.py`, `simulation.py`, `feedback.py`, `freecad_engine.py`,
`opencascade_engine.py`.

### M7 — Reliability & Confidence
Pillar 10. `confidence/`: `calibration.py`, `uncertainty.py`, `risk.py`,
`fallback.py`, `monitoring.py`, `confidence.py` facade. `evaluation/` metric
modules.

### M8 — Learning System: training infrastructure
Pillar 9. `training/`: `optimizer.py`, `scheduler.py`, `checkpoint.py`,
`callbacks.py`, `distributed.py`, `fsdp.py`, `deepspeed.py`, `metrics.py`,
`profiler.py`.

### M9 — Continual learning & adapters
Pillars 9, 12. `continual_learning/` (replay_buffer, ewc, knowledge_anchor,
adapter_isolation, updater, evaluator, continual_trainer). `adapters/`
(lifecycle, versioning, router, promotion, rollback, peft, qlora).

### M10 — Distillation & alignment completeness
Pillars 9. `distillation/` (soft_labels, hard_labels, rlaif, critique,
consensus, synthetic, pipeline, teachers).

### M11 — Evaluation & reproducibility (Research Infrastructure)
Pillar 12. `evaluation/` (cad_metrics, geometry_metrics, reasoning_metrics,
tokenizer_metrics, benchmark_runner, report_generator). `datasets/` loaders.

### M12 — Production Platform: serving & CLI & optimization
Pillar 11. `serving/` (api, batching, grpc, lifecycle). `cli/` (config, eval,
generate, serve). `optimization/` (quantization, pruning, onnx, kernels).

### M13 — Provenance & auditability (Blockchain)
Pillar 13. `provenance/` (hash-chain ledger, asset registration, verification,
anchoring adapter) — engineering purpose limited to provenance/auditability.

### M14 — Research economy & reproducibility
Pillar 14. `research_economy/` (experiment registry, reproducible bundles,
collaboration workflows).

### M15 — Frontier & Autonomous AI Research Laboratories
Pillars 16, 17. `research_lab/` — experiment orchestration, self-improvement
loops, and the **mandatory human-approval gate** before any architectural change
is adopted (persisted approval records).

### M16 — Quantum research interfaces (optional)
Pillar 15. `quantum/` — optional backend abstraction for optimization
experiments (QAOA/QUBO-style), inert unless explicitly enabled.

### M17 — Knowledge network, digital twin, autonomous platform
Pillars 18, 19, 20. `knowledge_network/`, `digital_twin/`, `platform/`.

### M18 — Final integration, documentation, benchmarks, acceptance audit
Cross-pillar integration tests, complete docs, reproducible benchmarks,
final acceptance run of `scripts/audit_repo.py`.

### M19 — Pillars 3 & 4: Multimodal understanding and world model
Pillars 3, 4. Implemented the full multimodal stack (`multimodal/`: common,
embeddings, 11 encoders, cross-modal, fusion, facade, integration), the
world-model stack (`world_model/`: objects, spatial, mechanical, functional,
assembly, affordances, design_intent, simulator, planning, facade,
integration) and their datasets/metrics/benchmarks/docs. See
`docs/pillar3_multimodal.md` and `docs/pillar4_world_model.md`.

### M19 — Pillars 5 & 6: Multi-agent intelligence and layer-integrated memory
Pillars 5, 6. P5 delivered the orchestration platform in `agents/` (registry,
loader, plugins, health, event bus, DAG scheduling, layered shared memory,
extended consensus, pipeline, 18-agent fleet, `AgentPlatform`, integration
adapters). P6 completed the memory pillar (`memory/long_term_memory.py`,
`register_store`, contextual routing, graph/symbolic/temporal/hybrid
retrieval, compression, versioned persistence + snapshot/replay,
`SemanticMemoryBridge`, transformer augmentation, replay buffer, memory
metrics, two integration bug fixes). See `docs/pillar5_multi_agent.md` and
`docs/pillar6_memory.md`.

### M19 — Pillar 7: Neuro-symbolic reasoning
Pillar 7 (plus pillar 18, the global engineering knowledge network). The
`reasoning/` core engines (rules, constraints, geometry, topology, symbolic
reasoner, neuro-symbolic engine) already existed; this milestone added
backward chaining and rule versioning (`prove`/`prove_all`, `Proof`,
`snapshot`/`by_version`/`diff`), the engineering standards library
(`reasoning/standards.py`: ISO 286/261/1302, ASME B4.1/Y14.5, DIN, ANSI,
company rules → `build_standards_graph`), casting/welding/tooling/tolerance
manufacturability checks, the best-first `SymbolicPlanner`, feature-dependency
and topology adjacency reasoning, constraint propagation/conflict
detection/repair, the `HybridReasoningPipeline` (critical/blocked semantics,
neural refinement, `explain()`), the new `knowledge_network/` package
(`KnowledgeNetwork` over the knowledge graph and standards), implemented
`evaluation/reasoning_metrics.py`, `benchmarks/reasoning_benchmarks.py`, and
the `NeuroSymbolicReasoningEngine.forward()` integration fix for
`agents/integration.py::NeuroSymbolicAdapter`. 106 new tests (full suite
1491). See `docs/pillar7_reasoning.md`.

## 4. Acceptance criteria (mapped)

| Criterion | Verification |
|-----------|--------------|
| Every pillar implemented per spec | `scripts/audit_repo.py` pillar coverage table |
| Every pillar integrates with the platform | Integration tests per milestone + `tests/test_all_subsystems.py` |
| Stable public APIs | `__init__.py` exports + `docs/api/` references |
| Everything configurable | `CADConfig` sub-configs; config-driven module activation |
| Automated tests per module | Test files mirroring `src/cadgenesis/` |
| Documentation per module | Module docstrings + `docs/` guide entries |
| Reproducible experiments | `research_economy/` registry + seed/config pinning |
| Versioned model artifacts | `tokenizer/versioning.py` + checkpoint hashing (`utils/hashing.py`) |
| Reproducible benchmarks | `benchmarks/` scripts with pinned seeds |
| Maintainable & modular | lint (ruff) + audit script + `docs/architecture.md` layout |

## 5. Verification loop

After every milestone:

1. `python -m pytest -q` — all existing + new tests green.
2. `python scripts/audit_repo.py` — stub count drops, coverage grows, API table
   is complete.
3. `python -m ruff check src tests` — clean.
4. `CHANGELOG.md` updated.
