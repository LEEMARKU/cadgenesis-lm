# Pillar 5 — Multi-Agent Intelligence

Implementation report for the **Multi-Agent Intelligence** pillar of the
CADGenesis-LM v6.0 roadmap (`docs/v6_roadmap.md`, milestone **M19**). Delivers
the production orchestration platform: agent lifecycle + registry + loader,
plugins, health, an event bus, DAG scheduling, layered shared memory, extended
consensus, a task-planning pipeline, integration adapters, an 18-agent fleet
and the `AgentPlatform` facade.

See `docs/pillar5_multi_agent_audit.md` for the pre-implementation requirements
audit; this document records the delivered design and API.

## 1. Scope (requirements → modules)

| # | Capability | Module |
|---|-----------|--------|
| 1 | Agent protocol + lifecycle (states, metadata, versioning) | `agents/infrastructure.py`, `agents/versioning.py` |
| 2 | Registry, discovery, loader, plugins | `agents/registry.py`, `agents/loader.py`, `agents/plugins.py` |
| 3 | Health monitoring | `agents/health.py` |
| 4 | Event bus (priority ordering, wildcard, broadcast, request/response) | `agents/event_bus.py` |
| 5 | Task scheduling (DAG, priority, deadline, dynamic, load balancing) | `agents/scheduling.py` |
| 6 | Layered shared memory + semantic mirror | `agents/shared_memory.py` |
| 7 | Extended consensus (quorum, tie-break, veto, trace) | `agents/consensus.py` |
| 8 | Task planning pipeline (intent → DAG → dispatch → aggregate) | `agents/pipeline.py` |
| 9 | 18-agent fleet (8 legacy + 10 specialized) | `agents/fleet.py`, `agents/{material,cost,documentation,safety,memory,retrieval,user,learning,monitoring,debugging}/` |
| 10 | Orchestrator facade | `agents/orchestrator.py` |
| 11 | Platform integration adapters (transformer, tokenizer, world model, memory, reasoning, continual learning) | `agents/integration.py` |
| 12 | Configuration + evaluation + benchmarks | `config/cad_config.py`, `evaluation/agent_metrics.py`, `benchmarks/agent_benchmarks.py` |

## 2. Architecture (UML)

```
AgentPlatform  (facade)
 ├── AgentRegistry  ──► 18× AgentBase (role agents)
 │     ├── AgentLifecycleManager    AgentVersion / AgentMetadata
 │     ├── AgentLoader  ──► AgentPlugin / PluginManifest
 │     └── AgentHealthMonitor  ──► AgentHealthStatus
 ├── EventBus  ──► Event / SharedEventStore (priority heap + sequence)
 ├── TaskPlanningPipeline  (IntentAnalyser → TaskGraphBuilder →
 │     TaskDecomposer → AgentAssigner → TaskValidator → ResultAggregator)
 │     └── DAGScheduler / PriorityScheduler / DeadlineScheduler /
 │         DynamicScheduler / LoadBalancer / WorkerPool  over TaskGraph
 ├── LayeredSharedMemory  (regions + mirror into MemorySystem)
 ├── ConsensusEngine  (opinions, quorum, veto, trace, summary)
 └── PlatformIntegrations  (TransformerAgentAdapter, TokenizerAdapter,
     WorldModelAdapter, MemoryAdapter, ReasoningAdapter, ExecutionAdapter,
     ConfidenceAdapter, ContinualLearningHooks, NeuroSymbolicAdapter)
```

## 3. Module layout

```
agents/
├── infrastructure.py     # AgentBase ABC, AgentMetadata, AgentState,
│                         #   Capability, AgentLifecycleManager
├── versioning.py         # AgentVersion (semver compare)
├── registry.py           # AgentRegistry (register/discover/find/capabilities)
├── loader.py             # AgentLoader (class/module/dir discovery), AgentLoadError
├── plugins.py            # AgentPlugin, PluginManifest (pre/post hooks)
├── health.py             # AgentHealthMonitor, AgentHealthStatus
├── event_bus.py          # Event, EventBus, SharedEventStore
├── scheduling.py         # TaskGraph/Node, DAGScheduler, PriorityScheduler,
│                         #   DeadlineScheduler, DynamicScheduler, LoadBalancer,
│                         #   WorkerPool, SchedulerStats
├── consensus.py          # AgentOpinion, ConsensusEngine
├── shared_memory.py      # SharedMemory (legacy), LayeredSharedMemory
├── pipeline.py           # IntentAnalyser, TaskGraphBuilder, TaskDecomposer,
│                         #   AgentAssigner, TaskValidator, ResultAggregator,
│                         #   TaskPlanningPipeline, PipelineReport
├── orchestrator.py       # AgentPlatform
├── integration.py        # platform adapters + PlatformIntegrations facade
├── fleet.py              # FLEET_ROLES, build_fleet, create_fleet_registry
└── {planner,geometry,constraint,assembly,manufacturing,simulation,
    optimization,validation}/        # 8 legacy role agents (unchanged)
└── {material,cost,documentation,safety,memory,retrieval,user,learning,
    monitoring,debugging}/           # 10 new specialized agents
```

## 4. The 18-agent fleet

`FLEET_ROLES` = planner, geometry, constraint, assembly, manufacturing,
simulation, optimization, validation, material, cost, documentation, safety,
memory, retrieval, user, learning, monitoring, debugging.

## 5. Key APIs

| Component | API |
|-----------|-----|
| `EventBus` | `subscribe(topic, fn, filter_fn)`, `unsubscribe`, `publish`, `broadcast`, `respond`, `request(topic, payload, timeout, min_responses)`, `flush`, `history`, `stats`, `subscriber_count` |
| `TaskGraph` | `add`, `add_dependency`, `ready_nodes`, `topological_order`, `is_acyclic`, `critical_path`, `dependencies_of`, `dependents_of` |
| `DAGScheduler` | `run(graph, execute_fn)` → `SchedulerStats` |
| `ConsensusEngine` | `record`, `record_many`, `majority`, `weighted_majority`, `mean`, `unanimous`, `decision`, `has_quorum`, `vetoed`, `summary`, `full_summary`, `trace` |
| `LayeredSharedMemory` | `set/get/update/remove/peek`, regions, `on_change`, `cache_put/cache_get`, `attach_memory` (semantic mirror), `snapshot`, `usage` |
| `AgentRegistry` | `register`, `register_many`, `get`, `discover`, `find_by_action`, `find_by_capability`, `capabilities`, `roles`, `snapshot`, `summary` |
| `AgentLoader` | `load_class`, `load_module`, `scan_package`, `scan_directory` |
| `TaskPlanningPipeline` | `run(goal)` → `PipelineReport` (intent → 5-task DAG → dispatch → aggregated results) |
| `AgentPlatform` | `register`, `load_fleet`, `dispatch`, `ask`, `share`, `publish`, `submit_pipeline`, `health_summary`, `fleet_snapshot`, `stats`, `shutdown` |

## 6. Sequence — task planning pipeline

```
IntentAnalyser(goal) ──► intent = TaskGraphBuilder.build(goal)
  ──► TaskDecomposer.decompose(graph)          # recursive leaf expansion
  ──► AgentAssigner.assign(graph, registry)    # role/action per node
  ──► TaskValidator.validate(graph)            # acyclic + assigned check
  ──► DAGScheduler.run(graph, dispatch)        # topo-order execution
  ──► ResultAggregator.collect(results) ──► PipelineReport(gate, tasks)
```

Payload seeding: each role receives defaults (e.g. material/cost use
`"Al 6061-T6"`, simulation uses `safety_factor 2.0`,
manufacturing uses `{part:{kind,dims,position,name,processes:[machining]}}`).

## 7. Configuration

`AgentsConfig` (in `config/cad_config.py`, exported) exposes
`enabled`, `workers`, `task_timeout`, `default_retries`, `quorum`,
`event_history`, `shared_memory_capacity`, `decompose_tasks`,
`heartbeat_timeout` — wired into `CADConfig.agents`.

## 8. Quality gates

- 107 new tests (`tests/agents/test_pillar5_*.py` × 8 files +
  `tests/evaluation/test_agent_metrics.py`) — all green; full suite
  1315 passed, legacy `tests/agents/*` untouched and green.
- `evaluation/agent_metrics.py` — latency, success, accuracy, reliability,
  availability, throughput, p95 latency, memory footprint +
  `run_agent_benchmark()`.
- `benchmarks/agent_benchmarks.py` — startup 0.26 ms, dispatch 0.03 ms,
  pipeline 0.37 ms, consensus 0.07 ms, share 0.003 ms (--reps 2).
- `scripts/audit_repo.py` — Multi-Agent Intelligence pillar OK.
- Ruff clean; mypy clean (28 milestone source files).
