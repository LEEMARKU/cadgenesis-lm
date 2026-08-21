# Pillar 5 — Multi-Agent Intelligence: Repository Audit

Audit performed before implementation (M19/M20, v6.0 roadmap). Goal: make the
MAS the **primary orchestration layer** of CADGenesis-LM without breaking the
existing 90-agent-test surface or the torch `MultiAgentSystem` used by the
transformer stack.

## 1. Existing agents

Core protocol (`agents/base.py`): `AgentRequest(role, action, payload, task_id)`,
`AgentResult(role, action, ok, output, message, task_id)`, `Agent(ABC)` with
`role`, `actions`, `can_handle`, `handle`, `process`, `describe`. No lifecycle,
state, version, or config.

Eight role agents exist as thin wrappers over existing engines:

| Agent | role | actions | Backend |
|---|---|---|---|
| `PlannerAgent` | planner | create_plan, refine_plan | `reasoning.TaskPlanner` |
| `GeometryAgent` | geometry | validate, volume, aabb, overlap, fit | `GeometryReasoner` |
| `ConstraintAgent` | constraint | solve, check | `ConstraintSolver` |
| `AssemblyAgent` | assembly | check_clearance, check_mate | `GeometryReasoner` |
| `ManufacturingAgent` | manufacturing | assess, check_process | `ManufacturingRules` |
| `OptimizationAgent` | optimization | optimize, suggest | heuristic scoring |
| `SimulationAgent` | simulation | check_safety, check_load_case | safety-factor heuristics |
| `ValidationAgent` | validation | validate, report | `DesignValidator` |

Torch layer (`agents/multi_agent_system.py`): `InternalAgentRole(nn.Module)` and
`MultiAgentSystem(d_model=1024)` — an in-model 8-agent "bus" (concat + proj),
consumed by `transformer/geometry_transformer.py` (line 73) and `DecoderStack`.
**Not** the orchestration layer.

## 2. Implemented workflows

- `AgentCoordinator`: register/unregister, `dispatch(role, action, payload)`,
  `run_batch`, `run_all` (sequential), `ask_consensus`, `share`, `publish`,
  summary. `run_batch` calls `scheduler.next_tasks()` and dispatches serially.

## 3. Communication mechanisms

- `MessageBus` (`agents/message_bus.py`): synchronous in-process pub/sub;
  `subscribe/unsubscribe/publish/history/stats`; bounded per-topic history
  (default 512); faulty subscribers isolated. `AgentMessage.topic/payload/
  sender/message_id/timestamp/priority` — **priority is inert**.
- Missing: async transport, request/response correlation, broadcast helper,
  priority ordering, persistent event store, wildcard topics.

## 4. Scheduling

- `TaskScheduler` (`agents/scheduler.py`): `submit/submit_many`, status
  lifecycle (pending→ready→running→completed/failed), `next_tasks` (deps met +
  priority desc), `step`, `progress`, `has_cycles` (Kahn). 
- Missing: DAG object + critical path, deadlines, timeouts, retries, parallel
  execution/worker pool, load balancing, resource awareness.

## 5. Consensus

- `ConsensusEngine` (`agents/consensus.py`): `majority`, `weighted_majority`,
  `mean`, `is_unanimous`, `confidence`, `summary`, `record/record_many`.
- Missing: quorum, fallback/arbitration, tie-break policy, veto, decision trace.

## 6. Shared memory

- `SharedMemory` (`agents/shared_memory.py`): thread-safe flat key/value
  blackboard only. No tiers, TTL, persistence, or change events.

## 7. Missing vs pillar spec

- Infrastructure: standalone registry, dynamic loader, plugin interface,
  lifecycle management, capability discovery, health monitoring, versioning.
- Communication: event bus, request/reply, broadcast, priority queue,
  shared event store, async/distributed.
- Scheduling: DAG/dynamic/priority/deadline/parallel/load-balance.
- Shared memory: working/session/project/global/agent tiers + knowledge cache.
- Consensus: arbitration, fallback, quorum, conflict resolution.
- Task planning pipeline (intent → task graph → decompose → assign → execute →
  monitor → validate → aggregate) — absent.
- 10 of 18 specialized agents missing: Material, Cost, Documentation,
  Safety & Compliance, Memory, Retrieval, User Interaction, Learning,
  Monitoring, Debugging.
- Integration + evaluation suite + docs absent.

## 8. Duplicated functionality

- Planning: `reasoning.TaskPlanner` wrapped by both `PlannerAgent` and
  `world_model.WorldModelPlanner`.
- DAG utilities: `TaskScheduler` deps/cycles vs `CADPlan.topological_order/
  is_cyclic`.
- Shared memory: `agents.SharedMemory` vs `memory.MemoryStore`.
- Agent "bus": torch tensor routing vs `MessageBus` pub/sub (unrelated).
- Health: `monitoring/health.py` exists but is not wired to agents.

## 9. Architecture plan (backward compatible)

All new functionality is **additive**: new modules + new classes; existing
classes only gain optional defaulted parameters. Existing 90 tests must stay
green. Design:

1. `infrastructure.py` — `AgentBase` (extends `Agent` with lifecycle,
   metadata, version, capabilities, health), `Capability`, `AgentState`,
   `AgentLifecycleManager`.
2. `registry.py` — `AgentRegistry` (register/unregister/discover/capabilities).
3. `loader.py` + `plugins.py` — dynamic importlib/plugin-dir loader, plugin
   interface with enable/disable + version compatibility.
4. `versioning.py` — semantic agent versioning.
5. `health.py` — heartbeat + health aggregation (wired to `monitoring/health`).
6. `event_bus.py` — `EventBus` with priority queue, broadcast, request/reply
   correlation, and a shared event store; composes `MessageBus`.
7. `scheduling.py` — `TaskGraph`, `DAGScheduler`, `PriorityScheduler`,
   `DeadlineScheduler`, `DynamicScheduler`, `LoadBalancer`, `WorkerPool`.
8. `shared_memory.py` — add `LayeredSharedMemory` (working/session/project/
   global/agent + knowledge cache, TTL, change events); keep `SharedMemory`.
9. `consensus.py` — additive: quorum, tie-break, veto, arbitration, fallback,
   decision trace (existing keys untouched).
10. `pipeline.py` — `TaskPlanningPipeline` implementing the 8-stage spec
    pipeline.
11. `orchestrator.py` — `AgentOrchestrator`/`AgentPlatform` facade composing
    registry + scheduler + event bus + consensus + pipeline; keeps
    `AgentCoordinator` working.
12. `specialized/` — 10 new role agents; `fleet.py` registers all 18.
13. `integration.py` — wiring to transformer/tokenizer/world model/memory/
    neuro-symbolic/execution/continual learning/confidence.
14. `config` — `AgentsConfig`.
15. `evaluation/agent_metrics.py` + benchmarks + docs + tests.
