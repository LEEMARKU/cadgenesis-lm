# M5 — Multi-Agent Intelligence Completeness

Milestone M5 of the CADGenesis-LM v6.0 Ultimate Architecture roadmap
(`docs/v6_roadmap.md`) completes the Multi-Agent Intelligence pillar
(pillar 5): `cadgenesis.agents` now ships a full orchestration layer alongside
the preserved torch-based internal agent bus.

## Scope

The existing `InternalAgentRole` / `MultiAgentSystem` (differentiable, 8-role
agent bus embedded in the transformer and conditioned on the layer-integrated
memory bank) is preserved unchanged.  M5 adds the **external orchestration
layer**: a role-typed request/result protocol, a pub/sub message bus, a shared
blackboard, a dependency-aware task scheduler, a consensus engine, a
coordinator facade, and eight specialised role agents that wrap the M3
reasoning APIs (planner, geometry, constraint, manufacturing, optimization,
assembly, simulation, validation).

## Modules delivered

| Module | Contents |
| --- | --- |
| `base.py` | `AgentRequest` / `AgentResult` protocol records and the `Agent` ABC (`role`, `actions`, `can_handle`, `handle`, `process`, `describe`); empty-role guard |
| `message_bus.py` | `AgentMessage` + `MessageBus` — topic pub/sub, bounded per-topic history, subscriber fault isolation, stats |
| `shared_memory.py` | `SharedMemory` — thread-safe (RLock) blackboard `get`/`set`/`update`/`remove`/`items` |
| `scheduler.py` | `AgentTask` + `TaskScheduler` — priority + dependency ordering, cycle detection, `next_tasks`/`step`/`progress`/`mark_*` |
| `consensus.py` | `AgentOpinion` + `ConsensusEngine` — majority / weighted-majority / mean / unanimity / confidence |
| `coordinator.py` | `AgentCoordinator` — agent registry, dispatch, `publish`/`share`, `ask_consensus`, `submit`/`run_batch`/`run_all`, `summary` |
| `planner/` | `PlannerAgent` — `create_plan` / `refine_plan` over `TaskPlanner` |
| `geometry/` | `GeometryAgent` — `validate` / `volume` / `aabb` / `overlap` / `fit` over `GeometryReasoner` |
| `constraint/` | `ConstraintAgent` — `solve` / `check` over `ConstraintSolver` |
| `manufacturing/` | `ManufacturingAgent` — `assess` / `check_process` over `ManufacturingRules` |
| `optimization/` | `OptimizationAgent` — `optimize` / `suggest` with objective scoring and recommendations |
| `assembly/` | `AssemblyAgent` — `check_clearance` / `check_mate` interference checks |
| `simulation/` | `SimulationAgent` — `check_safety` / `check_load_case` |
| `validation/` | `ValidationAgent` — `validate` / `report` over `DesignValidator` |
| `__init__.py` | Package facade exporting the full public API (21 names incl. legacy `InternalAgentRole` / `MultiAgentSystem`) |

## Design notes

- **Two complementary layers.** `multi_agent_system.py` remains the
  differentiable in-model bus; M5's orchestration layer is pure Python so it
  unit-tests instantly and runs standalone, routing requests to agents that
  wrap the same reasoning APIs used by the model.
- **Protocol-first.** Every interaction flows through `AgentRequest` /
  `AgentResult`; agents declare a `role` and `actions`, and `handle` performs
  role/action validation before delegating to `process`.
- **Fault isolation.** A faulty message-bus subscriber is swallowed without
  breaking delivery to other subscribers.
- **Scheduling & consensus.** `TaskScheduler` runs ready batches in dependency
  order with cycle detection; `ConsensusEngine` aggregates opinions across
  agents (majority, weighted, mean, unanimity) — used by
  `AgentCoordinator.ask_consensus`.
- **Configurable & testable.** Role agents accept injected reasoners/solvers;
  `AgentCoordinator` composes bus, blackboard, scheduler and consensus engines
  and is fully exercised by 79 new tests.

## Verification

```text
pytest           821 passed (79 new agent tests)
ruff check       clean for cadgenesis.agents (new modules) and tests/agents
audit_repo.py    182 modules · 309 public APIs · 16 897 LOC · 66 stubs
                 Multi-Agent Intelligence pillar: OK
```

Legacy hits in `agents/multi_agent_system.py` (I001/UP035/UP006/UP045/F401)
are pre-existing and deferred to M18.
