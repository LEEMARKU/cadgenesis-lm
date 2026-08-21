# Pillar 4 — World Model

Implementation report for the **World Model** pillar of the CADGenesis-LM v6.0
roadmap (`docs/v6_roadmap.md`, milestone **M19**). Delivers the central
reasoning engine over engineering objects: spatial / mechanical / functional /
assembly / affordance reasoning, a forward-kinematics simulator, hierarchical
planning, and integration with memory and multimodal embeddings.

See `docs/pillar4_world_model_audit.md` for the pre-implementation requirements
audit; this document records the delivered design and API.

## 1. Scope (requirements → modules)

| # | Capability | Module |
|---|-----------|--------|
| 1 | Object representation (features, geometry, material, state, relations) | `world_model/objects.py` |
| 2 | Spatial reasoning (bounds, clearance, fit, overlap, distance, relative pose, symmetry) | `world_model/spatial.py` |
| 3 | Mechanical reasoning (stress, FoS, load, stability, mass budget) | `world_model/mechanical.py` |
| 4 | Functional reasoning (DOF requirements, box fit, load path, flow continuity) | `world_model/functional.py` |
| 5 | Assembly reasoning (mate/joint vocab, references, connectivity, cycles, mobility) | `world_model/assembly.py` |
| 6 | Affordances (insert, access, mate, hold, supports-action) | `world_model/affordances.py` |
| 7 | Design intent (intent parsing, requirement graph, spec tracing, envelope) | `world_model/design_intent.py` |
| 8 | World simulator (joint-state FK, trajectories, path collision) | `world_model/simulator.py` |
| 9 | Planning pipeline (plan creation + execution against the object graph) | `world_model/planning.py` |
| 10 | Facade + state (snapshot/restore) | `world_model/world_model.py` |
| 11 | Integration (CAD projection, multimodal inputs, memory, conditioned reasoning) | `world_model/integration.py` |

## 2. Module layout

```
world_model/
├── __init__.py          # public exports
├── objects.py           # WorldObject, Material, BoundaryCondition, LoadCase,
│                        #   make_object, ObjectGraph (relate/neighbors/root_for/...)
├── spatial.py           # SpatialReasoner + SpatialReport
├── mechanical.py        # MechanicalReasoner + MechanicalResult
├── functional.py        # FunctionalReasoner + FunctionalCheck
├── assembly.py          # WorldAssembly, AssemblyValidator, AssemblyCheck
├── affordances.py       # AffordanceMapper, Affordance, AFFORDANCE_ACTIONS
├── design_intent.py     # DesignIntent, IntentAnnotation, DesignIntentCapture
├── simulator.py         # MotionSimulator, JointState, SimulatedPose
├── planning.py          # WorldModelPlanner, WorldStep, StepOutcome, ExecutionResult
├── world_model.py       # WorldModelSystem facade, WorldModelState
└── integration.py       # WorldModelIntegration
```

## 3. Data model

```
WorldObject
  ├── id, name, kind, params (full dimensions)
  ├── material -> Material(density_kg_m3, ...)
  ├── pose    -> Transform (4x4)
  ├── relations: dict[str, list[str]]
  └── to_dict()/from_dict() round-trip

ObjectGraph
  ├── objects: dict[str, WorldObject]
  ├── relate(a, b, kind) / relations_of(name) / neighbors(name)
  ├── set_pose(name, transform) / root_for(name)
  └── bounds(name), find(by_kind), serialize()/load()
```

## 4. Facade API — `WorldModelSystem.reason(kind, **kwargs)`

| key | Backend | Args → Result |
|-----|---------|---------------|
| `clearance` | SpatialReasoner | `a, b, minimum, axis` → `SpatialReport` |
| `overlap` | SpatialReasoner | `a, b` → overlap volume + intersection |
| `fits_inside` | SpatialReasoner | `a, b` → AABB containment |
| `distance` | SpatialReasoner | `a, b` → center distance |
| `safety` | MechanicalReasoner | `object, load_case` → `MechanicalResult` (stress, FoS) |
| `stability` | MechanicalReasoner | `object` → tipping analysis |
| `mass` | MechanicalReasoner | `limit_kg` → mass budget |
| `dof` | FunctionalReasoner | `required` → available DOF check |
| `affordances` | AffordanceMapper | `object` → `Affordance[]` |
| `supports_action` | AffordanceMapper | `object, action` → bool + reason |
| `assembly` | AssemblyValidator | `WorldAssembly` → `AssemblyCheck` |
| `simulate` | MotionSimulator | `mech, joint_states` → simulated pose/trajectory |
| `check_path` | MotionSimulator | `mech, path` → collision check |
| `plan` | WorldModelPlanner | goal string → `WorldStep[]` |
| `execute_plan` | WorldModelPlanner | steps, graph → `ExecutionResult` |

## 5. Sequence — world → CAD document

```
WorldModelSystem.add_object(...)   # build WorldObject + ObjectGraph
  -> reason("safety", ...)         # constraint/feasibility checks
  -> planner.plan("assemble ...") -> execute(graph)
  -> WorldModelIntegration.to_cad_document(graph)
        feature family map (block→extrude, cylinder/cone→revolve, ...)
        materials via Material.density lookup
  -> to_multimodal_sample(graph, system)   # cad + sensor + text modalities
  -> store(graph, memory)                  # 'engineering' pool recall
```

## 6. Quality gates

- `tests/world_model/test_objects.py`, `test_reasoners.py`,
  `test_world_model.py` + `tests/evaluation/test_world_model_metrics.py` —
  44 tests green (38 world_model + 6 metrics).
- `evaluation/world_model_metrics.py` — accuracy, safety_margin,
  assembly_integrity, affordance_coverage_with, path_collision_detection,
  planning_success, run_world_benchmark.
- `benchmarks/world_model_benchmarks.py` — reasoner 0.005–0.11 ms/call,
  simulate 0.09 ms, plan+execute 0.05 ms, embed_world ~12.6 ms.
- `src/cadgenesis/config/__init__.py` — `WorldModelConfig` exported.
- Ruff clean; mypy clean.
