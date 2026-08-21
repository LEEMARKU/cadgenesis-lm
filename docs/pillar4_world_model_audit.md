# Pillar 4 — World Model Audit Report

Completeness audit for the **World Model** pillar of the CADGenesis-LM v6.0
Ultimate Architecture roadmap (`docs/v6_roadmap.md`).  Audit performed before
implementation; the goal is a world model that acts as the **central reasoning
engine** feeding CAD generation.

## 1. Requirements (from the roadmap)

The world model must provide:

| # | Capability | Required module |
|---|-----------|-----------------|
| 1 | Internal object representation (primitive features, geometry, material, state, confidence) | `world_model/objects.py` |
| 2 | Spatial reasoning (dimensions, relative poses, clearance, fit, bounds, symmetry) | `world_model/spatial.py` |
| 3 | Mechanical reasoning (forces, stress, stability, factor of safety, materials) | `world_model/mechanical.py` |
| 4 | Functional reasoning (roles, purpose, dependencies, requirements satisfaction) | `world_model/functional.py` |
| 5 | Assembly reasoning (mates, joint chains, motion, interference, degree of freedom) | `world_model/assembly.py` |
| 6 | Affordances (grasp, insertion, attachment, load-bearing, tool clearance) | `world_model/affordances.py` |
| 7 | Design intent (intent parsing, requirement graphs, design-spec tracing) | `world_model/design_intent.py` |
| 8 | World simulator (state evolution, step(), physics, DFM, consistency validation) | `world_model/simulator.py` |
| 9 | Hierarchical planning pipeline (Intent Parser → Requirement Graph → World Model → Planning Engine → Geometry/Constraint/Execution Planner → CAD Generator) | `world_model/planning.py` |
| 10 | Facade + integration (memory, multimodal, agents, reasoning, execution, tokenizer, transformer) | `world_model/world_model.py`, `world_model/integration.py` |

## 2. Existing assets to reuse (no reimplementation)

Audited against `src/cadgenesis/`:

| Module | Reusable API |
|--------|--------------|
| `cad/geometry/core.py` | `Vec`, `Transform` (identity/translation/rotation/composed/apply), `Plane` |
| `cad/assembly/mates.py` | `MATE_TYPES`, `_MATE_DOF` (mate → DOF reduction table) |
| `cad/mechanisms/joints.py` | `JOINT_TYPES`, `Joint` with `dof` |
| `cad/parametric/sketch.py` | `Sketch`, `SketchEntity` |
| `reasoning/validator.py` | `CheckResult`, `ValidationReport`, `DesignValidator` |
| `reasoning/planner.py` | `TaskPlanner`, `CADPlan`, `PlanningStep` |
| `reasoning/geometry_reasoner.py` | volume / aabb / overlap / fit / symmetry primitives |
| `reasoning/knowledge_graph.py` | `KnowledgeGraph`, `GraphNode`, `GraphEdge` |
| `reasoning/manufacturing_rules.py` | `ManufacturingRules`, `ManufacturingAssessment`, `MfgCheck` |
| `memory/memory_system.py` | `MemorySystem` facade (remember/recall/retrieve/route, 8 pools) |
| `multimodal/` (Pillar 3) | `MultimodalSystem`, `SharedEngineeringEmbeddingSpace`, encoders |

## 3. Gap analysis

- **No `world_model/` package exists** (pillar 4 was routed to `memory/`,
  `transformer/`, `reasoning/`).  All ten required modules are new.
- **Affordances / design intent / world simulator** have no direct existing
  counterpart anywhere in the repo.
- **Objects**: no shared `WorldObject` dataclass; CAD features live in
  `cad/` backends and `tokenizer` token definitions.  A normalized internal
  representation is required.
- **Integration**: `MemorySystem` and `MultimodalSystem` exist and are wired
  through `CADConfig`; the world model must attach to both plus the
  reasoning/execution layer.

## 4. Implementation plan (module order)

1. `objects.py` — `WorldObject`, `Material`, `Pose`, `BoundaryCondition`,
   `SimulationResult`, object graphs.
2. `spatial.py` — `SpatialReasoner` (bounds, clearance, fit, pose math,
   symmetry, tesselation-free occupancy).
3. `mechanical.py` — `MechanicalReasoner` (load case, stress proxy, stability,
   safety factor, material properties).
4. `functional.py` — `FunctionalReasoner` (roles, requirements satisfaction,
   functional dependencies).
5. `assembly.py` — `AssemblyReasoner` (mates, joint chains, DOF, motion,
   interference) over `cad/assembly` + `cad/mechanisms`.
6. `affordances.py` — `AffordanceModel` (grasp / insertion / attachment /
   load-bearing / clearance heuristics).
7. `design_intent.py` — `IntentParser`, `RequirementNode`,
   `RequirementGraph`, `DesignIntentModel`.
8. `simulator.py` — `WorldSimulator` (state, `step`, physics, DFM +
   consistency validation via `DesignValidator`).
9. `planning.py` — `IntentParser` → `RequirementGraph` → `WorldModel` →
   `PlanningEngine` → Geometry/Constraint/Execution planner → CAD generator
   signal.
10. `world_model.py` — `WorldModel` facade orchestrating 1–9.
11. `integration.py` — memory / multimodal / agents / reasoning / execution /
    tokenizer / transformer adapters.
12. `evaluation/world_model_metrics.py` — plan validity, intent-trace
    coverage, simulator consistency, affordance plausibility.

## 5. Verification gates (after implementation)

```text
pytest           tests/world_model + tests/multimodal pass
ruff check       clean for cadgenesis.world_model / cadgenesis.multimodal
mypy             no issues in the new packages
audit_repo.py    no new stubs; World Model pillar modules listed
```

## 6. Invariants

- No existing module is modified beyond additive wiring in `CADConfig`,
  `datasets/__init__`, `evaluation/__init__` and the roadmap/architecture docs.
- The world model never re-implements `Vec`/`Transform`/`DesignValidator`;
  it composes them.
- No placeholder code: every module ships real, tested logic.
