# Pillar 8 — CAD Execution & Validation: Repository Audit

Audit performed before implementation (v6.0 roadmap, Pillar 8).

## 1. Current state of `src/cadgenesis/execution/`

12 files, **81 lines** total — the package is ~1% implemented.

| File | Lines | Status |
|---|---|---|
| `execution_engine.py` | 64 | Partial heuristic — `CADExecutionEngine.execute_and_evaluate(tokens)` matches token prefixes for cost (`BOX`→$25, `CYLINDER`→$35, else $50); never validates geometry/topology/manufacturing; never raises |
| `__init__.py` | 8 | Re-exports `CADExecutionEngine`, `CADExecutionResult` |
| `geometry_validation.py` | 7 | Stub |
| `topology_analysis.py` | 7 | Stub |
| `manufacturing.py` | 7 | Stub |
| `simulation.py` | 7 | Stub |
| `optimization.py` | 7 | Stub |
| `cost_estimation.py` | 7 | Stub |
| `exporter.py` | 7 | Stub |
| `feedback.py` | 7 | Stub |
| `freecad_engine.py` | 7 | Stub |
| `opencascade_engine.py` | 7 | Stub |

`execution_engine.py` fails ruff (11 errors); mypy passes only vacuously.

## 2. Real capability elsewhere (scattered, unintegrated)

- **CAD features**: `cad/features/` — full `FeatureType` vocabulary
  (EXTRUDE/REVOLVE/SWEEP/LOFT/FILLET/CHAMFER/SHELL/DRAFT/HOLE/RIB/MIRROR/
  PATTERN/BOOLEAN_*), `FeatureTree.execution_order()` — but kernels are stubs.
- **B-Rep / mesh**: `cad/modeling/brep.py` (`TopologyGraph.is_manifold/is_closed/
  genus/validate`, `BRepSolid.from_prism`), `cad/modeling/` (CSG), `cad/mesh/mesh.py`
  (`is_watertight`, `boundary_edges`), `cad/mesh/io.py` (**STL binary+ascii, OBJ,
  PLY**), `cad/mesh/repair.py` (`remove_duplicate_vertices`, `fill_holes`, `diagnose`).
- **Validation**: `cad/validation/pipeline.py` (`CadValidator`, mirrors reasoning
  `CheckResult`), `cad/validation/checks.py` (mesh/topology/manufacturability/
  constraints checks), `reasoning/validator.py` (`DesignValidator`).
- **Manufacturing**: `cad/manufacturing/process.py` (`ProcessSelector`),
  `cad/manufacturing/features.py`, `reasoning/manufacturing_rules.py`.
- **Constraints**: `cad/parametric/constraints.py` (`SketchConstraintSolver`,
  DOF analysis), `cad/assembly/mates.py` (`MateSolver`), `reasoning/constraint_solver.py`.
- **Assembly**: `world_model/assembly.py` (`AssemblyValidator`, Gruebler
  mobility), `world_model/spatial.py` (AABB overlap/clearance).
- **Simulation**: none real. `cad/integration/simulation_bridge.py` persists to
  `SimulationMemory`; `world_model/simulator.py` is forward-kinematics only;
  `world_model/mechanical.py` is first-order proxy.
- **Cost/optimization**: none (`optimization/` package is inference-time
  quantization, also stubs).

## 3. Missing capabilities (vs. mission scope)

| Capability | Gap |
|---|---|
| CAD program executor (sketch→boolean ops) | Nothing executes features; backends are stubs |
| Geometry validator (self-intersection, open edges, invalid faces) | Partial elsewhere, nothing in `execution/`; self-intersection nowhere |
| Constraint validator (dimensional, dependency, propagation, repair) | Only sketch-level `SketchConstraintSolver` |
| Assembly validator (interference/collision/missing refs/mates/hierarchy) | Partial in `world_model` |
| Manufacturing validator (CNC/additive/casting/injection/sheet metal/welding/tooling) | Partial in `reasoning` + `cad/manufacturing` |
| Optimization engine (weight/material/complexity/print time/cost/structure) | Nothing |
| Simulation interfaces (FEA/CFD/motion/thermal/tolerance) | Nothing (proxy only) |
| Automatic repair (topology/geometry/constraint/mfg/assembly) | Mesh-level only |
| Export engine (STEP/IGES/Parasolid/STL/OBJ/GLTF/DXF/DWG/Fusion/SolidWorks/FreeCAD/OpenSCAD) | STL/OBJ/PLY read/write only |
| Feedback loop (execution → model) | Static suggestions only |
| FreeCAD / OCC backends | Empty stubs |

## 4. Duplicated logic

1. Manifold/closed/watertight checks **3×**: `reasoning/topology.py`,
   `cad/modeling/brep.py`, `cad/mesh/mesh.py`.
2. AABB overlap **3×**: `reasoning/geometry_reasoner.py`,
   `world_model/spatial.py`, `world_model/simulator.py` (inline).
3. Validation orchestrators **2×**: `DesignValidator` vs `CadValidator`.
4. Manufacturing feasibility **2×**: `ManufacturingRules.assess` vs
   `ProcessSelector.select`.
5. Constraint solvers **2×**: `ConstraintSolver` vs `SketchConstraintSolver`.
6. Closed-profile check **2×**: `cad/parametric/sketch.py` vs
   `multimodal/encoders/sketch.py`.

## 5. Integration gaps

- `cad/integration/execution_bridge.py` wraps only the heuristic engine
  (`run_tokens`/`run_design`/`summary`).
- `distillation/distill_pipeline.py::QualityFilteringEngine` gates dataset
  samples on the heuristic `is_valid_geometry` — data quality trusts a
  token-prefix match.
- `agents/integration.py::ExecutionAdapter` lazy-imports the heuristic engine,
  not exported from `agents/__init__.py`.
- Transformer has no execution/validation hook; `cad/integration/pipeline.py`
  (validate→reason→tokenize→memorize) omits execution.
- World model: `WorldModelPlanner.execute()` validate/simulate ops are no-ops.
- Memory: execution results never persisted.
- Confidence: `confidence_score` hardcoded 0.95.
- Digital twin: package does not exist.

## 6. Tests / benchmarks

- `tests/execution/` **does not exist**; only indirect coverage:
  `tests/test_all_subsystems.py::test_execution_engine` (asserts
  `is_valid_geometry` for `["BOX", ...]`) and
  `tests/cad/test_integration.py::TestExecutionBridge` (3 tests).
- No execution benchmark; `evaluation/` metric modules for cad/geometry are stubs.

## 7. Architectural plan (backward compatible)

1. **Virtual analytic kernel**: pure-Python feature executor (sketch/extrude/
   revolve/sweep/loft/fillet/chamfer/shell/draft/hole/rib/mirror/pattern/
   boolean) building B-Rep + mesh from the existing `cad/modeling`/`cad/mesh`
   substrate; FreeCAD/OCC backends become real plugin interfaces with
   pure-Python fallback (optional-import).
2. **Validators** in `execution/`: geometry (incl. self-intersection), topology
   (adjacency + reuse), constraint (propagation/conflict/repair), assembly
   (interference/collision/mates/hierarchy), manufacturing (CNC/additive/
   casting/injection/sheet metal/welding/tooling).
3. **Optimization engine** (weight/material/complexity/print time/cost/
   structural efficiency) + **cost estimator**.
4. **Simulation interfaces** (FEA/CFD/motion/thermal/tolerance — analytic
   first-order solvers, pluggable).
5. **Automatic repair** (topology/geometry/constraint/manufacturability/
   assembly) layered over `cad/mesh/repair.py` + feature-level adjustments.
6. **Export engine**: STEP/IGES/Parasolid/STL/OBJ/GLTF/DXF + script/manifest
   exporters for DWG/Fusion 360/SolidWorks/FreeCAD/OpenSCAD (binary formats
   get documented structured fallbacks).
7. **Pipeline**: `CADExecutionEngine` orchestrates intent → program →
   execute → validate → simulate → optimize → repair → export → feedback,
   keeping the existing `CADExecutionResult` contract intact.
8. **Integrations**: extend `ExecutionBridge` (additive), wire execution into
   world model, memory persistence, agents, confidence scoring, digital twin
   (new minimal package), and the CAD pipeline (flag-gated).
9. **Evaluation**: `evaluation/execution_metrics.py` + `benchmarks/execution_benchmarks.py`.
10. **Tests**: new `tests/execution/` suite; existing tests stay green.

Nothing existing is removed; the old `execute_and_evaluate` token behavior is
preserved as a compatibility path.
