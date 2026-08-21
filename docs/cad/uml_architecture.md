# CAD Intelligence — Architecture & Data Flow

High-level structure of the `cadgenesis.cad` package and how it plugs into the
existing CADGenesis-LM subsystems.

## Module dependency graph

```
                       ┌──────────────────────────────┐
                       │      CADIntelligencePipeline  │  (integration/pipeline.py)
                       └───────┬──────────┬───────────┘
                               │          │
               ┌───────────────▼──┐   ┌────▼────────────────┐
               │  CadValidator     │   │  TokenizerBridge     │
               │ (validation/)     │   │  TransformerBridge   │
               └───────┬──────────┘   │  MemoryBridge         │
                       │              │  ReasoningBridge      │
                       │              └──────────┬───────────┘
                       │                         │
         ┌─────────────▼─────────────┐   ┌───────▼──────────┐
         │  checks.py  → reasoning/  │   │  existing        │
         │  (topology, DFM, gdt...)  │   │  tokenizer/mem-  │
         └─────────────┬─────────────┘   │  ory/reasoning   │
                       │                 └──────────────────┘
   ┌───────┬───────┬────▼──┬────────┬────────┬───────┬────────┐
   │geometry│parametric│features│modeling│  mesh │assembly│ ...    │
   └───┬────┴────┬───┴────┬───┴───┬────┴──┬─────┴───┬────┴─────┘
       │         │        │       │       │         │
       │ material │ gdt  │manufacturing│mechanisms  │benchmarks
```

All leaf subsystems are pure Python and depend only on `geometry/core` types
(`Vec`, `Transform`, `Plane`).

## Subsystem data flow

| Stage | Input | Output | Module |
| --- | --- | --- | --- |
| Geometry | numbers | `Vec`, `Transform`, `Plane` | `geometry/core.py` |
| Parametric | entities + constraints | solved `ConstraintSolution` | `parametric/` |
| Features | sketch profile + params | validated `Feature` tree | `features/` |
| Modeling | primitives / ops | `BRepSolid`, `CSGTree` | `modeling/` |
| Mesh | primitives / files | `Mesh` + repair/simplify | `mesh/` |
| Assembly | components + mates | world transforms, DOF | `assembly/` |
| Materials/GD&T | enums & specs | validated specs | `materials/`, `gdt.py` |
| Manufacturing | material + batch | ranked `ProcessSelection` | `manufacturing/` |
| Mechanisms | links/joints | kinematics | `mechanisms/` |
| Validation | any design object | `CadValidationReport` | `validation/` |
| Integration | design dict | tokens + memory entry | `integration/` |

## Key relationships

```
Component ──< parent/children ── o Assembly
AssemblyConstraint ──> (reference_a, reference_b) : Reference
FeatureTree ──> Feature (ordered, dependency-aware, topologically sorted)
BRepSolid ──> Face ──> Edge ──> Vertex       (manifold validation)
CSGTree ──> CSGNode (binary op | primitive leaf)
Sketch ──> SketchEntity (Point/Line/Circle/Arc/Spline) + GeometricConstraint
MaterialDatabase ──[*]─>  Material (name, category, properties, aiases)
ProcessSelector ──> ProcessSelection (.best, .by_group)
Mechanism ──> Joint ; SpurGear ──> GearPair ──> GearTrain
```

## Integration with CADGenesis-LM

- **Tokenizer**: `TokenizerBridge.design_to_tokens()` emits `TICKETS`-style
  tokens (`PRIM_*`, `NUM_*`, `FEAT_*`, `MAT_*`) that the
  `AutonomousCADTokenizer` accepts.
- **Memory**: `MemoryBridge` (`store_design`/`recall`) persists designs into
  `CADMemory` automatically when `pipeline.run(..., remember=True)`.
- **Reasoning**: `ReasoningBridge` maps primitives to geometry-reasoner inputs
  and DFM part dicts to manufacturing-rules assessments.
- **Transformer**: `TransformerBridge` produces `MultiModalBatch` / tensor
  dicts via the tokenizer's `collate`.