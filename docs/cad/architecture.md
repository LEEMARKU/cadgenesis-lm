# CAD Intelligence Package (`src/cadgenesis/cad/`)

This document describes the Pillar 2 "CAD Intelligence" package delivered on top
of the existing CADGenesis-LM stack. The package adds reusable, pure-Python CAD
kernels (geometry, parametric modelling, features, B-Rep, meshes, assemblies,
materials, GD&T, manufacturing, mechanisms) plus the validation, integration and
benchmark layers that connect them to the existing tokenizer / transformer /
reasoning / memory subsystems.

Everything is **additive**: no existing module was rewritten, and the execution
backends (`execution/freecad_engine.py`, `execution/opencascade_engine.py`)
remain as stubs outside the scope of this package.

## Package layout

```
src/cadgenesis/cad/
├── __init__.py            # public exports + GD&T module
├── geometry/              # Vec, Transform, Plane, curves (Bezier/NURBS)
├── parametric/            # parameters, sketches, constraint solver
├── features/              # feature base, solids, dress, patterns, boolean
├── modeling/              # primitives, B-Rep topology, CSG tree
├── mesh/                  # triangle mesh, IO (STL/OBJ/PLY), repair, simplify
├── assembly/              # components, assemblies, mates / DOF
├── materials/             # material database (metals, polymers, ceramics...)
├── manufacturing/         # manufacturing features + process selection
├── mechanisms/            # joints, gears, cams, linkages, machine parts
├── validation/            # CadValidator, checks, reports
├── integration/           # bridges to tokenizer / transformer / memory / reasoning
└── benchmarks/            # benchmark kernels for each CAD subsystem
```

## Module index

| Subsystem | Public entry points |
| --- | --- |
| Geometry | `Vec`, `Point`, `Plane`, `Axis`, `Frame`, `Transform`, `bezier_*`, `NurbsCurve`, `NurbsSurface` |
| Parametric | `Parameter`, `ParameterTable`, `Sketch`, `SketchProfile`, `SketchConstraintSolver`, `GeometricConstraint` |
| Features | `FeatureTree`, `ExtrudeFeature`, `RevolveFeature`, `FilletFeature`, `LinearPatternFeature`, `BooleanFeature` (registry-driven) |
| Modeling | `SolidPrimitive`, `make_box/cylinder/sphere/cone/torus`, `BRepSolid`, `CSGTree` |
| Mesh | `Mesh`, `read/write_stl/obj/ply`, `repair_mesh`, `quadric_simplify`, `simplify_cluster` |
| Assembly | `Component`, `Assembly`, `AssemblyConstraint`, `MateSolver` |
| Materials | `Material`, `MaterialDatabase` |
| Manufacturing | `ManufacturingFeature`, `ProcessSelector`, `ProcessSelection`, process-group builders |
| Mechanisms | `Joint`, `Mechanism`, `SpurGear`, `GearPair`, `GearTrain`, `CamProfile`, `FourBarLinkage`, `Bearing`, `Shaft` |
| GD&T | `GDTSpecification`, `Datum`, `FeatureControlFrame`, `ManufacturingTolerance` |
| Validation | `CadValidator`, `CadCheckResult`, `CadValidationReport`, per-domain checks |
| Integration | `CADIntelligencePipeline`, `PipelineResult`, bridge classes |

## Design principles

1. **Pure Python, no numpy/torch dependency** for the CAD kernels — they run in
   any environment that can install `cadgenesis`.
2. **Dict-friendly**: every major object has `to_dict()` / `from_dict()`, so
   designs flow easily into the existing tokenizer and memory bridges.
3. **Validation-first**: `Feature.validate()`, `BRepSolid.validate()`,
   `GDTSpecification.validate()` and the aggregate `CadValidator` all return
   problem lists (empty == valid), reusing the reasoning toolkit's topology
   analyzer and manufacturing rules where possible.
4. **Incremental integration**: the `integration/` layer is a thin adapter over
   the existing `AutonomousCADTokenizer`, `CADMemory`, reasoning solvers and
   transformer collate, preserving their public contracts.
