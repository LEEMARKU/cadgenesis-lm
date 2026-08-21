"""cadgenesis.cad
===============
Pillar 2 — "CAD Intelligence": native CAD understanding for CADGenesis-LM.

Subpackages and modules:

- :mod:`geometry`      — vectors, curves, NURBS, surfaces
- :mod:`parametric`    — parameters, sketches, geometric constraints
- :mod:`features`      — feature-based modelling operations
- :mod:`modeling`      — solid modelling: primitives, B-Rep, CSG
- :mod:`mesh`          — triangle meshes: I/O, repair, simplification
- :mod:`assembly`      — assemblies, hierarchy, mates, DOF analysis
- :mod:`materials`     — material database with SI properties
- :mod:`manufacturing` — manufacturing features and process selection
- :mod:`mechanisms`    — joints, gears, cams, linkages, machine parts
- :mod:`validation`    — validation pipeline over designs
- :mod:`integration`   — bridges to tokenizer/transformer/memory/reasoning
- :mod:`benchmarks`    — performance micro-benchmarks
- :mod:`gdt`           — GD&T specifications
"""

from cadgenesis.cad.features import FEATURE_REGISTRY, Feature, FeatureType
from cadgenesis.cad.gdt import (
    Datum,
    DatumReference,
    FeatureControlFrame,
    GDTSpecification,
    ManufacturingTolerance,
)
from cadgenesis.cad.materials import MATERIALS, MaterialDatabase
from cadgenesis.cad.modeling import (
    BRepSolid,
    CSGNode,
    CSGTree,
    SolidPrimitive,
    TopologyGraph,
    make_box,
    make_cylinder,
    make_sphere,
)
from cadgenesis.cad.parametric import (
    GeometricConstraint,
    Sketch,
    SketchConstraintSolver,
)
from cadgenesis.cad.validation import CadValidationReport, CadValidator

__all__ = [
    "FEATURE_REGISTRY",
    "MATERIALS",
    "BRepSolid",
    "CSGNode",
    "CSGTree",
    "CadValidationReport",
    "CadValidator",
    "Datum",
    "DatumReference",
    "Feature",
    "FeatureControlFrame",
    "FeatureType",
    "GDTSpecification",
    "GeometricConstraint",
    "ManufacturingTolerance",
    "MaterialDatabase",
    "Sketch",
    "SketchConstraintSolver",
    "SolidPrimitive",
    "TopologyGraph",
    "make_box",
    "make_cylinder",
    "make_sphere",
]
