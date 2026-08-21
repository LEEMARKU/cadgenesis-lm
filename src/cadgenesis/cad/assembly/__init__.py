"""cadgenesis.cad.assembly
========================
Assembly modelling: hierarchical assemblies, mates, constraints and
references with degree-of-freedom analysis.
"""

from cadgenesis.cad.assembly.assembly import Assembly, Component
from cadgenesis.cad.assembly.mates import (
    MATE_TYPES,
    AssemblyConstraint,
    MateAnalysis,
    MateSolver,
    Reference,
)

__all__ = [
    "MATE_TYPES",
    "Assembly",
    "AssemblyConstraint",
    "Component",
    "MateAnalysis",
    "MateSolver",
    "Reference",
]
