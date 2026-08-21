"""cadgenesis.cad.parametric
==========================
Parametric CAD intelligence: parameters, sketch modelling and geometric /
dimensional constraints.
"""

from cadgenesis.cad.parametric.constraints import (
    ConstraintSolution,
    GeometricConstraint,
    SketchConstraintSolver,
    is_tangent,
)
from cadgenesis.cad.parametric.parameters import Parameter, ParameterTable
from cadgenesis.cad.parametric.sketch import (
    ArcEntity,
    CircleEntity,
    Dimension,
    LineEntity,
    PointEntity,
    Sketch,
    SketchEntity,
    SketchProfile,
    SplineEntity,
)

__all__ = [
    "ArcEntity",
    "CircleEntity",
    "ConstraintSolution",
    "Dimension",
    "GeometricConstraint",
    "LineEntity",
    "Parameter",
    "ParameterTable",
    "PointEntity",
    "Sketch",
    "SketchConstraintSolver",
    "SketchEntity",
    "SketchProfile",
    "SplineEntity",
    "is_tangent",
]
