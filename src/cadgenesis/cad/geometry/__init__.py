"""cadgenesis.cad.geometry
=========================
Geometry foundation of the CAD Intelligence core: 3D math primitives
(:mod:`cadgenesis.cad.geometry.core`) and curve/surface mathematics
(:mod:`cadgenesis.cad.geometry.curves`).
"""

from cadgenesis.cad.geometry.core import (
    Axis,
    Frame,
    Plane,
    Transform,
    Vec,
    angle_between,
    is_parallel,
    is_perpendicular,
)
from cadgenesis.cad.geometry.curves import (
    BezierSurface,
    NurbsCurve,
    NurbsSurface,
    bezier_point,
    lofted_surface_points,
    ruled_surface_points,
)
from cadgenesis.cad.geometry.surfaces import (
    SurfacePatch,
    point_in_polygon,
    stitch_surfaces,
    trim_surface,
)

__all__ = [
    "Axis",
    "BezierSurface",
    "Frame",
    "NurbsCurve",
    "NurbsSurface",
    "Plane",
    "SurfacePatch",
    "Transform",
    "Vec",
    "angle_between",
    "bezier_point",
    "is_parallel",
    "is_perpendicular",
    "lofted_surface_points",
    "point_in_polygon",
    "ruled_surface_points",
    "stitch_surfaces",
    "trim_surface",
]
