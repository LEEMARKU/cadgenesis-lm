"""cadgenesis.cad.modeling
========================
Solid modelling: boundary representation (B-Rep) with topology graphs,
constructive solid geometry (CSG) trees, analytic primitives and manifold
validation.
"""

from cadgenesis.cad.modeling.brep import (
    BRepSolid,
    Edge,
    EdgeGraph,
    Face,
    FaceGraph,
    GraphBase,
    Shell,
    TopologyGraph,
    Vertex,
    VertexGraph,
)
from cadgenesis.cad.modeling.csg import CSGNode, CSGOperation, CSGTree
from cadgenesis.cad.modeling.primitives import (
    SolidPrimitive,
    make_box,
    make_cone,
    make_cylinder,
    make_sphere,
    make_torus,
)

__all__ = [
    "BRepSolid",
    "CSGNode",
    "CSGOperation",
    "CSGTree",
    "Edge",
    "EdgeGraph",
    "Face",
    "FaceGraph",
    "GraphBase",
    "Shell",
    "SolidPrimitive",
    "TopologyGraph",
    "Vertex",
    "VertexGraph",
    "make_box",
    "make_cone",
    "make_cylinder",
    "make_sphere",
    "make_torus",
]
