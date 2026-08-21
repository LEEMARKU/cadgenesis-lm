"""cadgenesis.cad.mesh
====================
Mesh modelling: triangle meshes, STL / OBJ / PLY IO, mesh repair and mesh
simplification.
"""

from cadgenesis.cad.mesh.io import (
    read_obj,
    read_ply,
    read_stl,
    write_obj,
    write_ply,
    write_stl,
)
from cadgenesis.cad.mesh.mesh import Mesh
from cadgenesis.cad.mesh.repair import (
    diagnose,
    fill_holes,
    orient_faces,
    remove_degenerate_faces,
    remove_duplicate_vertices,
    remove_unreferenced_vertices,
)
from cadgenesis.cad.mesh.simplify import quadric_simplify, simplify_cluster

__all__ = [
    "Mesh",
    "diagnose",
    "fill_holes",
    "orient_faces",
    "quadric_simplify",
    "read_obj",
    "read_ply",
    "read_stl",
    "remove_degenerate_faces",
    "remove_duplicate_vertices",
    "remove_unreferenced_vertices",
    "simplify_cluster",
    "write_obj",
    "write_ply",
    "write_stl",
]
