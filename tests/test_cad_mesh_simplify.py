"""Test CAD mesh simplify module."""
import sys
sys.path.insert(0, 'src')


def test_cad_mesh_simplify():
    from cadgensis.cad.mesh.simplify import MeshSimplify
    simplify = MeshSimplify()
    assert simplify is not None