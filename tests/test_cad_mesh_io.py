"""Test CAD mesh IO module."""
import sys
sys.path.insert(0, 'src')


def test_cad_mesh_io():
    from cadgensis.cad.mesh.io import MeshIO
    mesh = MeshIO()
    assert mesh is not None