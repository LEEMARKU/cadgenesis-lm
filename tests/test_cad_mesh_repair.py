"""Test CAD mesh repair module."""
import sys
sys.path.insert(0, 'src')


def test_cad_mesh_repair():
    from cadgensis.cad.mesh.repair import MeshRepair
    repair = MeshRepair()
    assert repair is not None