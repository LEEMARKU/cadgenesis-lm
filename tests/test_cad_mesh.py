"""Test CAD mesh module."""
import sys
sys.path.insert(0, 'src')

from cadgensis.cad.mesh import __all__ as mesh_list


def test_mesh_import():
    assert mesh_list is not None