"""Test CAD geometry core module."""
import sys
sys.path.insert(0, 'src')


def test_cad_geometry_core():
    from cadgensis.cad.geometry.core import CoreGeometry
    geom = CoreGeometry()
    assert geom is not None