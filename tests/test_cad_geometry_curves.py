"""Test CAD geometry curves module."""
import sys
sys.path.insert(0, 'src')


def test_cad_geometry_curves():
    from cadgensis.cad.geometry.curves import Curves
    curves = Curves()
    assert curves is not None