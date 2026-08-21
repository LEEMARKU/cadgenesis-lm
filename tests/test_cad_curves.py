"""Test CAD curves module."""
import sys
sys.path.insert(0, 'src')

from cadgensis.cad.geometry.curves import Curves


def test_curves_import():
    curves = Curves()
    assert curves is not None