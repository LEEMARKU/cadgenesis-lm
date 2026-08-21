"""Test CAD mechanisms gears module."""
import sys
sys.path.insert(0, 'src')


def test_cad_mechanisms_gears():
    from cadgensis.cad.mechanisms.gears import Gears
    gears = Gears()
    assert gears is not None