"""Test CAD parametric parameters module."""
import sys
sys.path.insert(0, 'src')


def test_cad_parametric_parameters():
    from cadgensis.cad.parametric.parameters import Parameters
    params = Parameters()
    assert params is not None