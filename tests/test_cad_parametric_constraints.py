"""Test CAD parametric constraints module."""
import sys
sys.path.insert(0, 'src')


def test_cad_parametric_constraints():
    from cadgensis.cad.parametric.constraints import Constraints
    constraints = Constraints()
    assert constraints is not None