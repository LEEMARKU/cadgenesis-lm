"""Test CAD parametric module."""
import sys
sys.path.insert(0, 'src')

from cadgensis.cad.parametric import __all__ as param_list


def test_parametric_import():
    assert param_list is not None