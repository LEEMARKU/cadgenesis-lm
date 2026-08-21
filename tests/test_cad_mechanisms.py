"""Test CAD mechanisms module."""
import sys
sys.path.insert(0, 'src')

from cadgensis.cad.mechanisms import __all__ as mechanism_list


def test_mechanisms_import():
    assert mechanism_list is not None