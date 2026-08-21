"""Test CAD solids module."""
import sys
sys.path.insert(0, 'src')

from cadgensis.cad.features.solids import __all__ as solids_list


def test_solids_import():
    assert solids_list is not None