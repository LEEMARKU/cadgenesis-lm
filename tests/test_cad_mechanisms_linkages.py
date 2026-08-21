"""Test CAD mechanisms linkages module."""
import sys
sys.path.insert(0, 'src')


def test_cad_mechanisms_linkages():
    from cadgensis.cad.mechanisms.linkages import Linkages
    linkages = Linkages()
    assert linkages is not None