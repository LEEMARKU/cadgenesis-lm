"""Test CAD mechanisms parts module."""
import sys
sys.path.insert(0, 'src')


def test_cad_mechanisms_parts():
    from cadgensis.cad.mechanisms.parts import Parts
    parts = Parts()
    assert parts is not None