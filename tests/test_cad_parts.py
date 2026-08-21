"""Test CAD parts module."""
import sys
sys.path.insert(0, 'src')

from cadgensis.cad.mechanisms.parts import Parts


def test_parts_import():
    parts = Parts()
    assert parts is not None