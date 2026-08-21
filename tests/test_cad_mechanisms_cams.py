"""Test CAD mechanisms cams module."""
import sys
sys.path.insert(0, 'src')


def test_cad_mechanisms_cams():
    from cadgensis.cad.mechanisms.cams import CAMs
    cams = CAMs()
    assert cams is not None