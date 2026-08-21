"""Test CAD visualization module."""
import sys
sys.path.insert(0, 'src')


def test_visual_import():
    # Just test that we can import the package
    from cadgensis import cad
    assert cad is not None