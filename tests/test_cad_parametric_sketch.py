"""Test CAD parametric sketch module."""
import sys
sys.path.insert(0, 'src')


def test_cad_parametric_sketch():
    from cadgensis.cad.parametric.sketch import Sketch
    sketch = Sketch()
    assert sketch is not None