"""Test CAD modeling primitives module."""
import sys
sys.path.insert(0, 'src')


def test_cad_modeling_primitives():
    from cadgensis.cad.modeling.primitives import Primitives
    primitives = Primitives()
    assert primitives is not None