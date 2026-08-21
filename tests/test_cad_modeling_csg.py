"""Test CAD modeling CSG module."""
import sys
sys.path.insert(0, 'src')


def test_cad_modeling_csg():
    from cadgensis.cad.modeling.csg import CSGModeler
    modeler = CSGModeler()
    assert modeler is not None