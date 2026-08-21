"""Test CAD modeling brep module."""
import sys
sys.path.insert(0, 'src')


def test_cad_modeling_brep():
    from cadgensis.cad.modeling.brep import BREPModeler
    modeler = BREPModeler()
    assert modeler is not None