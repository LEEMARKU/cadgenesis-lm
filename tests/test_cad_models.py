"""Test CAD models module."""
import sys
sys.path.insert(0, 'src')

from cadgensis.cad.modeling import __all__ as model_list


def test_models_import():
    assert model_list is not None