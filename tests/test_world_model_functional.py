import sys
sys.path.insert(0, 'src')

from cadgensis.world_model.functional import FunctionalModel


def test_functional_init():
    model = FunctionalModel()
    assert model is not None