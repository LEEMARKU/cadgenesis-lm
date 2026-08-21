import sys
sys.path.insert(0, 'src')

from cadgensis.world_model.mechanical import MechanicalModel


def test_mechanical_init():
    model = MechanicalModel()
    assert model is not None