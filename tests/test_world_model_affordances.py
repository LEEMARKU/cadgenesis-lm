import sys
sys.path.insert(0, 'src')

from cadgensis.world_model.affordances import Affordances


def test_affordances_init():
    affordances = Affordances()
    assert affordances is not None