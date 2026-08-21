import sys
sys.path.insert(0, 'src')

from cadgensis.world_model.planning import Planning


def test_planning_init():
    planning = Planning()
    assert planning is not None