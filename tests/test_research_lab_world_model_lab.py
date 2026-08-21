import sys
sys.path.insert(0, 'src')

from cadgensis.research_lab.world_model_lab import WorldModelLab


def test_world_model_lab_init():
    lab = WorldModelLab()
    assert lab is not None