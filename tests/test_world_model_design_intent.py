import sys
sys.path.insert(0, 'src')

from cadgensis.world_model.design_intent import DesignIntent


def test_design_intent_init():
    intent = DesignIntent()
    assert intent is not None