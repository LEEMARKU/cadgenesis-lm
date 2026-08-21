import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.interaction import InteractionLayer


def test_interaction_layer_init():
    layer = InteractionLayer()
    assert layer is not None