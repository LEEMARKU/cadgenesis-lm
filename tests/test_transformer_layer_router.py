import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.layer_router import LayerRouter


def test_layer_router_init():
    router = LayerRouter()
    assert router is not None