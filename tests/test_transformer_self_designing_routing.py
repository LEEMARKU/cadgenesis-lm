import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.self_designing.routing import SelfDesigningRouting


def test_self_designing_routing_init():
    routing = SelfDesigningRouting()
    assert routing is not None