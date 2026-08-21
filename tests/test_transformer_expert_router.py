import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.expert_router import ExpertRouter


def test_expert_router_init():
    router = ExpertRouter()
    assert router is not None