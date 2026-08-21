import sys
sys.path.insert(0, 'src')

from cadgensis.knowledge_network.network import KnowledgeNetwork


def test_knowledge_network_init():
    network = KnowledgeNetwork()
    assert network is not None