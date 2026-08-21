"""Test agents infrastructure module."""
import sys
sys.path.insert(0, 'src')


def test_agents_infrastructure():
    from cadgensis.agents.infrastructure import InfrastructureManager
    infra = InfrastructureManager()
    assert infra is not None