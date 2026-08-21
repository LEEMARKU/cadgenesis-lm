"""Test agents fleet module."""
import sys
sys.path.insert(0, 'src')


def test_agents_fleet():
    from cadgensis.agents.fleet import FleetManager
    fleet = FleetManager(max_vehicles=10)
    assert fleet is not None