"""Test agents event bus module."""
import sys
sys.path.insert(0, 'src')


def test_agents_event_bus():
    from cadgensis.agents.event_bus import EventBus
    bus = EventBus(history_limit=100)
    assert bus is not None