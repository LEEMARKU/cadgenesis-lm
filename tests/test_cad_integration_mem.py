"""Test CAD integration memory bridge module."""
import sys
sys.path.insert(0, 'src')


def test_cad_integration_mem():
    from cadgensis.cad.integration.memory_bridge import MemoryBridge
    bridge = MemoryBridge()
    assert bridge is not None