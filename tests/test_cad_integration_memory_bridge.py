import sys
sys.path.insert(0, 'src')

from cadgensis.cad.integration.memory_bridge import MemoryBridge


def test_memory_bridge_init():
    bridge = MemoryBridge()
    assert bridge is not None