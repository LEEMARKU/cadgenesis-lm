import sys
sys.path.insert(0, 'src')

from cadgensis.cad.integration.reasoning_bridge import ReasoningBridge


def test_reasoning_bridge_init():
    bridge = ReasoningBridge()
    assert bridge is not None