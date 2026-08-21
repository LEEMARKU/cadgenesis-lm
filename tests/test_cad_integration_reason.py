"""Test CAD integration reasoning bridge module."""
import sys
sys.path.insert(0, 'src')


def test_cad_integration_reason():
    from cadgensis.cad.integration.reasoning_bridge import ReasoningBridge
    bridge = ReasoningBridge()
    assert bridge is not None