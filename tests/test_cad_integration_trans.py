"""Test CAD integration transformer bridge module."""
import sys
sys.path.insert(0, 'src')


def test_cad_integration_trans():
    from cadgensis.cad.integration.transformer_bridge import TransformerBridge
    bridge = TransformerBridge()
    assert bridge is not None