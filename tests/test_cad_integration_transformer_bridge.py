import sys
sys.path.insert(0, 'src')

from cadgensis.cad.integration.transformer_bridge import TransformerBridge


def test_transformer_bridge_init():
    bridge = TransformerBridge()
    assert bridge is not None