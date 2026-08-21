import sys
sys.path.insert(0, 'src')

from cadgensis.multimodal.multimodal import MultimodalSystem


def test_multimodal_init():
    system = MultimodalSystem()
    assert system is not None