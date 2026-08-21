import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.self_designing.adaptive_heads import AdaptiveHeads


def test_adaptive_heads_init():
    heads = AdaptiveHeads()
    assert heads is not None