import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.geometry_attention import GeometryAttention


def test_geometry_attention_init():
    attention = GeometryAttention()
    assert attention is not None