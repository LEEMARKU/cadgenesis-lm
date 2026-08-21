import sys
sys.path.insert(0, 'src')

from cadgensis.tokenizer.geometry_tokens import GeometryTokens


def test_geometry_tokens_init():
    tokens = GeometryTokens()
    assert tokens is not None