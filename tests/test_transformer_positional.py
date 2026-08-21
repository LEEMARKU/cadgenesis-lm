import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.positional import GeometryPositionalEncoding


def test_positional_encoding_init():
    encoding = GeometryPositionalEncoding()
    assert encoding is not None