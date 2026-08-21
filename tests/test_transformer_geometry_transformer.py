import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.geometry_transformer import GeometryTransformer


def test_geometry_transformer_init():
    transformer = GeometryTransformer()
    assert transformer is not None