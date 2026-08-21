import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.self_designing.architecture


def test_self_designing_architecture_init():
    architecture = __import__('cadgensis.transformer.self_designing.architecture', fromlist=[''])
    assert architecture is not None