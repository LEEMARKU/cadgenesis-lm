import sys
sys.path.insert(0, 'src')

from cadgensis.trust.registries import Registries


def test_registries_init():
    registries = Registries()
    assert registries is not None