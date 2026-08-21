import sys
sys.path.insert(0, 'src')

from cadgensis.tokenizer.legacy_shim import LegacyShim


def test_legacy_shim_init():
    shim = LegacyShim()
    assert shim is not None