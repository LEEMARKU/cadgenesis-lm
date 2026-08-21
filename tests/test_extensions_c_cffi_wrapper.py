import sys
sys.path.insert(0, 'src')

from cadgensis.extensions.c import cffi_wrapper


def test_cffi_wrapper_init():
    wrapper = cffi_wrapper()
    assert wrapper is not None