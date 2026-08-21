import sys
sys.path.insert(0, 'src')

from cadgensis.optimization.kernels import KernelBank


def test_kernel_bank_init():
    bank = KernelBank()
    assert bank is not None