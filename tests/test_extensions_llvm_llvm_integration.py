import sys
sys.path.insert(0, 'src')

from cadgensis.extensions.llvm.llvm_integration import LLVMIntegration


def test_llvm_integration_init():
    llvm = LLVMIntegration()
    assert llvm is not None