import sys
sys.path.insert(0, 'src')

from cadgensis.extensions.mlir.mlir_integration import MLIRIntegration


def test_mlir_integration_init():
    mlir = MLIRIntegration()
    assert mlir is not None