"""Test CAD manufacturing process module."""
import sys
sys.path.insert(0, 'src')


def test_cad_manufacturing_process():
    from cadgensis.cad.manufacturing.process import ManufacturingProcess
    process = ManufacturingProcess()
    assert process is not None