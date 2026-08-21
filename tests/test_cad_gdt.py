"""Test CAD GDT module."""
import sys
sys.path.insert(0, 'src')


def test_cad_gdt():
    from cadgensis.cad.gdt import GDT
    gdt = GDT()
    assert gdt is not None