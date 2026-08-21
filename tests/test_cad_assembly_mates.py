"""Test CAD assembly mates module."""
import sys
sys.path.insert(0, 'src')


def test_cad_assembly_mates():
    from cadgensis.cad.assembly.mates import Mates
    mates = Mates()
    assert mates is not None