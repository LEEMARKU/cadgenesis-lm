import sys
sys.path.insert(0, 'src')

from cadgensis.trust.provenance import Provenance


def test_provenance_init():
    provenance = Provenance()
    assert provenance is not None