"""Test CAD features solids module."""
import sys
sys.path.insert(0, 'src')


def test_cad_features_solids():
    from cadgensis.cad.features.solids import SolidsFeature
    feature = SolidsFeature()
    assert feature is not None