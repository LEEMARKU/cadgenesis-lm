"""Test CAD features dress module."""
import sys
sys.path.insert(0, 'src')


def test_cad_features_dress():
    from cadgensis.cad.features.dress import DressFeature
    feature = DressFeature()
    assert feature is not None