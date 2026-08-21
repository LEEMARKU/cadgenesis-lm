"""Test CAD features patterns module."""
import sys
sys.path.insert(0, 'src')


def test_cad_features_patterns():
    from cadgensis.cad.features.patterns import PatternsFeature
    feature = PatternsFeature()
    assert feature is not None