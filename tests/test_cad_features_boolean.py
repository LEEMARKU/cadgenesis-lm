"""Test CAD features boolean module."""
import sys
sys.path.insert(0, 'src')


def test_cad_features_boolean():
    from cadgensis.cad.features.boolean import BooleanFeature
    feature = BooleanFeature()
    assert feature is not None