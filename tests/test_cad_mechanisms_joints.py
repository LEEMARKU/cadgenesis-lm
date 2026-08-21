"""Test CAD mechanisms joints module."""
import sys
sys.path.insert(0, 'src')


def test_cad_mechanisms_joints():
    from cadgensis.cad.mechanisms.joints import Joints
    joints = Joints()
    assert joints is not None