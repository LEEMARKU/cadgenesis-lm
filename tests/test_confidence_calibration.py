import sys
sys.path.insert(0, 'src')

from cadgensis.confidence.calibration import CalibrationManager


def test_calibration_manager_init():
    cal = CalibrationManager()
    assert cal is not None