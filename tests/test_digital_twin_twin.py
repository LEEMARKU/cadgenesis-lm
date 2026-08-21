import sys
sys.path.insert(0, 'src')

from cadgensis.digital_twin.twin import DigitalTwin


def test_digital_twin_init():
    twin = DigitalTwin()
    assert twin is not None