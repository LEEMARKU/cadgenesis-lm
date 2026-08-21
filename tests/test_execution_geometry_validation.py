import sys
sys.path.insert(0, 'src')

from cadgensis.execution.geometry_validation import GeometryValidation


def test_geometry_validation_init():
    validation = GeometryValidation()
    assert validation is not None