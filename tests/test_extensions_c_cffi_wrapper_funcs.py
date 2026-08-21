import sys
sys.path.insert(0, 'src')

from cadgensis.extensions.c.cffi_wrapper import point3d_create, point3d_add


def test_point3d_create():
    result = point3d_create(1.0, 2.0, 3.0)
    assert result is not None


def test_point3d_add():
    p1 = point3d_create(1.0, 2.0, 3.0)
    p2 = point3d_create(4.0, 5.0, 6.0)
    result = point3d_add(p1, p2)
    assert result is not None