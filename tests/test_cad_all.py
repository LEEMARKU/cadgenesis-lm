"""Test all CAD modules."""
import sys
sys.path.insert(0, 'src')


def test_all_imports():
    import cadgenesis
    assert cadgenesis is not None