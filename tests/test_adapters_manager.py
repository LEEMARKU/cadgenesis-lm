"""Test adapters manager module."""
import sys
sys.path.insert(0, 'src')


def test_adapters_manager_import():
    from cadgenesis import adapters
    # Check if manager submodule exists
    import cadgenesis.adapters.manager as m
    assert m is not None