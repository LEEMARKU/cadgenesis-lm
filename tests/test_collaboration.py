"""Test collaboration module."""
import sys
sys.path.insert(0, 'src')


def test_collaboration_imports():
    from cadgensis import collaboration
    assert collaboration is not None