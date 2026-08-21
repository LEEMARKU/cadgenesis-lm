"""Test collaboration contributors module."""
import sys
sys.path.insert(0, 'src')


def test_collab_contributors():
    from cadgensis.collaboration.contributors import Contributors
    contributors = Contributors()
    assert contributors is not None