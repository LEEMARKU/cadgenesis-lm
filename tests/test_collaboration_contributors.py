import sys
sys.path.insert(0, 'src')

from cadgensis.collaboration.contributors import Contributors


def test_contributors_init():
    contributors = Contributors()
    assert contributors is not None