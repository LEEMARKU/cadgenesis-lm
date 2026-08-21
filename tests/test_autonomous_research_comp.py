"""Test autonomous research comparator module."""
import sys
sys.path.insert(0, 'src')


def test_autonomous_research_comp():
    from cadgensis.autonomous_research.comparator import Comparator
    comp = Comparator()
    assert comp is not None