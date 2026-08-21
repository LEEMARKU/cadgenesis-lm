"""Test autonomous research failure analyzer module."""
import sys
sys.path.insert(0, 'src')


def test_autonomous_research_fail():
    from cadgensis.autonomous_research.failure_analyzer import FailureAnalyzer
    analyzer = FailureAnalyzer()
    assert analyzer is not None