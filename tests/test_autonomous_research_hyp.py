"""Test autonomous research hyperparameter search module."""
import sys
sys.path.insert(0, 'src')


def test_autonomous_research_hyp():
    from cadgensis.autonomous_research.hyperparameter_search import HyperparameterSearch
    search = HyperparameterSearch()
    assert search is not None