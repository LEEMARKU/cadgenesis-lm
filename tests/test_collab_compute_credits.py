"""Test collaboration compute credits module."""
import sys
sys.path.insert(0, 'src')


def test_collab_compute_credits():
    from cadgensis.collaboration.compute_credits import ComputeCredits
    credits = ComputeCredits()
    assert credits is not None