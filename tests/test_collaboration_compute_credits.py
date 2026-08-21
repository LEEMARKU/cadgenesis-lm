import sys
sys.path.insert(0, 'src')

from cadgensis.collaboration.compute_credits import ComputeCredits


def test_compute_credits_init():
    credits = ComputeCredits()
    assert credits is not None