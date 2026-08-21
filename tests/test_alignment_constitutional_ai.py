"""Test alignment constitutional AI module."""
import sys
sys.path.insert(0, 'src')


def test_alignment_constitutional_ai():
    from cadgensis.alignment.constitutional_ai import RLAIFRewardModel
    model = RLAIFRewardModel()
    assert model is not None