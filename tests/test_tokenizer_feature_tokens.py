import sys
sys.path.insert(0, 'src')

from cadgensis.tokenizer.feature_tokens import FeatureTokens


def test_feature_tokens_init():
    tokens = FeatureTokens()
    assert tokens is not None