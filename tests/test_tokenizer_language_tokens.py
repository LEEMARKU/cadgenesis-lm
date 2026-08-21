import sys
sys.path.insert(0, 'src')

from cadgensis.tokenizer.language_tokens import LanguageTokens


def test_language_tokens_init():
    tokens = LanguageTokens()
    assert tokens is not None