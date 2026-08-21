import sys
sys.path.insert(0, 'src')

from cadgensis.tokenizer.language import LanguageTokenizer


def test_language_init():
    tokenizer = LanguageTokenizer()
    assert tokenizer is not None