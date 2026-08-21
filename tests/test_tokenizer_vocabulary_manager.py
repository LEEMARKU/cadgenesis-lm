import sys
sys.path.insert(0, 'src')

from cadgensis.tokenizer.vocabulary_manager import VocabularyManager


def test_vocabulary_manager_init():
    vocab = VocabularyManager()
    assert vocab is not None