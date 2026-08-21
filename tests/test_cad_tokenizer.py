"""Test CAD tokenizer module."""
import sys
sys.path.insert(0, 'src')

from cadgensis.tokenizer import cad_tokens


def test_cad_tokens_import():
    tokens = cad_tokens()
    assert tokens is not None