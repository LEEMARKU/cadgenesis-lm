"""Test CAD tokens module."""
import sys
sys.path.insert(0, 'src')


def test_tokens_import():
    from cadgensis.tokenizer import cad_tokens
    tokens = cad_tokens()
    assert tokens is not None