import sys
sys.path.insert(0, 'src')

from cadgensis.transformer.memory_attention import MemoryAttention


def test_memory_attention_init():
    attention = MemoryAttention()
    assert attention is not None