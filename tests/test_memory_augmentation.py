import sys
sys.path.insert(0, 'src')

from cadgensis.memory.augmentation import MemoryAugmentation


def test_memory_augmentation_init():
    mem = MemoryAugmentation()
    assert mem is not None