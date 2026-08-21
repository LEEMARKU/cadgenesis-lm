"""Test collaboration dataset marketplace module."""
import sys
sys.path.insert(0, 'src')


def test_collab_dataset_mktp():
    from cadgensis.collaboration.dataset_marketplace import DatasetMarketplace
    marketplace = DatasetMarketplace()
    assert marketplace is not None