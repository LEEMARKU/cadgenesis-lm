"""Engineering Trust Infrastructure (Pillar 13).

Provides cryptographic provenance, integrity verification, and tamper-resistant
audit trails for datasets, models, CAD assets, experiments, plugins, and adapters.
Blockchain backend is optional and pluggable.
"""

from __future__ import annotations

from .blockchain import BlockchainAdapter, BlockchainBackend, BlockchainConfig
from .core import RecordType, TrustConfig, TrustLayer, TrustRecord
from .ledger import ExperimentLedger, FederatedTrainingLedger
from .provenance import (
    CADAssetProvenance,
    DatasetProvenance,
    ExperimentProvenance,
    ModelProvenance,
)
from .registries import (
    AdapterRegistry,
    PluginRegistry,
    SecureModelRegistry,
)

__all__ = [
    "AdapterRegistry",
    "BlockchainAdapter",
    "BlockchainBackend",
    "BlockchainConfig",
    "CADAssetProvenance",
    "DatasetProvenance",
    "ExperimentLedger",
    "ExperimentProvenance",
    "FederatedTrainingLedger",
    "ModelProvenance",
    "PluginRegistry",
    "RecordType",
    "SecureModelRegistry",
    "TrustConfig",
    "TrustLayer",
    "TrustRecord",
]
