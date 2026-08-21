"""cadgenesis.datasets
===================
Dataset loaders, builders, and pipelines for CADGenesis training and evaluation.
"""

from cadgenesis.datasets.cad_jsonl import (
    CADJsonlDataset,
    load_jsonl,
    minhash_dedup,
    minhash_signature,
    split_records,
)
from cadgenesis.datasets.cad_program_synth import (
    NUM_MAX,
    build_synthetic_records,
    token_coverage,
    write_synthetic_jsonl,
)
from cadgenesis.datasets.multimodal import (
    MultimodalBatch,
    MultimodalBatchCollator,
    MultimodalDataset,
    MultimodalSample,
)

__all__ = [
    "NUM_MAX",
    "CADJsonlDataset",
    "MultimodalBatch",
    "MultimodalBatchCollator",
    "MultimodalDataset",
    "MultimodalSample",
    "build_synthetic_records",
    "load_jsonl",
    "minhash_dedup",
    "minhash_signature",
    "split_records",
    "token_coverage",
    "write_synthetic_jsonl",
]
