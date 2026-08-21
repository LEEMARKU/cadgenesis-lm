"""cadgenesis.datasets.multimodal
================================
Multimodal dataset loaders and augmentation (Pillar 3).

Provides :class:`MultimodalDataset` — a ``torch.utils.data.Dataset`` holding
aligned samples ``{modality: input}`` plus an optional fused label — and
:class:`MultimodalBatchCollator`, which encodes each sample through a
:class:`~cadgenesis.multimodal.multimodal.MultimodalSystem` into
``MultimodalEncoding``-shaped batches for training.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from cadgenesis.multimodal.common import Modality


@dataclass
class MultimodalSample:
    """One aligned multimodal sample."""

    inputs: dict[Modality, Any]
    label: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": {m.value: v for m, v in self.inputs.items()},
            "label": self.label,
        }


@dataclass
class MultimodalBatch:
    """A collated + encoded batch."""

    inputs: dict[Modality, torch.Tensor]
    labels: Any
    batch_size: int

    def raw_features(self) -> dict[Modality, torch.Tensor]:
        return self.inputs

    def modality_names(self) -> list[str]:
        return [m.value for m in self.inputs]


class MultimodalDataset(Dataset):
    """Dataset of aligned multimodal samples.

    Parameters
    ----------
    samples : list[MultimodalSample]
        Aligned samples.
    transform : Callable[[dict[Modality, Any]], Any] | None
        Optional per-sample augmentation applied lazily.
    """

    def __init__(
        self,
        samples: list[MultimodalSample],
        transform: Callable[[MultimodalSample], MultimodalSample] | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> MultimodalSample:
        sample = self.samples[index]
        return self.transform(sample) if self.transform is not None else sample

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, Any]],
        modality_key: str = "inputs",
        label_key: str = "label",
    ) -> MultimodalDataset:
        """Build a dataset from plain dict records.

        ``records`` entries look like::

            {"inputs": {"text": "...", "cad": {...}}, "label": 0}
        """
        samples: list[MultimodalSample] = []
        for record in records:
            raw_inputs = record[modality_key]
            inputs = {Modality(name): value for name, value in raw_inputs.items()}
            samples.append(MultimodalSample(inputs=inputs, label=record.get(label_key)))
        return cls(samples)

    @staticmethod
    def paired_sentences(
        text: list[str],
        cad: list[Any],
        labels: list[Any] | None = None,
    ) -> MultimodalDataset:
        """Convenience builder for text<->CAD aligned pairs."""
        if len(text) != len(cad):
            raise ValueError("text and cad lists must be the same length")
        samples = [
            MultimodalSample(
                inputs={Modality.TEXT: t, Modality.CAD: c},
                label=labels[i] if labels is not None else None,
            )
            for i, (t, c) in enumerate(zip(text, cad, strict=True))
        ]
        return MultimodalDataset(samples)


class MultimodalBatchCollator:
    """Collate a batch of :class:`MultimodalSample` into encoded tensors.

    ``system`` must implement ``encode_modality(modality, inputs)`` (the
    :class:`~cadgenesis.multimodal.multimodal.MultimodalSystem` contract).
    """

    def __init__(self, system: Any) -> None:
        self.system = system

    def __call__(self, batch: list[MultimodalSample]) -> MultimodalBatch:
        encoded: dict[Modality, list[torch.Tensor]] = {}
        labels: list[Any] = []
        for sample in batch:
            for modality, data in sample.inputs.items():
                features = self.system.encode_modality(modality, data)
                encoded.setdefault(modality, []).append(features)
            labels.append(sample.label)
        tensors = {modality: torch.cat(parts, dim=0) for modality, parts in encoded.items()}
        return MultimodalBatch(
            inputs=tensors,
            labels=labels,
            batch_size=len(batch),
        )


__all__ = [
    "MultimodalBatch",
    "MultimodalBatchCollator",
    "MultimodalDataset",
    "MultimodalSample",
]
