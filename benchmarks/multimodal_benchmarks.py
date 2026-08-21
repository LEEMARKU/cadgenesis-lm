"""
benchmarks/multimodal_benchmarks.py
===================================
Benchmarks for the Pillar 3 (v6.0) multimodal stack.

Measures wall-clock forward time for:
* the 11 modality encoders on realistic batch sizes,
* each fusion strategy (early / late / hierarchical / adaptive / attention),
* the cross-modal attention engine over the headline pairs,
* end-to-end ``MultimodalSystem.encode``.

Run with::

    python benchmarks/multimodal_benchmarks.py             # all
    python benchmarks/multimodal_benchmarks.py --sections encoders
"""

from __future__ import annotations

import argparse
import time

import torch

from cadgenesis.config import MultimodalConfig
from cadgenesis.multimodal import (
    FusionEngine,
    FusionStrategy,
    Modality,
    MultimodalSystem,
)
from cadgenesis.multimodal.cross_modal import HEADLINE_PAIRS
from cadgenesis.multimodal.encoders.audio import AudioDocument
from cadgenesis.multimodal.encoders.cad import CADDocument, CADFileFormat
from cadgenesis.multimodal.encoders.image import ImageDocument
from cadgenesis.multimodal.encoders.mesh import MeshDocument
from cadgenesis.multimodal.encoders.pdf import PDFDocument, PDFPage
from cadgenesis.multimodal.encoders.point_cloud import PointCloudDocument
from cadgenesis.multimodal.encoders.sensor import SensorDocument
from cadgenesis.multimodal.encoders.sketch import SketchDocument
from cadgenesis.multimodal.encoders.video import VideoDocument

torch.manual_seed(0)

SECTIONS = ("encoders", "fusion", "cross_modal", "end_to_end")


def time_forward(fn, reps: int) -> float:
    fn()
    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times)


def _sensor_docs(batch: int) -> list[SensorDocument]:
    return [
        SensorDocument(
            data=[[float((i + j) % 17)] for i in range(256)],
            channels=["ch0"],
            source="vibration",
        )
        for j in range(batch)
    ]


def _image_docs(batch: int) -> list[ImageDocument]:
    return [ImageDocument(data=torch.randn(3, 64, 64), width=64, height=64) for _ in range(batch)]


def bench_encoders(batch: int, reps: int) -> None:
    print(f"\n== modality encoders (forward, batch={batch}) ==")
    sys = MultimodalSystem.from_config(MultimodalConfig())
    inputs: dict[Modality, object] = {
        Modality.TEXT: "gearbox housing with four mounting holes and a rib",
        Modality.CAD: CADDocument(format=CADFileFormat.STEP, name="housing"),
        Modality.SKETCH: SketchDocument(name="profile"),
        Modality.IMAGE: _image_docs(batch),
        Modality.PDF: PDFDocument(name="drawing", pages=[PDFPage(number=1, text="bracket 4x")]),
        Modality.POINT_CLOUD: PointCloudDocument(points=torch.randn(4096, 3).tolist()),
        Modality.MESH: MeshDocument(vertices=torch.randn(2000, 3).tolist()),
        Modality.AUDIO: AudioDocument(data=[[0.01 * (i % 7)] for i in range(32000)]),
        Modality.VIDEO: VideoDocument(frames=[torch.randn(3, 32, 32) for _ in range(16)]),
        Modality.SENSOR: _sensor_docs(batch),
    }
    print(f"\n{'modality':>14} | {'ms/forward':>12}")
    for modality, data in inputs.items():
        ms = time_forward(lambda m=modality, d=data: sys.encode_modality(m, d), reps) * 1e3
        print(f"{modality.value:>14} | {ms:>12.3f}")


def bench_fusion(batch: int, reps: int) -> None:
    print(f"\n== fusion strategies (forward, batch={batch}) ==")
    feats = {m: torch.randn(batch, 256) for m in Modality}
    for strategy in FusionStrategy:
        engine = FusionEngine(strategy=strategy)
        ms = time_forward(lambda e=engine: e.forward(feats), reps) * 1e3
        print(f"{strategy.value:>14} | {ms:>12.3f} ms/forward")


def bench_cross_modal(batch: int, reps: int) -> None:
    print(f"\n== cross-modal attention (forward, batch={batch}) ==")
    sys = MultimodalSystem.from_config(MultimodalConfig())
    dims = sys.raw_feature_dims()
    header = f"{'pair':>24} | " + f"{'ms/forward':>12}"
    print(header)
    print("-" * len(header))
    for a_mod, b_mod in HEADLINE_PAIRS:
        a_feats = torch.randn(batch, dims[a_mod.value])
        b_feats = torch.randn(batch, dims[b_mod.value])
        ms = (
            time_forward(
                lambda am=a_mod, bm=b_mod, af=a_feats, bf=b_feats: sys.cross_modal.attend(
                    am, bm, af, bf
                ),
                reps,
            )
            * 1e3
        )
        print(f"{a_mod.value} <-> {b_mod.value:>16} | {ms:>12.3f}")


def bench_end_to_end(batch: int, reps: int) -> None:
    print(f"\n== end-to-end MultimodalSystem.encode (batch={batch}) ==")
    sys = MultimodalSystem.from_config(MultimodalConfig())
    inputs = {
        Modality.TEXT: "bracket",
        Modality.CAD: CADDocument(format=CADFileFormat.STEP, name="b"),
        Modality.SENSOR: _sensor_docs(1),
    }
    ms = time_forward(lambda: sys.encode(inputs), reps) * 1e3
    print(f"{ms:>12.3f} ms/forward")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sections", nargs="+", default=SECTIONS, choices=SECTIONS)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()
    for section in args.sections:
        if section == "encoders":
            bench_encoders(args.batch, args.reps)
        elif section == "fusion":
            bench_fusion(args.batch, args.reps)
        elif section == "cross_modal":
            bench_cross_modal(args.batch, args.reps)
        elif section == "end_to_end":
            bench_end_to_end(args.batch, args.reps)


if __name__ == "__main__":
    main()
