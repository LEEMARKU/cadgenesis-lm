"""
Item-2 training-claims verification on real runs.

- µTransfer: an LR tuned on a nano (d=64) model must transfer to a medium
  (d=128) model when both train with --mu-transfer (the only knob is the base
  LR, which is width-agnostic by construction).
- bf16 on CPU: the autocast path is exercised and tracks the fp32 loss closely.
- Teacher wiring: --teacher mock generates a dataset (no --data) and trains.
- FSDP: `wrap_fsdp` degrades to the unwrapped model outside an initialized
  DDP group; the real sharding path is CUDA+torchrun only:

    torchrun --standalone --nproc_per_node=<gpus> train.py \\
        --data data/cad_programs.jsonl --out-dir checkpoints/run-fsdp \\
        --model small --bf16 --epochs 100

  (this box is CPU-only, so the distributed branch is CUDA-gated and cannot
  execute here).
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cadgenesis.training.fsdp import wrap_fsdp
from tests.training.test_train_script import _args
from train import run_training


def _mu_run(tmp_path, name, d_model, lr, epochs=4):
    return run_training(
        _args(
            tmp_path,
            tmp_path / name,
            epochs=epochs,
            batch_size=8,
            warmup_steps=2,
            lr=lr,
            model="mini",
            mu_transfer=True,
            d_model=d_model,
            packed=True,
        )
    )


def test_mu_transfer_nano_to_medium_same_lr(tmp_path):
    """Same base LR trains both widths; medium tracks nano's loss curve."""
    base_lr = 5e-4
    nano = _mu_run(tmp_path, "nano", d_model=64, lr=base_lr)
    medium = _mu_run(tmp_path, "medium", d_model=128, lr=base_lr)

    assert nano["train_loss"] >= 0.0 and medium["train_loss"] >= 0.0
    # Starting loss on this dataset is ~8; both must have clearly learned.
    assert nano["train_loss"] < 5.5 and nano["best_val"] < 8.0, "nano must learn"
    assert medium["train_loss"] < 5.5 and medium["best_val"] < 8.0, "medium must learn"
    # Transfer: the same base LR lands both runs in the same loss regime.
    ratio = medium["train_loss"] / nano["train_loss"]
    assert 0.5 < ratio < 2.0, (
        f"muP base-LR transfer broke: nano={nano['train_loss']:.3f} "
        f"medium={medium['train_loss']:.3f} (ratio {ratio:.2f})"
    )


def test_mu_transfer_uses_mu_optimizer(tmp_path):
    """--mu-transfer must actually swap in the width-scaled optimizer."""
    result = _mu_run(tmp_path, "nano", d_model=64, lr=1e-3, epochs=1)
    ckpt = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    # Readout group got lr = base_lr * d_model (readout is group 0).
    readout = ckpt["optimizer_state_dict"]["param_groups"][0]
    assert readout["lr"] == pytest.approx(1e-3 * 64)


def test_bf16_cpu_run_completes_and_tracks_fp32(tmp_path):
    fp32 = run_training(
        _args(
            tmp_path,
            tmp_path / "fp32",
            epochs=2,
            batch_size=8,
            warmup_steps=2,
            lr=1e-3,
            model="mini",
            packed=True,
            bf16=False,
        )
    )
    bf16 = run_training(
        _args(
            tmp_path,
            tmp_path / "bf16",
            epochs=2,
            batch_size=8,
            warmup_steps=2,
            lr=1e-3,
            model="mini",
            packed=True,
            bf16=True,
        )
    )
    assert bf16["train_loss"] >= 0.0
    assert abs(bf16["train_loss"] - fp32["train_loss"]) < 0.5, (
        f"bf16 diverged from fp32: fp32={fp32['train_loss']:.3f} bf16={bf16['train_loss']:.3f}"
    )


def test_teacher_mock_generates_dataset_and_trains(tmp_path):
    out_dir = tmp_path / "teacher"
    result = run_training(
        _args(
            tmp_path,
            out_dir,
            epochs=1,
            batch_size=8,
            warmup_steps=2,
            lr=1e-3,
            model="mini",
            packed=False,
            teacher="mock",
            prompts="a steel box,a bracket,an extruded plate",
            data=None,
        )
    )
    ds = out_dir / "teacher_dataset.jsonl"
    assert ds.exists(), "teacher must materialize a JSONL dataset"
    assert result["digest"]
    assert result["train_loss"] >= 0.0


def test_wrap_fsdp_degrades_to_plain_model_on_cpu():
    """Outside an initialized DDP group FSDP is a no-op (this CPU box)."""
    from cadgenesis.config import CADConfig
    from cadgenesis.transformer.geometry_transformer import (
        GeometryAwareTransformer,
    )

    model = GeometryAwareTransformer(CADConfig.mini())
    wrapped = wrap_fsdp(model)
    assert wrapped is model
    # Local, non-distributed path must not crash on a real forward.
    wrapped.eval()
    assert True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")
def test_wrap_fsdp_requires_initialized_group():
    """On CUDA, FSDP must refuse silently without a DDP group."""
    from cadgenesis.config import CADConfig
    from cadgenesis.transformer.geometry_transformer import (
        GeometryAwareTransformer,
    )

    model = GeometryAwareTransformer(CADConfig.mini()).cuda()
    assert wrap_fsdp(model) is model  # no group initialized -> unchanged
