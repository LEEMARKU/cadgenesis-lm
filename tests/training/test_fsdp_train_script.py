"""
FSDP / torchrun wiring for ``train.py`` (verified on CPU with a gloo group).

A real sharded run needs CUDA + ``torchrun`` (nccl), which cannot execute on
this CPU-only box (torchrun's TCP store also needs a libuv-enabled torch
build).  What IS verifiable here:

- ``_init_distributed`` degrades to single-process outside torchrun;
- with a pre-initialised gloo group the FSDP path runs end-to-end
  (world_size=1 -> no-op wrap, rank-0 checkpoint I/O, ``include_optimizer``
  skipped under FSDP, resume without an optimizer state);
- the run digest is shard-count sensitive (different world sizes are
  different experiments and refuse to resume).
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train
from cadgenesis.config import CADConfig
from tests.training.test_train_script import _args


def _init_gloo_world1():
    f = Path(os.environ.get("TMP", ".")) / "cadgenesis_gloo_store"
    if f.exists():
        f.unlink()
    if dist.is_initialized():
        return
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(f).replace("\\", "/"),
        rank=0,
        world_size=1,
    )


def _ns(tmp_path, out_dir, fsdp=True, **kw):
    return argparse.Namespace(
        data=None,
        out_dir=str(out_dir),
        epochs=1,
        batch_size=8,
        seed=7,
        lr=1e-3,
        schedule="wsd",
        packed=True,
        val_fraction=0.2,
        max_records=None,
        model="mini",
        warmup_steps=2,
        resume_from=None,
        bf16=False,
        fsdp=fsdp,
        mu_transfer=False,
        d_model=None,
        optimizer="adamw",
        device=None,
        packed_max_src_len=None,
        packed_max_tgt_len=None,
        teacher=None,
        prompts=None,
        prompts_file=None,
        **kw,
    )


def test_init_distributed_degrades_outside_torchrun():
    saved = os.environ.get("RANK")
    os.environ.pop("RANK", None)
    try:
        info = train._init_distributed(_ns(None, Path(".")))
        assert not info.enabled
    finally:
        if saved is not None:
            os.environ["RANK"] = saved
        else:
            os.environ.pop("RANK", None)


def test_resolve_device():
    assert train._resolve_device(train._DistInfo(False, 0, 0, 1)) == (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # Enabled but dist never initialized -> backend is not nccl -> CPU.
    info = train._DistInfo(True, 0, 0, 1)
    assert train._resolve_device(info) == "cpu"
    # Explicit device override wins.
    assert train._resolve_device(info, "cuda:0") == "cuda:0"
    assert train._resolve_device(info, "cpu") == "cpu"


def test_fsdp_digest_is_shard_sensitive(tmp_path):
    cfg = CADConfig.mini()
    d1 = train._run_digest(cfg, _ns(tmp_path, tmp_path / "a"), "h", 1)
    d8 = train._run_digest(cfg, _ns(tmp_path, tmp_path / "b"), "h", 8)
    assert d1 != d8


def test_fsdp_run_end_to_end_gloo_world1(tmp_path):
    _init_gloo_world1()
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    try:
        out_dir = tmp_path / "fsdp"
        result = train.run_training(
            _args(
                tmp_path,
                out_dir,
                epochs=2,
                batch_size=8,
                warmup_steps=2,
                lr=1e-3,
                model="mini",
                packed=True,
                fsdp=True,
            )
        )
        assert result["digest"]
        ckpt = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" not in ckpt, (
            "FSDP checkpoint must not carry a rank-local optimizer shard"
        )
        assert ckpt["run_digest"] == result["digest"]
    finally:
        dist.destroy_process_group()


def test_fsdp_resume_without_optimizer_state(tmp_path):
    _init_gloo_world1()
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    try:
        out_dir = tmp_path / "fsdp"
        r1 = train.run_training(
            _args(
                tmp_path,
                out_dir,
                epochs=1,
                batch_size=8,
                warmup_steps=2,
                lr=1e-3,
                model="mini",
                packed=True,
                fsdp=True,
            )
        )
        ckpt = out_dir / "checkpoint_epoch_1.pt"
        r2 = train.run_training(
            _args(
                tmp_path,
                out_dir / "resume",
                epochs=2,
                batch_size=8,
                warmup_steps=2,
                lr=1e-3,
                model="mini",
                packed=True,
                fsdp=True,
                resume_from=str(ckpt),
            )
        )
        # Same digest -> resume accepted; optimizer state rebuilt from scratch.
        assert r2["digest"] == r1["digest"]
        assert r2["train_loss"] >= 0.0
    finally:
        dist.destroy_process_group()
