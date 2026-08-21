"""Two-rank FSDP demo (CPU/gloo) proving the multi-rank sharding path.

This machine has ONE CUDA GPU (GTX 1650) plus an Intel iGPU, which is not a
CUDA device and cannot participate in FSDP.  To prove the *multi-rank*
sharding code actually works we run two CPU ranks over gloo — FSDP shards the
params, DistributedSampler splits the data, gradients all-reduce, and rank 0
writes the full-state checkpoint.  The exact same code runs on a real
multi-GPU box via ``torchrun --nproc_per_node=N --fsdp``.

    python scripts/gpu/fsdp_cpu_demo.py
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import train

_STORE = Path(tempfile.mkdtemp()) / "fsdp_store"


def _worker(rank: int, world: int, out_dir: Path, data: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(_STORE).replace("\\", "/"),
        rank=rank,
        world_size=world,
    )
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    args = argparse.Namespace(
        data=data,
        out_dir=str(out_dir),
        epochs=2,
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
        fsdp=True,
        mu_transfer=False,
        d_model=None,
        optimizer="adamw",
        teacher=None,
        prompts=None,
        prompts_file=None,
    )
    result = train.run_training(args)
    print(f"[rank {rank}] digest={result['digest']} best_val={result['best_val']:.4f}")
    dist.destroy_process_group()


def main() -> None:
    if _STORE.exists():
        _STORE.unlink()
    world = 2
    out_dir = Path("checkpoints/fsdp-cpu-demo")
    mp.spawn(_worker, args=(world, out_dir, "data/cad_programs.jsonl"), nprocs=world, join=True)


if __name__ == "__main__":
    main()
