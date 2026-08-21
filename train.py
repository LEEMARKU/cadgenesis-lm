"""
train.py
========
Checkpointable training entrypoint for CADGenesis-LM (modern data pipeline).

Loads a real ``(prompt, CAD program)`` dataset (JSONL), extends the tokenizer
vocabulary to cover every token the data uses, then trains the model with a
WSD or cosine schedule, saving an auditable checkpoint every epoch:

    * ``run_digest``      — SHA-256 of (config, seed, dataset hash, schedule)
    * ``dataset_sha256``  — content hash of the JSONL the run was trained on
    * ``seed``            — the seed that produced this run (reproducibility)

Resuming from a checkpoint rebuilds the model from the checkpoint's own
config and refuses to continue when the digest no longer matches, so a
"resumed" run is guaranteed to be the same run, not a different experiment.

Usage::

    python train.py --data data/cad_programs.jsonl --out-dir checkpoints/run1
    python train.py --teacher mock --prompts "a steel box, a bracket" --epochs 3
    python train.py --teacher deepseek --prompts-file data/prompts.txt  # GPU box
    python train.py --resume-from checkpoints/run1/checkpoint_epoch_3.pt

GPU (FSDP multi-GPU, run under torchrun)::

    torchrun --standalone --nproc_per_node=8 train.py \\
        --data data/cad_programs.jsonl --out-dir checkpoints/run-fsdp \\
        --model small --fsdp --bf16 --epochs 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from cadgenesis.config import CADConfig
from cadgenesis.datasets.cad_jsonl import CADJsonlDataset, load_jsonl, split_records
from cadgenesis.datasets.cad_program_synth import token_coverage
from cadgenesis.tokenizer import (
    AutonomousCADTokenizer,
    TokenFamily,
    restore_vocab_tokens,
    vocab_tokens,
)
from cadgenesis.training.packing import pack_batch
from cadgenesis.training.scheduler import build_scheduler
from cadgenesis.training.trainer import CADTrainer, cad_collate_fn
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer
from cadgenesis.utils.hashing import content_hash, sha256_file


class _CadPairs(Dataset):
    """Yields ``(src_ids, tgt_ids)`` pairs; optionally terminates the target
    with BOS/EOS (padded collate) or leaves it raw (packing adds them)."""

    def __init__(self, base, bos: int, eos: int, add_terminals: bool):
        self.base = base
        self.bos = bos
        self.eos = eos
        self.add_terminals = add_terminals

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        src, cad = self.base[idx]
        if self.add_terminals:
            cad = [self.bos, *cad, self.eos]
        return src, cad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CADGenesis-LM on a real (prompt, CAD program) dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="JSONL dataset of {text, cad} records (or omit + use --teacher).",
    )
    parser.add_argument(
        "--teacher",
        choices=["mock", "deepseek"],
        default=None,
        help="Generate the dataset with the DeepSeek-R1 teacher instead of "
        "--data: 'mock' is instant, 'deepseek' downloads/loads "
        "DeepSeek-R1-Distill-Qwen-1.5B locally (~3 GB, ~30 s/program on "
        "CPU — use a GPU box for real datasets).",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Comma-separated prompts used by --teacher (or prefix @file to "
        "read one prompt per line).",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="File with one prompt per line for --teacher.",
    )
    parser.add_argument("--out-dir", type=str, default="checkpoints", help="Checkpoint output dir.")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--seed", type=int, default=0, help="Determinism seed.")
    parser.add_argument("--lr", type=float, default=None, help="Override peak learning rate.")
    parser.add_argument(
        "--schedule",
        choices=["wsd", "cosine"],
        default="wsd",
        help="Learning-rate schedule (WSD = warmup-stable-decay).",
    )
    parser.add_argument(
        "--packed",
        action="store_true",
        default=True,
        help="Use token-efficient sequence packing (default).",
    )
    parser.add_argument(
        "--no-packed",
        dest="packed",
        action="store_false",
        help="Use plain pad-to-max collation instead of packing.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of records held out for validation.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Only load the first N records from the dataset.",
    )
    parser.add_argument(
        "--model",
        choices=["mini", "small", "base", "1.5b", "large"],
        default="mini",
        help="Architecture preset (small+ is for a GPU box).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=200,
        help="Linear warmup steps for the LR schedule.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint path to resume training from.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Enable bf16 autocast (CUDA, or CPU when the CPU supports it).",
    )
    parser.add_argument(
        "--mu-transfer",
        action="store_true",
        default=False,
        help="Apply µP (maximal update parametrization): rescale init and use "
        "width-scaled LR groups so an LR tuned on nano transfers to large.",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        default=False,
        help="FSDP multi-GPU training under torchrun (initialises the "
        "distributed group from $LOCAL_RANK, shards data across ranks, "
        "saves full-state checkpoints from rank 0).",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=None,
        help="Override d_model (e.g. 64 for a nano-scale µP ladder run).",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adamw8bit"],
        default="adamw",
        help="Optimizer: adamw (fp32 states) or adamw8bit (8-bit, 1.5B-scale "
        "models on limited RAM/VRAM; requires bitsandbytes).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force training device ('cpu' or 'cuda'). Default: auto. Use "
        "'cpu' for 1.5B-scale models on boxes with small VRAM.",
    )
    parser.add_argument(
        "--packed-max-src-len",
        type=int,
        default=None,
        help="Override packed encoder row length (default 256). Shrink to "
        "32-64 for 1.5B-scale models on CPU.",
    )
    parser.add_argument(
        "--packed-max-tgt-len",
        type=int,
        default=None,
        help="Override packed decoder row length (default 128). Shrink to "
        "16-32 for 1.5B-scale models on CPU.",
    )
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


class _DistInfo:
    """Resolved torch.distributed state for the current process."""

    def __init__(self, enabled: bool, rank: int, local_rank: int, world_size: int):
        self.enabled = enabled
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size

    @property
    def primary(self) -> bool:
        return not self.enabled or self.rank == 0


def _init_distributed(args: argparse.Namespace) -> _DistInfo:
    """Initialise a torch.distributed group for ``--fsdp`` (torchrun).

    Returns a ``_DistInfo``; when distributed is not requested (or the group
    cannot start — e.g. ``--fsdp`` outside torchrun) the process behaves as a
    single rank on CPU.
    """
    if not args.fsdp:
        return _DistInfo(False, 0, 0, 1)
    try:
        import torch.distributed as dist
    except ImportError:
        return _DistInfo(False, 0, 0, 1)
    if "RANK" not in os.environ:
        print(
            "[Warning] --fsdp requested but no torchrun environment (RANK "
            "unset); continuing as a single process."
        )
        return _DistInfo(False, 0, 0, 1)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
        else:
            dist.init_process_group(backend="gloo")
    return _DistInfo(True, dist.get_rank(), local_rank, dist.get_world_size())


def _broadcast_path(path: str, dist_info: _DistInfo) -> str:
    """Rank 0 resolves the dataset path; all ranks agree on it."""
    if not dist_info.enabled:
        return path
    import torch.distributed as dist

    obj = [path] if dist_info.rank == 0 else [None]
    dist.broadcast_object_list(obj, src=0)
    return obj[0]


def _resolve_device(dist_info: _DistInfo, device: str | None = None) -> str:
    if device is not None:
        return device
    if dist_info.enabled:
        # The device must match the process-group backend: nccl -> CUDA,
        # gloo -> CPU (e.g. the 2-rank CPU FSDP demo).
        import torch.distributed as dist

        backend = dist.get_backend() if dist.is_initialized() else ""
        if backend == "nccl" and torch.cuda.is_available():
            return f"cuda:{dist_info.local_rank}"
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _FSDP_CONFIG():
    from cadgenesis.training.fsdp import FSDPConfig

    return FSDPConfig(sharding_strategy="full_shard", mixed_precision="bf16")


def _register_dataset_tokens(records, tokenizer) -> int:
    """Extend the tokenizer vocab so every token the data uses has an id."""
    covered = token_coverage(records)
    tok2id = tokenizer.vocab.to_tok2id()
    missing = sorted(tok for tok in covered if tok2id.get(tok) is None)
    if missing:
        tokenizer.vocab.register_many(
            [
                (
                    tok,
                    TokenFamily.NUMERIC if tok.startswith("NUM_") else TokenFamily.FEATURE,
                )
                for tok in missing
            ]
        )
    return len(missing)


def _write_jsonl(records, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_loader(
    records,
    tokenizer,
    config: CADConfig,
    packed: bool,
    seed: int,
    out_dir: Path,
    name: str,
    shuffle: bool,
    generator=None,
    dist_rank=None,
    dist_world_size=None,
):
    jsonl = out_dir / f"{name}.jsonl"
    _write_jsonl(records, jsonl)
    base = CADJsonlDataset(jsonl, tokenizer)
    ds = _CadPairs(base, bos=tokenizer.bos_id, eos=tokenizer.eos_id, add_terminals=not packed)
    if packed:
        collate = lambda batch: pack_batch(  # noqa: E731
            batch,
            max_src_len=config.training.packed_max_src_len,
            max_tgt_len=config.training.packed_max_tgt_len,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            pad_id=tokenizer.pad_id,
            seed=seed,
        )
    else:
        collate = cad_collate_fn
    sampler = None
    if dist_rank is not None and dist_world_size is not None and dist_world_size > 1 and shuffle:
        # Each rank trains on a different shard; ``set_epoch`` per epoch keeps
        # resume byte-exact on every rank (same seed, same shard).
        sampler = torch.utils.data.DistributedSampler(
            ds,
            num_replicas=dist_world_size,
            rank=dist_rank,
            shuffle=True,
            seed=seed,
        )
        sampler.set_epoch(1)
    return DataLoader(
        ds,
        batch_size=config.training.batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        collate_fn=collate,
        generator=generator,
    )


def _build_config(args: argparse.Namespace, checkpoint: dict | None) -> CADConfig:
    if checkpoint is not None:
        config = CADConfig.from_dict(checkpoint["config"])
    elif args.model in ("small", "base", "1.5b", "large"):
        config = CADConfig.from_preset(args.model)
    else:
        config = CADConfig.mini()
    if args.epochs is not None:
        config.training.max_epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.lr = args.lr
    config.training.schedule = args.schedule
    config.training.warmup_steps = args.warmup_steps
    if args.bf16 or args.fsdp:
        config.training.mixed_precision = "bf16"
    if args.d_model is not None:
        config.model.d_model = args.d_model
    if args.packed_max_src_len is not None:
        config.training.packed_max_src_len = args.packed_max_src_len
    if args.packed_max_tgt_len is not None:
        config.training.packed_max_tgt_len = args.packed_max_tgt_len
    return config


def _run_digest(
    config: CADConfig, args: argparse.Namespace, data_hash: str, world_size: int = 1
) -> str:
    # ``max_epochs`` is normalised away so that *extending* the horizon on
    # resume is the same run (the WSD scheduler is rebuilt for the new horizon
    # anyway); a different lr / seed / dataset / schedule / batch / warmup /
    # precision / parametrization / shard-count is still a different experiment
    # and will be refused on resume.
    snapshot = config.to_dict()
    snapshot["training"]["max_epochs"] = 0
    return content_hash(
        snapshot,
        args.seed,
        data_hash,
        args.max_records,
        args.schedule,
        args.packed,
        args.model,
        args.mu_transfer,
        args.bf16,
        args.fsdp,
        args.optimizer,
        world_size,
    )


def _load_prompts(args: argparse.Namespace) -> list[str]:
    prompts: list[str] = []
    if args.prompts_file:
        with open(args.prompts_file, encoding="utf-8") as fh:
            prompts.extend(line.strip() for line in fh if line.strip())
    if args.prompts:
        text = args.prompts
        if text.startswith("@"):
            with open(text[1:], encoding="utf-8") as fh:
                prompts.extend(line.strip() for line in fh if line.strip())
        else:
            prompts.extend(p.strip() for p in text.split(",") if p.strip())
    return prompts


def _generate_teacher_data(args: argparse.Namespace, out_dir: Path) -> str:
    from cadgenesis.adapters.deepseek_r1 import (
        DeepSeekR1DataGenerator,
        DeepSeekR1Reasoner,
        DeepSeekR1Teacher,
        MockDeepSeekR1Teacher,
    )

    prompts = _load_prompts(args)
    if not prompts:
        raise SystemExit("no prompts for --teacher: pass --prompts 'a,b,c' or --prompts-file")
    if args.teacher == "deepseek":
        teacher_device = _resolve_device(_DistInfo(False, 0, 0, 1))
        reasoner = DeepSeekR1Reasoner(
            device=teacher_device,
            torch_dtype=torch.bfloat16,
            max_new_tokens=64,
            temperature=0.7,
        )
        print(
            f"loading DeepSeek-R1-Distill-Qwen-1.5B on {teacher_device} "
            "(first run downloads ~3 GB)..."
        )
        teacher = DeepSeekR1Teacher(reasoner)
    else:
        teacher = MockDeepSeekR1Teacher()
    generator = DeepSeekR1DataGenerator(teacher, vocab=None)
    records = generator.generate_feature_records(prompts, reasoning=False)
    path = out_dir / "teacher_dataset.jsonl"
    _write_jsonl(records, path)
    print(f"[Teacher] generated {len(records)} records -> {path}")
    return str(path)


def _resolve_data(args: argparse.Namespace, out_dir: Path) -> str:
    if args.data:
        return args.data
    if args.teacher is None:
        raise SystemExit("provide --data (JSONL) or --teacher to generate a dataset")
    return _generate_teacher_data(args, out_dir)


def run_training(args: argparse.Namespace) -> dict:
    _seed_everything(args.seed)
    dist_info = _init_distributed(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Teacher generation runs on the primary rank only (DeepSeek-R1 sampling
    # is non-deterministic, so every rank must train on the SAME generated
    # file); the resolved path is then broadcast to all ranks.
    data = _resolve_data(args, out_dir) if dist_info.primary else args.data
    data = _broadcast_path(data, dist_info)

    records = load_jsonl(data, max_records=args.max_records)
    if not records:
        raise SystemExit(f"no usable records in {data}")

    checkpoint = None
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location="cpu", weights_only=False)

    tokenizer = AutonomousCADTokenizer.build_mini()
    if args.model in ("small", "base", "1.5b", "large"):
        tokenizer = AutonomousCADTokenizer.build()
    if checkpoint is not None:
        restored = restore_vocab_tokens(tokenizer, checkpoint.get("vocab_tokens", []))
        print(f"[Tokenizer] restored {restored} tokens from checkpoint vocab")
    registered = _register_dataset_tokens(records, tokenizer)
    # The encoder embeds the *text* side (``lang_embed``); without a built
    # language vocabulary every word falls back to <unk> and the model can
    # only learn an unconditional CAD-sequence prior.  Build the language
    # vocab from the dataset texts (deterministic: sorted word set).
    tokenizer.build_lang_vocab([r.get("text", "") for r in records])
    print(
        f"[Tokenizer] vocab={tokenizer.vocab_size:,} (registered {registered} "
        f"new tokens from data, lang_vocab={tokenizer.lang_tok.vocab_size:,})"
    )

    data_hash = sha256_file(data)
    train_records, val_records = split_records(records, args.val_fraction, args.seed)

    config = _build_config(args, checkpoint=checkpoint)
    digest = _run_digest(
        config, args, data_hash, world_size=dist_info.world_size if dist_info.enabled else 1
    )

    train_dl = _build_loader(
        train_records,
        tokenizer,
        config,
        packed=args.packed,
        seed=args.seed,
        out_dir=out_dir,
        name="train",
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        dist_rank=dist_info.rank if dist_info.enabled else None,
        dist_world_size=dist_info.world_size if dist_info.enabled else None,
    )
    val_dl = None
    if val_records:
        val_dl = _build_loader(
            val_records,
            tokenizer,
            config,
            packed=args.packed,
            seed=args.seed,
            out_dir=out_dir,
            name="val",
            shuffle=False,
        )
    train_gen = train_dl.generator

    model = GeometryAwareTransformer(config)
    if args.mu_transfer:
        from cadgenesis.training.mu_transfer import apply_mu_transfer

        # Init scaling must happen BEFORE FSDP flattens the parameters.
        apply_mu_transfer(model, config.model.d_model)

    trainer = CADTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        device=_resolve_device(dist_info, args.device),
        use_fsdp=dist_info.enabled,
        fsdp_config=None if not dist_info.enabled else _FSDP_CONFIG(),
        optimizer=args.optimizer,
    )
    if args.mu_transfer:
        if dist_info.enabled:
            print(
                "[Warning] --mu-transfer + --fsdp: applying the µP init "
                "scaling only; FSDP owns the optimizer LR groups (readout "
                "isolation needs use_orig_params)."
            )
        else:
            from cadgenesis.training.mu_transfer import build_mu_optimizer

            trainer.optimizer = build_mu_optimizer(
                model,
                base_lr=config.training.lr,
                d_model=config.model.d_model,
                weight_decay=config.training.weight_decay,
            )
    effective_steps = (
        math.ceil(len(train_dl) / max(1, config.training.grad_accum_steps))
        * config.training.max_epochs
    )
    trainer.scheduler = build_scheduler(
        trainer.optimizer,
        schedule=config.training.schedule,
        num_train_steps=effective_steps,
        warmup_steps=config.training.warmup_steps,
        wsd_stable_ratio=config.training.wsd_stable_ratio,
        wsd_decay_ratio=1.0 - config.training.wsd_stable_ratio,
        wsd_min_lr_ratio=config.training.wsd_min_lr_ratio,
    )

    start_epoch = 1
    best_val = float("inf")
    if checkpoint is not None:
        if checkpoint.get("run_digest") != digest:
            raise SystemExit(
                f"[Resume] run digest mismatch: checkpoint {checkpoint.get('run_digest')} "
                f"!= this run {digest}. Refusing to resume a different experiment."
            )
        if dist_info.enabled:
            from cadgenesis.training.fsdp import fsdp_full_state_dict

            fsdp_full_state_dict(trainer.model)
        trainer.load_checkpoint(args.resume_from)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_val = float(checkpoint.get("validation_loss", float("inf")))
        print(f"[Resume] digest verified: {digest} (resuming from epoch {start_epoch})")

    print(f"[Run] digest={digest}")
    print(
        f"[Run] data={data} ({len(train_records)} train / "
        f"{len(val_records)} val records) packed={args.packed} "
        f"fsdp={dist_info.enabled}"
    )
    print(f"[Model] {sum(p.numel() for p in model.parameters()):,} parameters")
    print(
        f"[Config] d_model={config.model.d_model} "
        f"enc={config.model.num_encoder_layers} "
        f"dec={config.model.num_decoder_layers} "
        f"epochs={config.training.max_epochs} "
        f"batch={config.training.batch_size} lr={config.training.lr} "
        f"schedule={config.training.schedule} "
        f"mp={config.training.mixed_precision} "
        f"mu_transfer={args.mu_transfer}"
    )

    def _save_ckpt(path: str, epoch: int, step: int, val_loss: float) -> None:
        """FSDP-safe checkpoint: state collectives on ALL ranks, I/O on rank 0."""
        if dist_info.enabled:
            from cadgenesis.training.fsdp import fsdp_full_state_dict

            fsdp_full_state_dict(trainer.model)
        trainer.save_checkpoint(
            str(path),
            epoch=epoch,
            step=step,
            validation_loss=val_loss,
            include_optimizer=not dist_info.enabled,
            write=dist_info.primary,
            **extra,
        )

    for epoch in range(start_epoch, config.training.max_epochs + 1):
        # Deterministic per-epoch shuffling AND dropout: ``(seed, epoch)`` so
        # a resumed run replays the exact same batches and masks as an
        # uninterrupted one.  With FSDP the DistributedSampler shards per rank
        # using the same ``(seed, epoch)`` so resume stays byte-exact.
        torch.manual_seed(args.seed * 10_000 + epoch)
        train_gen.manual_seed(args.seed * 10_000 + epoch)
        if train_dl.sampler is not None and hasattr(train_dl.sampler, "set_epoch"):
            train_dl.sampler.set_epoch(epoch)
        train_loss = (
            trainer.train_packed_epoch(train_dl) if args.packed else trainer.train_epoch(train_dl)
        )
        if val_records:
            val_loss = trainer.validate_packed(val_dl) if args.packed else trainer.validate(val_dl)
        else:
            # Tiny dataset: nothing left after the train split.
            val_loss = train_loss
        step = epoch * len(train_dl)
        extra = dict(
            run_digest=digest,
            seed=args.seed,
            dataset_sha256=data_hash,
            schedule=args.schedule,
            packed=args.packed,
            vocab_tokens=vocab_tokens(tokenizer),
        )
        checkpoint_path = out_dir / f"checkpoint_epoch_{epoch}.pt"
        _save_ckpt(checkpoint_path, epoch, step, val_loss)
        is_best = val_loss < best_val
        best_val = min(best_val, val_loss)
        if is_best:
            _save_ckpt(out_dir / "best_checkpoint.pt", epoch, step, val_loss)
        print(
            f"epoch {epoch:2d}/{config.training.max_epochs} "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"lr={trainer.optimizer.param_groups[0]['lr']:.2e} "
            f"{'[best]' if is_best else ''} -> {checkpoint_path}"
        )

    return {
        "digest": digest,
        "out_dir": str(out_dir),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val": best_val,
        "checkpoints": [
            str(out_dir / f"checkpoint_epoch_{e}.pt")
            for e in range(start_epoch, config.training.max_epochs + 1)
        ],
        "best_checkpoint": str(out_dir / "best_checkpoint.pt"),
    }


def main() -> None:
    result = run_training(parse_args())
    print("=" * 60)
    print(f"training complete: digest={result['digest']} best_val={result['best_val']:.4f}")
    print(f"checkpoints in {result['out_dir']}")


if __name__ == "__main__":
    main()
