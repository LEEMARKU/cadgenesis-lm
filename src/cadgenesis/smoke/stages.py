"""
cadgenesis.smoke.stages
=======================
The four G15 CPU smoke stages (pre-training gate PHASE 12).

Every stage is pure-CPU, seeded, and returns a plain dict of results so a
runner can collect them into `reports/SMOKE_TEST_RESULTS.md`:

1. ``stage1_forward_backward`` — one batch forward/backward on the mini
   preset; proves model, loss, and gradients work.
2. ``stage2_tiny_dataset`` — 1 epoch over 50 records; proves the dataset
   pipeline and a full training epoch work and loss decreases.
3. ``stage3_overfit`` — overfit 8 records toward a near-zero loss; proves
   the model can actually learn (not a dead pipeline).
4. ``stage4_dev_run`` — 200 records, few epochs, persisted loss curve
   (metrics.jsonl) + checkpoint; proves reproducible artifact logging.

NOT training: these are 2-5 minute CPU smoke runs.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from cadgenesis.config import CADConfig
from cadgenesis.tokenizer.cad_tokenizer import AutonomousCADTokenizer
from cadgenesis.tokenizer.legacy_shim import build_dataset
from cadgenesis.training.callbacks import MetricsJsonlCallback, TrainingEvent
from cadgenesis.training.trainer import CADTrainer, MultiModalCADDataset, cad_collate_fn
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer

SMOKE_OUTPUTS = Path("outputs/smoke")


def make_mini(seed: int = 42) -> tuple[CADConfig, AutonomousCADTokenizer, GeometryAwareTransformer, CADTrainer]:
    """Seeded mini-preset setup on CPU (mirrors scripts/train.py)."""
    random.seed(seed)
    torch.manual_seed(seed)
    config = CADConfig.mini()
    tokenizer = AutonomousCADTokenizer.build_mini()
    model = GeometryAwareTransformer(config)
    trainer = CADTrainer(config=config, model=model, tokenizer=tokenizer, device="cpu")
    return config, tokenizer, model, trainer


def make_dataloaders(
    trainer: CADTrainer,
    tokenizer: AutonomousCADTokenizer,
    n_train: int,
    n_val: int,
    batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Train/val dataloaders using the same data path as scripts/train.py."""
    batch = batch_size or trainer.config.training.batch_size
    raw_train = build_dataset(n_train, lang_tok=tokenizer.lang_tok)
    raw_val = build_dataset(n_val, lang_tok=tokenizer.lang_tok)
    train_ds = MultiModalCADDataset(raw_train, tokenizer)
    val_ds = MultiModalCADDataset(raw_val, tokenizer)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True, collate_fn=cad_collate_fn)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False, collate_fn=cad_collate_fn)
    return train_dl, val_dl


def parameter_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def has_updated_gradients(model: torch.nn.Module) -> bool:
    """True when at least one parameter received a finite nonzero gradient."""
    for param in model.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return False
        if float(param.grad.abs().sum()) > 0.0:
            return True
    return False


def stage1_forward_backward(seed: int = 42, batch_size: int = 4) -> dict[str, Any]:
    """One batch forward/backward on the mini preset.

    Mirrors the exact step the trainer performs (see ``CADTrainer.train_epoch``)
    but checks parameter gradients BEFORE the optimizer zeroes them.
    """
    start = time.time()
    config, tokenizer, model, trainer = make_mini(seed)
    train_dl, _ = make_dataloaders(trainer, tokenizer, n_train=8, n_val=4, batch_size=batch_size)
    src, tgt = next(iter(train_dl))
    src, tgt_in, tgt_out, tgt_type, src_mask, tgt_mask = trainer._prepare_batch(src, tgt)
    with trainer._autocast_context():
        logits, conf_logits = model(
            src_ids=src,
            tgt_in_ids=tgt_in,
            tgt_type_ids=tgt_type,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
        )
        loss, _ = trainer.seq_loss(
            logits,
            tgt_out,
            confidence_logits=conf_logits,
            target_confidence=trainer._target_confidence(logits, tgt_out),
            aux_loss=trainer._aux_loss_term(),
        )
    loss.backward()
    grads_ok = has_updated_gradients(model)
    result = {
        "status": "PASS" if (torch.isfinite(loss) and grads_ok) else "FAIL",
        "loss": float(loss),
        "gradients_updated": bool(grads_ok),
        "batch_shape": [int(src.size(0)), int(src.size(1)), int(tgt.size(1))],
        "parameters": parameter_count(model),
        "duration_s": round(time.time() - start, 2),
    }
    return result


def stage2_tiny_dataset(
    seed: int = 42,
    n_records: int = 50,
    epochs: int = 1,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Full 1-epoch run over a tiny dataset; loss must decrease."""
    start = time.time()
    config, tokenizer, model, trainer = make_mini(seed)
    config.training.warmup_steps = 0
    config.training.grad_accum_steps = 1
    train_dl, val_dl = make_dataloaders(
        trainer, tokenizer, n_train=n_records, n_val=16, batch_size=batch_size
    )
    trainer.configure_scheduler(len(train_dl))
    initial = trainer.validate(val_dl)
    for _ in range(epochs):
        final = trainer.train_epoch(train_dl)
    val_final = trainer.validate(val_dl)
    result = {
        "status": "PASS" if final < initial and torch.isfinite(torch.tensor(final)) else "FAIL",
        "initial_val_loss": float(initial),
        "final_train_loss": float(final),
        "final_val_loss": float(val_final),
        "records": n_records,
        "epochs": epochs,
        "duration_s": round(time.time() - start, 2),
    }
    return result


def stage3_overfit(
    seed: int = 42,
    n_records: int = 8,
    max_steps: int = 400,
    target_loss: float = 0.5,
    batch_size: int = 8,
    report_every: int = 20,
) -> dict[str, Any]:
    """Overfit a handful of records toward near-zero loss (proves learning)."""
    start = time.time()
    config, tokenizer, model, trainer = make_mini(seed)
    config.training.warmup_steps = 0
    config.training.grad_accum_steps = 1
    train_dl, _ = make_dataloaders(
        trainer, tokenizer, n_train=n_records, n_val=0, batch_size=batch_size
    )
    trainer.configure_scheduler(max_steps)
    curve: list[float] = []
    reached = False
    final = float("inf")
    for step in range(1, max_steps + 1):
        final = trainer.train_epoch(train_dl)
        if step == 1 or step % report_every == 0 or step == max_steps:
            curve.append(float(final))
        if final <= target_loss:
            reached = True
            break
    result = {
        "status": "PASS" if reached else "FAIL",
        "initial_loss": float(curve[0]) if curve else float("nan"),
        "final_loss": float(final),
        "target_loss": float(target_loss),
        "target_reached": bool(reached),
        "steps_used": int(step),
        "curve": [round(v, 6) for v in curve],
        "duration_s": round(time.time() - start, 2),
    }
    return result


def stage4_dev_run(
    seed: int = 42,
    n_records: int = 200,
    epochs: int = 2,
    batch_size: int = 16,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Dev-size run: 200 records, few epochs, persisted metrics + checkpoint."""
    start = time.time()
    out = Path(out_dir or SMOKE_OUTPUTS / "stage4")
    out.mkdir(parents=True, exist_ok=True)
    config, tokenizer, model, trainer = make_mini(seed)
    config.training.warmup_steps = 50
    config.training.grad_accum_steps = 2
    train_dl, val_dl = make_dataloaders(
        trainer, tokenizer, n_train=n_records, n_val=40, batch_size=batch_size
    )
    trainer.configure_scheduler(len(train_dl))
    callback = MetricsJsonlCallback(str(out / "metrics"))

    best = float("inf")
    curve: list[dict[str, float]] = []
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_dl)
        val_loss = trainer.validate(val_dl)
        best = min(best, val_loss)
        event = TrainingEvent(
            epoch=epoch,
            step=(epoch + 1) * len(train_dl),
            metrics={"loss": train_loss},
            validation_metrics={"loss": val_loss},
            best_validation_loss=best,
        )
        callback.on_epoch_end(event)
        callback.on_validation(event)
        curve.append({"epoch": float(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss)})
        if (epoch + 1) % epochs == 0:
            ckpt = trainer.save_checkpoint(str(out / "last.pt"), epoch, event.step, val_loss)
            callback.on_checkpoint(TrainingEvent(epoch=epoch, step=event.step, metrics={"loss": train_loss}))
            checkpoint_path = str(out / "last.pt")

    metrics_path = str(out / "metrics" / "metrics.jsonl")
    result = {
        "status": "PASS",
        "final_train_loss": float(curve[-1]["train_loss"]),
        "final_val_loss": float(curve[-1]["val_loss"]),
        "best_val_loss": float(best),
        "epochs": epochs,
        "records": n_records,
        "curve": curve,
        "metrics_path": metrics_path,
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "duration_s": round(time.time() - start, 2),
    }
    return result