"""
cadgenesis.training.trainer
===========================
Production Multi-Modal CAD Trainer for CADGenesis-LM v2.0.

Features:
- Multi-modal paired CAD dataset handling
- Cross-entropy loss with padding ignore
- Confidence / Uncertainty weighted loss
- Mixed precision (bf16 / fp16) support
- Cosine Annealing with Warmup scheduler (plus WSD via ``training.scheduler``)
- Sequence packing (block-diagonal masked rows) for token-efficient training
- Optional DDP / FSDP wrapping when a torch.distributed group is initialised
- Checkpoint saving & loading
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
from typing import Any, cast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cadgenesis.config import CADConfig
from cadgenesis.tokenizer.cad_tokenizer import AutonomousCADTokenizer
from cadgenesis.training.fsdp import FSDPConfig
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer

logger = logging.getLogger(__name__)


class MultiModalCADDataset(Dataset):
    """Dataset wrapper for multi-modal (text -> CAD tokens) pairs."""

    def __init__(self, pairs: list[tuple[str, list[int]]], tokenizer: AutonomousCADTokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        text, cad_ids = self.pairs[idx]
        src_ids = self.tokenizer.encode_text(text)
        return src_ids, cad_ids


def cad_collate_fn(batch, pad_id=0, max_src_len=64, max_tgt_len=32):
    B = len(batch)
    src_tensor = torch.full((B, max_src_len), pad_id, dtype=torch.long)
    tgt_tensor = torch.full((B, max_tgt_len), pad_id, dtype=torch.long)

    for i, (src, tgt) in enumerate(batch):
        src_len = min(len(src), max_src_len)
        tgt_len = min(len(tgt), max_tgt_len)
        src_tensor[i, :src_len] = torch.tensor(src[:src_len], dtype=torch.long)
        tgt_tensor[i, :tgt_len] = torch.tensor(tgt[:tgt_len], dtype=torch.long)

    return src_tensor, tgt_tensor


def packed_collate_fn(
    batch,
    bos_id: int,
    eos_id: int,
    pad_id: int = 0,
    max_src_len: int = 256,
    max_tgt_len: int = 128,
    seed: int | None = None,
):
    """
    Collate a list of ``(src_ids, tgt_ids)`` samples into *packed* rows using
    :func:`cadgenesis.training.packing.pack_batch`.  Returns the packed dict
    consumed by :meth:`CADTrainer.train_packed_epoch`.
    """
    from cadgenesis.training.packing import pack_batch

    return pack_batch(
        batch,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        seed=seed,
    )


class CADTrainer:
    """
    Production Trainer for CADGenesis-LM v2.0.
    """

    def __init__(
        self,
        config: CADConfig,
        model: GeometryAwareTransformer,
        tokenizer: AutonomousCADTokenizer,
        device: str | None = None,
        use_fsdp: bool = False,
        use_ddp: bool = False,
        fsdp_config: FSDPConfig | None = None,
        optimizer: str = "adamw",
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = self._normalize_device(device)
        self._apply_runtime_preset()

        # Distributed wrapping (DDP / FSDP).  Both no-op when no
        # torch.distributed group is initialised, so single-process and
        # CPU-only runs are unaffected.
        self.use_fsdp = use_fsdp
        self.use_ddp = use_ddp
        if self.use_fsdp or self.use_ddp:
            self._maybe_wrap_distributed(fsdp_config)
        self.model.to(self.device)

        from cadgenesis.training.optimizer import build_optimizer

        self.optimizer = build_optimizer(
            self.model,
            optimizer_type=optimizer,
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        from cadgenesis.transformer.losses import CADSequenceLoss

        self.seq_loss = CADSequenceLoss(
            pad_id=self.tokenizer.pad_id,
            label_smoothing=config.training.label_smoothing,
            confidence_weight=config.training.confidence_loss_weight,
            moe_aux_scale=config.training.moe_aux_scale,
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)

        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        self._build_type_id_map()

        self.use_amp = self.config.training.mixed_precision in ("fp16", "bf16") and (
            self.device.startswith("cuda")
            or (self.device.startswith("cpu") and self.config.training.mixed_precision == "bf16")
        )
        self.autocast_dtype = None
        self.scaler = None
        if self.use_amp:
            self.autocast_dtype = (
                torch.float16 if self.config.training.mixed_precision == "fp16" else torch.bfloat16
            )
            # fp16 needs a grad scaler; bf16 (CUDA or CPU) does not.
            if self.config.training.mixed_precision == "fp16":
                self.scaler = torch.cuda.amp.GradScaler()

    def _apply_runtime_preset(self) -> None:
        """
        HardwareAwareRuntime (v6.2): resolve ``config.runtime.preset`` and,
        when ``enforce_preset`` is set, clamp training batch/seq and enable
        gradient checkpointing to fit the device budget.

        Non-enforcing mode only *reports* the resolution (no silent config
        mutation); the preset is still stored on ``self.runtime_preset`` for
        callers (CLI, benchmarks) to read.
        """
        from cadgenesis.runtime.hardware import select_preset

        self.runtime_preset = select_preset(self.config.runtime.preset)
        if not self.config.runtime.enforce_preset:
            return

        t = self.config.training
        t.batch_size = min(t.batch_size, self.runtime_preset.max_train_batch)
        t.gradient_checkpointing = (
            t.gradient_checkpointing or self.runtime_preset.grad_checkpointing
        )
        if self.runtime_preset.supports_bf16 and t.mixed_precision == "fp16":
            print(
                "[Warning] runtime preset supports bf16; consider "
                "training.mixed_precision='bf16'."
            )
        if (
            not self.runtime_preset.supports_bf16
            and t.mixed_precision == "bf16"
            and self.device.startswith("cuda")
        ):
            print(
                "[Warning] GPU compute capability < 8.0 cannot use bf16 "
                "efficiently; consider training.mixed_precision='fp16'."
            )

    def _normalize_device(self, device: str | None) -> str:
        if device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"

        normalized = device.lower()
        if normalized == "cuda" and not torch.cuda.is_available():
            print("[Warning] CUDA requested but unavailable; falling back to CPU.")
            return "cpu"
        if normalized == "mps" and not getattr(torch, "has_mps", lambda: False)():
            print("[Warning] MPS requested but unavailable; falling back to CPU.")
            return "cpu"
        return normalized

    def _maybe_wrap_distributed(self, fsdp_config: FSDPConfig | None) -> None:
        """
        Wrap ``self.model`` with FSDP or DDP when a torch.distributed group is
        initialised with world_size > 1; otherwise leave the model untouched.
        """
        try:
            import torch.distributed as dist
        except ImportError:
            return
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        if self.use_fsdp:
            from cadgenesis.training.fsdp import wrap_fsdp

            self.model = wrap_fsdp(self.model, fsdp_config)
        elif self.use_ddp:
            self.model = cast(
                GeometryAwareTransformer,
                torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,
                ),
            )

    def _build_type_id_map(self) -> None:
        records = list(self.tokenizer.vocab)
        max_id = max((r.token_id for r in records), default=0)
        table = torch.zeros(max_id + 1, dtype=torch.long)
        for record in records:
            table[record.token_id] = record.type_id
        self._type_table = table

    def _map_type_ids(self, tgt_in: torch.Tensor) -> torch.Tensor:
        """Vectorised token-id → type-id mapping (O(T) instead of O(V·T))."""
        table = self._type_table.to(tgt_in.device)
        return table[tgt_in]

    def configure_scheduler(self, num_train_steps: int) -> None:
        if num_train_steps <= 0:
            return

        effective_steps = math.ceil(num_train_steps / max(1, self.config.training.grad_accum_steps))
        total_steps = effective_steps * self.config.training.max_epochs
        warmup = self.config.training.warmup_steps

        def lr_lambda(step: int) -> float:
            if total_steps <= 0:
                return 1.0
            if step < warmup:
                return float(step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _map_type_ids(self, tgt_in: torch.Tensor) -> torch.Tensor:
        """Vectorised token-id → type-id mapping (O(T) instead of O(V·T))."""
        table = self._type_table.to(tgt_in.device)
        return table[tgt_in]

    def _aux_loss_term(self) -> torch.Tensor | None:
        """MoE auxiliary loss summed over all blocks (None when no MoE)."""
        fn = self._model_attr("aux_loss")
        if callable(fn):
            return fn()
        return None

    def _reward_term(self) -> torch.Tensor:
        """RLAIF reward-maximisation term (zero when disabled)."""
        reward = self._model_attr("last_reward")
        weight = float(getattr(self.config.training, "rlaf_reward_weight", 0.0))
        if reward is not None and weight > 0:
            return -weight * reward.mean()
        return torch.tensor(0.0, device=self.device)

    def _target_confidence(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Per-token correctness signal for the confidence head (detached)."""
        return (logits.detach().argmax(dim=-1) == targets).float()

    def _autocast_context(self):
        if self.use_amp and self.autocast_dtype is not None:
            return torch.autocast(device_type=self.device, dtype=self.autocast_dtype)
        return contextlib.nullcontext()

    def _prepare_batch(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_type = self._map_type_ids(tgt_in)
        src_mask = src == self.tokenizer.pad_id
        tgt_mask = tgt_in == self.tokenizer.pad_id
        return src, tgt_in, tgt_out, tgt_type, src_mask, tgt_mask

    def _model_attr(self, name: str):
        """Attribute access that works through DDP/FSDP wrappers."""
        model = getattr(self.model, "module", self.model)
        return getattr(model, name, None)

    def _mtp_term(self, hidden: torch.Tensor, tgt_out: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Optional multi-token-prediction loss term.  Returns
        ``(0.0, {})`` when the MTP head is disabled.
        """
        mtp_head = self._model_attr("mtp_head")
        if mtp_head is None:
            return torch.tensor(0.0, device=self.device), {}
        mtp_loss, mtp_breakdown = self._model_attr("mtp_loss")(hidden, tgt_out)
        if mtp_loss is None:
            return torch.tensor(0.0, device=self.device), {}
        weight = float(getattr(self.config.model, "mtp_weight", 0.1))
        return weight * mtp_loss, {f"mtp_{k}": v for k, v in mtp_breakdown.items()}

    def _prepare_packed_batch(self, batch: dict):
        """
        Move a packed batch (from :func:`packed_collate_fn`) onto the device
        and split the decoder targets into input/output slices.
        """
        src = batch["src"].to(self.device)
        tgt = batch["tgt"].to(self.device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        tgt_type = self._map_type_ids(tgt_in)
        loss_mask = batch["loss_mask"].to(self.device)
        src_pad = batch["src_pad_mask"].to(self.device)
        src_attn_mask = batch["src_attn_mask"].to(self.device)
        # Masks are defined over the *full* decoder row; the model sees
        # ``tgt_in`` (one position shorter), so trim the trailing row/col.
        tgt_attn_mask = batch["tgt_attn_mask"][:, :, :-1, :-1].to(self.device)
        cross_attn_mask = batch["cross_attn_mask"][:, :, :-1, :].to(self.device)
        return (
            src,
            tgt_in,
            tgt_out,
            tgt_type,
            loss_mask,
            src_pad,
            src_attn_mask,
            tgt_attn_mask,
            cross_attn_mask,
        )

    def _packed_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        (
            src,
            tgt_in,
            tgt_out,
            tgt_type,
            loss_mask,
            src_pad,
            src_attn_mask,
            tgt_attn_mask,
            cross_attn_mask,
        ) = self._prepare_packed_batch(batch)

        with self._autocast_context():
            mtp_enabled = self._model_attr("mtp_head") is not None
            if mtp_enabled:
                logits, conf_logits, hidden = self.model(
                    src_ids=src,
                    tgt_in_ids=tgt_in,
                    tgt_type_ids=tgt_type,
                    src_key_padding_mask=src_pad,
                    src_attn_mask=src_attn_mask,
                    tgt_attn_mask=tgt_attn_mask,
                    cross_attn_mask=cross_attn_mask,
                    return_hidden=True,
                )
            else:
                logits, conf_logits = self.model(
                    src_ids=src,
                    tgt_in_ids=tgt_in,
                    tgt_type_ids=tgt_type,
                    src_key_padding_mask=src_pad,
                    src_attn_mask=src_attn_mask,
                    tgt_attn_mask=tgt_attn_mask,
                    cross_attn_mask=cross_attn_mask,
                )
                hidden = None
            if hasattr(self, "seq_loss"):
                total, breakdown = self.seq_loss(
                    logits,
                    tgt_out,
                    confidence_logits=conf_logits,
                    target_confidence=self._target_confidence(logits, tgt_out),
                    mask=loss_mask,
                    aux_loss=self._aux_loss_term(),
                )
                total = total + self._reward_term()
            else:
                total = self.loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1),
                )
                breakdown = {"ce": float(total.item()), "total": float(total.item())}
            if mtp_enabled:
                mtp_term, mtp_bd = self._mtp_term(hidden, tgt_out)
                total = total + mtp_term
                breakdown.update(mtp_bd)
        return total, breakdown

    def train_packed_epoch(self, dataloader: DataLoader) -> float:
        """
        Token-efficient training epoch over *packed* sequences.

        The dataloader must use :func:`packed_collate_fn` (which returns the
        packed batch dict).  Loss is masked to real target tokens; padding and
        cross-sample positions are never scored.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()
        grad_accum = max(1, self.config.training.grad_accum_steps)
        last_step = len(dataloader) - 1

        for step, batch in enumerate(dataloader):
            total, _ = self._packed_loss(batch)
            loss_value = total.item()
            total_loss += loss_value
            total = total / grad_accum

            if self.scaler is not None:
                self.scaler.scale(total).backward()
            else:
                total.backward()

            if (step + 1) % grad_accum == 0 or step == last_step:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.max_grad_norm
                )
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()
            n_batches += 1

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def validate_packed(self, dataloader: DataLoader) -> float:
        """Validation loss over packed sequences (see :meth:`train_packed_epoch`)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            total, _ = self._packed_loss(batch)
            total_loss += total.item()
            n_batches += 1
        return total_loss / max(1, n_batches)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()
        grad_accum = max(1, self.config.training.grad_accum_steps)
        last_step = len(dataloader) - 1

        for step, (src, tgt) in enumerate(dataloader):
            src, tgt_in, tgt_out, tgt_type, src_mask, tgt_mask = self._prepare_batch(src, tgt)

            with self._autocast_context():
                mtp_enabled = self._model_attr("mtp_head") is not None
                out = self.model(
                    src_ids=src,
                    tgt_in_ids=tgt_in,
                    tgt_type_ids=tgt_type,
                    src_key_padding_mask=src_mask,
                    tgt_key_padding_mask=tgt_mask,
                    return_hidden=mtp_enabled,
                )
                logits, conf_logits = out[0], out[1]
                loss, _ = self.seq_loss(
                    logits,
                    tgt_out,
                    confidence_logits=conf_logits,
                    target_confidence=self._target_confidence(logits, tgt_out),
                    aux_loss=self._aux_loss_term(),
                )
                loss = loss + self._reward_term()
                if mtp_enabled:
                    mtp_term, _ = self._mtp_term(out[2], tgt_out)
                    loss = loss + mtp_term

            loss_value = loss.item()
            total_loss += loss_value
            loss = loss / grad_accum

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % grad_accum == 0 or step == last_step:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.max_grad_norm
                )
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()

        return total_loss / len(dataloader)

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0

        for src, tgt in dataloader:
            src, tgt_in, tgt_out, tgt_type, _src_mask, _tgt_mask = self._prepare_batch(src, tgt)
            with self._autocast_context():
                logits, conf_logits = self.model(
                    src_ids=src,
                    tgt_in_ids=tgt_in,
                    tgt_type_ids=tgt_type,
                )
                loss, _ = self.seq_loss(
                    logits,
                    tgt_out,
                    confidence_logits=conf_logits,
                    target_confidence=self._target_confidence(logits, tgt_out),
                    aux_loss=self._aux_loss_term(),
                )
                loss = loss + self._reward_term()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        step: int,
        validation_loss: float | None = None,
        include_optimizer: bool = True,
        write: bool = True,
        **extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Build (and, when ``write``, persist) a checkpoint.

        FSDP notes: with ``include_optimizer=False`` the optimizer state is
        omitted (FSDP shards it across ranks; only the model weights are
        checkpointed, the optimizer restarts fresh on resume — the WSD
        scheduler is rebuilt from the horizon anyway).  Set the model to
        ``FULL_STATE_DICT`` on every rank before calling so the collective
        ``state_dict()`` runs everywhere while only the primary rank writes.
        """
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        state: dict[str, Any] = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "config": self.config.to_dict(),
        }
        if include_optimizer:
            state["optimizer_state_dict"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()
        if validation_loss is not None:
            state["validation_loss"] = float(validation_loss)
        state.update(extra)
        if write:
            torch.save(state, path)
            logger.info("Saved checkpoint to %s", path)
        return state

    def load_checkpoint(self, path: str) -> dict:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        return checkpoint
