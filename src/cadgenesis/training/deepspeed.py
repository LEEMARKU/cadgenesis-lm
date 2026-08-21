"""
cadgenesis.training.deepspeed
=============================
DeepSpeed integration for CADGenesis-LM.

``build_deepspeed_engine`` configures a ZeRO stage / offload setup and
initializes the DeepSpeed engine around a model+optimizer; ``deepspeed_ckpt``
saves/loads the engine state.  The module is a no-op returning ``None`` when
DeepSpeed is not installed.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    import deepspeed  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - deepspeed optional
    deepspeed = None  # type: ignore[assignment]

logger = logging.getLogger("cadgenesis.training.deepspeed")


def _make_deepspeed_config(
    zero_stage: int,
    offload_optimizer: bool,
    offload_params: bool,
    batch_size: int,
    micro_batch_size: int,
    grad_accum_steps: int,
    fp16: bool,
    bf16: bool,
) -> dict[str, Any]:
    """Build a DeepSpeed config dict for ``deepspeed.initialize``."""
    zero: dict[str, Any] = {"stage": zero_stage, "stage3_gather_16bit_weights_on_model_save": True}
    if zero_stage >= 3 and offload_params:
        zero["offload_param"] = {"device": "cpu", "pin_memory": True}
    if offload_optimizer:
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    config: dict[str, Any] = {
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "zero_optimization": zero,
        "gradient_clipping": 1.0,
        "steps_per_print": 50,
    }
    if bf16:
        config["bf16"] = {"enabled": True}
    elif fp16:
        config["fp16"] = {"enabled": True}
    return config


def build_deepspeed_engine(
    model: Any,
    optimizer: Any | None = None,
    zero_stage: int = 2,
    offload_optimizer: bool = False,
    offload_params: bool = False,
    batch_size: int = 32,
    micro_batch_size: int = 4,
    grad_accum_steps: int = 1,
    fp16: bool = False,
    bf16: bool = False,
    config_path: str | None = None,
) -> tuple[Any, Any, Any, Any] | None:
    """Initialize DeepSpeed; returns ``(engine, optimizer, dataloader, lr_scheduler)``.

    Returns ``None`` when DeepSpeed is unavailable so callers can fall back
    to plain PyTorch training.
    """
    if deepspeed is None:  # pragma: no cover - deepspeed optional
        logger.info("DeepSpeed not installed; skipping engine init")
        return None
    ds_config: dict[str, Any] | str = _make_deepspeed_config(
        zero_stage=zero_stage,
        offload_optimizer=offload_optimizer,
        offload_params=offload_params,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        grad_accum_steps=grad_accum_steps,
        fp16=fp16,
        bf16=bf16,
    )
    if config_path:
        ds_config = config_path
    engine, ds_optimizer, ds_dataloader, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )
    return engine, ds_optimizer, ds_dataloader, lr_scheduler


def deepspeed_ckpt(
    engine: Any,
    ckpt_dir: str,
    tag: str,
    save: bool,
) -> str:
    """Save or load a DeepSpeed engine checkpoint under ``ckpt_dir/tag``."""
    if deepspeed is None:
        raise RuntimeError("deepspeed_ckpt requires DeepSpeed")
    path = f"{ckpt_dir}/{tag}"
    if save:
        engine.save_checkpoint(ckpt_dir, tag=tag)
    else:
        _, _ = engine.load_checkpoint(ckpt_dir, tag=tag)
    return path


__all__ = ["build_deepspeed_engine", "deepspeed_ckpt"]
