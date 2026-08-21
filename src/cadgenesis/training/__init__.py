"""
cadgenesis.training
==================
Training framework for CADGenesis-LM v2.0 foundation model.

Includes the core trainer (``CADTrainer``), learning-rate schedulers,
optimizer factory, metrics tracking, callback registry, checkpointing,
distributed launchers (DDP / FSDP / DeepSpeed) and a training profiler.
"""

from cadgenesis.training.callbacks import (
    CallbackRegistry,
    CheckpointCallback,
    EarlyStoppingCallback,
    MetricsLoggingCallback,
    StopTraining,
    TrainerCallback,
    TrainingEvent,
)
from cadgenesis.training.checkpoint import (
    META_FILE,
    CheckpointManager,
    cleanup_checkpoints,
    move_checkpoint,
)
from cadgenesis.training.metrics import MetricsTracker, compute_accuracy, log_summary
from cadgenesis.training.optimizer import OPTIMIZERS, build_optimizer, lora_param_groups
from cadgenesis.training.packing import pack_batch
from cadgenesis.training.rlvr_pipeline import RLVRPipeline
from cadgenesis.training.scheduler import SCHEDULES, build_scheduler, build_wsd_scheduler
from cadgenesis.training.trainer import (
    CADTrainer,
    MultiModalCADDataset,
    cad_collate_fn,
    packed_collate_fn,
)

__all__ = [
    "META_FILE",
    "OPTIMIZERS",
    "SCHEDULES",
    "CADTrainer",
    "CallbackRegistry",
    "CheckpointCallback",
    "CheckpointManager",
    "EarlyStoppingCallback",
    "MetricsLoggingCallback",
    "MetricsTracker",
    "MultiModalCADDataset",
    "RLVRPipeline",
    "StopTraining",
    "TrainerCallback",
    "TrainingEvent",
    "build_optimizer",
    "build_scheduler",
    "build_wsd_scheduler",
    "cad_collate_fn",
    "cleanup_checkpoints",
    "compute_accuracy",
    "log_summary",
    "lora_param_groups",
    "move_checkpoint",
    "pack_batch",
    "packed_collate_fn",
]
