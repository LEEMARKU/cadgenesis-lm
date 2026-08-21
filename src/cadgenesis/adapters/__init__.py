"""
cadgenesis.adapters
===================
Adapter Framework for CADGenesis-LM v2.0.

Consolidates the former ``cadgenesis.adapters`` (LoRA/QLoRA) and
``cadgenesis.adapters`` (self-evolving adapter bank) subsystems under one
architecture.  Provides:

- ``lora`` — LoRA and QLoRA PEFT modules.
- ``manager`` — the self-evolving adapter bank (promotion / rollback / lifecycle).
- ``peft``, ``qlora``, ``router``, ``lifecycle``, ``promotion``, ``rollback``,
  ``versioning`` — extension points.
"""

from cadgenesis.adapters.deepseek_r1 import (
    DEFAULT_MODEL_ID,
    DeepSeekR1DataGenerator,
    DeepSeekR1Reasoner,
    DeepSeekR1Teacher,
    MockDeepSeekR1Teacher,
)
from cadgenesis.adapters.lifecycle import (
    ALLOWED_TRANSITIONS,
    AdapterLifecycle,
    AdapterLifecycleState,
    LifecycleEvent,
)
from cadgenesis.adapters.lora import LoRALinear, apply_lora
from cadgenesis.adapters.manager import AdapterMetadata, SelfEvolvingAdapterBank
from cadgenesis.adapters.peft import PEFTAdapter
from cadgenesis.adapters.promotion import (
    PROMOTED_STATUS,
    AdapterPromotion,
    PromotionCriteria,
    PromotionDecision,
)
from cadgenesis.adapters.qlora import QLoRAAdapter, QuantizedLinear, QuantizedModel
from cadgenesis.adapters.rollback import AdapterRollback
from cadgenesis.adapters.router import AdapterRouter, RoutingDecision
from cadgenesis.adapters.versioning import AdapterVersion, AdapterVersionRegistry

__all__ = [
    "ALLOWED_TRANSITIONS",
    "DEFAULT_MODEL_ID",
    "PROMOTED_STATUS",
    "AdapterLifecycle",
    "AdapterLifecycleState",
    "AdapterMetadata",
    "AdapterPromotion",
    "AdapterRollback",
    "AdapterRouter",
    "AdapterVersion",
    "AdapterVersionRegistry",
    "DeepSeekR1DataGenerator",
    "DeepSeekR1Reasoner",
    "DeepSeekR1Teacher",
    "LifecycleEvent",
    "LoRALinear",
    "MockDeepSeekR1Teacher",
    "PEFTAdapter",
    "PromotionCriteria",
    "PromotionDecision",
    "QLoRAAdapter",
    "QuantizedLinear",
    "QuantizedModel",
    "RoutingDecision",
    "SelfEvolvingAdapterBank",
    "apply_lora",
]
