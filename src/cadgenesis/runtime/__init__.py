"""
cadgenesis.runtime
==================
HardwareAwareRuntime (v6.2): device presets, memory planning and live
benchmarks.  All model/batch sizing decisions that depend on the machine's
GPU/CPU should consult this package instead of hardcoding constants.
"""

from cadgenesis.runtime.benchmarks import (
    DecodeBenchmark,
    ForwardBenchmark,
    benchmark_decode,
    benchmark_forward,
)
from cadgenesis.runtime.hardware import (
    PRESETS,
    RuntimePreset,
    clamp_to_preset,
    detect_device,
    select_preset,
)
from cadgenesis.runtime.memory_planner import (
    MemoryEstimate,
    RuntimeRecommendation,
    estimate_training_memory,
    fits,
    recommend_config_overrides,
)

__all__ = [
    "PRESETS",
    "DecodeBenchmark",
    "ForwardBenchmark",
    "MemoryEstimate",
    "RuntimePreset",
    "RuntimeRecommendation",
    "benchmark_decode",
    "benchmark_forward",
    "clamp_to_preset",
    "detect_device",
    "estimate_training_memory",
    "fits",
    "recommend_config_overrides",
    "select_preset",
]
