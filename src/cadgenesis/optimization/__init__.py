"""cadgenesis.optimization
=======================
Inference-time and model-level optimizations (quantization, pruning, ONNX, kernel selection).
"""

from cadgenesis.optimization.kernels import FusedAttention, MoEKernel
from cadgenesis.optimization.onnx import export_model
from cadgenesis.optimization.pruning import magnitude_unstructured, structured_head_pruning
from cadgenesis.optimization.quantization import QuantizedLinear, quantize_model_qt

__all__ = [
    "FusedAttention",
    "MoEKernel",
    "QuantizedLinear",
    "export_model",
    "magnitude_unstructured",
    "quantize_model_qt",
    "structured_head_pruning",
]
