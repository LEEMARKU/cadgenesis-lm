"""cadgenesis.optimization.onnx
=============================
ONNX export and optimization of the inference graph.

Provides export utilities and passes to convert CADGenesis-LM models to
ONNX format, with simplifications for deployment targeting ONNX Runtime,
TensorRT, and other inference engines.
"""

from __future__ import annotations

from typing import Any

import torch


def export_model(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    simplify: bool = True,
) -> Any:
    """Export a model to ONNX format.

    - ``model``: PyTorch model to export.
    - ``example_input``: Example input tensor for tracing.
    - ``input_names``: Names for model inputs.
    - ``output_names``: Names for model outputs.
    - ``dynamic_axes``: Dynamic axis specifications.
    - ``simplify``: Whether to apply ONNX simplification pass.

    Returns the ONNX model as a ``torch.Tensor`` (for further processing)
    or writes to ``model.onnx`` if ``output_path`` is specified.
    """
    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]
    if dynamic_axes is None:
        dynamic_axes = {"input": {0: "batch", 1: "sequence"}, "output": {0: "batch", 1: "sequence"}}

    exported = torch.onnx.export(
        model,
        (example_input,),
        "cadgenesis.onnx",
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=21,
    )

    if simplify:
        # NOTE: ONNX simplification requires the onnx package;
        # see https://github.com/onnx/onnx
        try:
            import onnx
            from onnx import simplification

            onnx_model = onnx.load("cadgenesis.onnx")
            onnx_model = simplification.simplify(onnx_model)
            onnx.save_model(onnx_model, "cadgenesis.onnx")
        except ImportError:
            pass  # simplification not available; keep original export

    if exported is not None:
        return exported
    return example_input.new_empty(0)


__all__ = [
    "export_model",
]
