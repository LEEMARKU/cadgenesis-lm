from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cadgenesis.serving.quantization import (
    FP8Linear,
    INT4Linear,
    quantize_model,
    report_quantization,
)


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).norm() / b.detach().norm())


class TestFP8Linear:
    def test_from_linear_matches_original(self):
        torch.manual_seed(0)
        linear = nn.Linear(8, 16)
        linear.weight.data.uniform_(0.5, 1.5)
        linear.bias.data.uniform_(-0.5, 0.5)

        qlinear = FP8Linear.from_linear(linear)
        assert qlinear.weight_int8.dtype == torch.int8
        assert qlinear.weight_int8.shape == linear.weight.shape
        assert qlinear.scale.shape == (16,)

        x = torch.randn(4, 8)
        assert _relative_error(qlinear(x), linear(x)) < 1e-1

    def test_memory_bytes_smaller_than_fp32(self):
        linear = nn.Linear(8, 16)
        qlinear = FP8Linear.from_linear(linear)
        assert qlinear.memory_bytes() < qlinear.fp32_memory_bytes()

    def test_gradient_flows_through_dequantization(self):
        linear = nn.Linear(8, 16)
        qlinear = FP8Linear.from_linear(linear)
        x = torch.randn(4, 8, requires_grad=True)
        out = qlinear(x).sum()
        out.backward()
        assert x.grad is not None
        assert qlinear.scale.grad is not None
        assert qlinear.bias.grad is not None


class TestINT4Linear:
    def test_from_linear_matches_original(self):
        torch.manual_seed(1)
        linear = nn.Linear(16, 8)
        linear.weight.data.uniform_(-1.0, 1.0)
        linear.bias.data.uniform_(-0.5, 0.5)

        qlinear = INT4Linear.from_linear(linear, group_size=8)
        assert qlinear.weight_int8.dtype == torch.int8
        assert qlinear.weight_int8.shape == (8, 16)
        assert qlinear.scale.shape == (8, 2)
        assert qlinear.weight_int8.min().item() >= -7
        assert qlinear.weight_int8.max().item() <= 7

        x = torch.randn(4, 16)
        assert _relative_error(qlinear(x), linear(x)) < 1e-1

    def test_memory_bytes_smaller_than_fp32(self):
        linear = nn.Linear(16, 8)
        qlinear = INT4Linear.from_linear(linear, group_size=8)
        assert qlinear.memory_bytes() < qlinear.fp32_memory_bytes()

    def test_raises_when_in_features_not_divisible(self):
        linear = nn.Linear(15, 8)
        with pytest.raises(ValueError, match="divisible"):
            INT4Linear.from_linear(linear, group_size=8)
        with pytest.raises(ValueError, match="divisible"):
            INT4Linear(15, 8, group_size=8)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(16, 8)
        self.lin1 = nn.Linear(8, 16)
        self.lin2 = nn.Linear(16, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.lin1(self.emb(x)))
        return self.lin2(h)


class TestQuantizeModel:
    def test_fp8_replaces_all_linears(self):
        model = _TinyModel()
        quantized = quantize_model(model, mode="fp8")
        assert isinstance(quantized.lin1, FP8Linear)
        assert isinstance(quantized.lin2, FP8Linear)
        assert isinstance(quantized.emb, nn.Embedding)
        with torch.no_grad():
            out = quantized(torch.randint(0, 16, (2, 4)))
        assert out.shape == (2, 4, 8)

    def test_int4_replaces_all_linears(self):
        model = _TinyModel()
        quantized = quantize_model(model, mode="int4", group_size=8)
        assert isinstance(quantized.lin1, INT4Linear)
        assert isinstance(quantized.lin2, INT4Linear)
        assert isinstance(quantized.emb, nn.Embedding)
        with torch.no_grad():
            out = quantized(torch.randint(0, 16, (2, 4)))
        assert out.shape == (2, 4, 8)

    def test_none_returns_same_object(self):
        model = _TinyModel()
        assert quantize_model(model, mode="none") is model

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            quantize_model(_TinyModel(), mode="fp16")

    def test_report_quantization(self):
        fp32_model = _TinyModel()
        quantized = _TinyModel()
        quantize_model(quantized, mode="fp8")

        report = report_quantization(quantized, fp32_model)
        assert report["n_linears"] == 2
        assert report["fp32_bytes"] > 0
        assert report["quant_bytes"] < report["fp32_bytes"]
        assert report["ratio"] < 1.0
        assert report["ratio"] > 0.0
