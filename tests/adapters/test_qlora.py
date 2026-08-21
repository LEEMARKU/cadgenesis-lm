"""Tests for cadgenesis.adapters.qlora."""

import pytest
import torch
import torch.nn as nn

from cadgenesis.adapters.lora import LoRALinear
from cadgenesis.adapters.qlora import QLoRAAdapter, QuantizedLinear, QuantizedModel


def make_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))


def fill_lora(wrapper: LoRALinear, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        wrapper.lora_A.copy_(torch.randn(wrapper.lora_A.shape, generator=generator) * 0.1)
        wrapper.lora_B.copy_(torch.randn(wrapper.lora_B.shape, generator=generator) * 0.1)


def test_quantize_replaces_linears():
    model = make_model()
    qmodel = QLoRAAdapter().quantize(model)
    assert isinstance(qmodel, QuantizedModel)
    assert isinstance(qmodel.model[0], QuantizedLinear)
    assert isinstance(qmodel.model[1], QuantizedLinear)


def test_int8_weight_range():
    model = make_model()
    qmodel = QLoRAAdapter().quantize(model)
    int8_min = qmodel.model[0].weight_int8.min().item()
    int8_max = qmodel.model[0].weight_int8.max().item()
    assert -127 <= int8_min <= int8_max <= 127


def test_quantized_forward_close_to_fp32():
    model = make_model().eval()
    x = torch.randn(4, 16)
    reference = model(x)
    qmodel = QLoRAAdapter().quantize(model)
    output = qmodel(x)
    relative_error = (output - reference).abs().max() / reference.abs().max()
    assert relative_error.item() < 1e-1


def test_memory_report_sane():
    model = make_model()
    qmodel = QLoRAAdapter().quantize(model)
    report = qmodel.memory_report()
    assert report["original_bytes"] > 0
    assert report["quantized_bytes"] > 0
    assert report["quantized_bytes"] < report["original_bytes"]
    assert 0.0 < report["savings_pct"] < 100.0
    # Linear(16,16): weight 256*4 + bias 16*4 = 1088; quantized 256 + 64 + 64 = 384
    # Linear(16,8):  weight 128*4 + bias 8*4  = 544;  quantized 128 + 32 + 32 = 192
    assert report["original_bytes"] == 1088.0 + 544.0
    assert report["quantized_bytes"] == 384.0 + 192.0


def test_attach_lora_changes_output():
    model = make_model().eval()
    qa = QLoRAAdapter(rank=4, alpha=8.0, dropout=0.0)
    qmodel = qa.quantize(model)
    x = torch.randn(3, 16)
    base_output = qmodel(x)
    qa.attach_lora(qmodel, "qlora_v1", target_layers=["0"])
    fill_lora(qmodel.model[0], seed=5)
    lora_output = qmodel(x)
    assert not torch.allclose(lora_output, base_output, atol=1e-6)
    assert qa.attached == "qlora_v1"


def test_attach_lora_requires_quantized_model():
    with pytest.raises(ValueError, match="no QuantizedLinear"):
        QLoRAAdapter().attach_lora(QuantizedModel(nn.Linear(4, 4)), "x")


def test_merge_folds_quantization_and_lora():
    model = make_model().eval()
    qa = QLoRAAdapter(rank=4, alpha=8.0, dropout=0.0)
    qmodel = qa.quantize(model)
    x = torch.randn(3, 16)
    qa.attach_lora(qmodel, "qlora_v1", target_layers=["0"])
    fill_lora(qmodel.model[0], seed=7)
    with_lora = qmodel(x)
    merged = qa.merge(qmodel)
    assert isinstance(merged[0], nn.Linear)
    assert isinstance(merged[1], nn.Linear)
    assert torch.allclose(merged(x), with_lora, atol=1e-5)
    assert qa.attached is None


def test_merge_without_lora_matches_quantized_forward():
    model = make_model().eval()
    qa = QLoRAAdapter()
    qmodel = qa.quantize(model)
    x = torch.randn(3, 16)
    quantized_output = qmodel(x)
    merged = qa.merge(qmodel)
    assert torch.allclose(merged(x), quantized_output, atol=1e-6)
