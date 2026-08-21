"""Tests for cadgenesis.adapters.peft."""

import pytest
import torch
import torch.nn as nn

from cadgenesis.adapters.lora import LoRALinear
from cadgenesis.adapters.peft import PEFTAdapter


def make_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 4))


def fill_lora(wrapper: LoRALinear, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        wrapper.lora_A.copy_(torch.randn(wrapper.lora_A.shape, generator=generator) * 0.1)
        wrapper.lora_B.copy_(torch.randn(wrapper.lora_B.shape, generator=generator) * 0.1)


def test_attach_installs_wrappers_and_returns_model():
    model = make_model().eval()
    facade = PEFTAdapter()
    returned = facade.attach(model, "lora_a", r=4, alpha=8.0, dropout=0.0, target_layers=["0", "1"])
    assert returned is model
    assert isinstance(model[0], LoRALinear)
    assert isinstance(model[1], LoRALinear)
    assert facade.list_adapters() == ["lora_a"]
    assert facade.active == "lora_a"


def test_attach_no_targets_raises():
    model = make_model()
    facade = PEFTAdapter()
    with pytest.raises(ValueError, match="no Linear layers matched"):
        facade.attach(model, "lora_a", target_layers=["embeddings"])


def test_attach_duplicate_raises():
    model = make_model()
    facade = PEFTAdapter()
    facade.attach(model, "lora_a", target_layers=["0"])
    with pytest.raises(ValueError, match="already attached"):
        facade.attach(model, "lora_a", target_layers=["0"])


def test_attach_second_model_raises():
    facade = PEFTAdapter()
    facade.attach(make_model(), "lora_a", target_layers=["0"])
    with pytest.raises(ValueError, match="different model"):
        facade.attach(make_model(), "lora_b", target_layers=["0"])


def test_lora_forward_matches_manual_delta():
    model = make_model().eval()
    facade = PEFTAdapter()
    facade.attach(model, "lora_a", r=4, alpha=8.0, dropout=0.0, target_layers=["0"])
    fill_lora(model[0], seed=1)
    x = torch.randn(2, 8)
    wrapper = model[0]
    lora_out = (x @ wrapper.lora_A.T) @ wrapper.lora_B.T
    expected = wrapper.original_linear(x) + lora_out * wrapper.scaling
    assert torch.allclose(wrapper(x), expected, atol=1e-5)
    assert wrapper(x).shape == x.shape


def test_lora_changes_output():
    model = make_model().eval()
    facade = PEFTAdapter()
    base_out = model(torch.randn(3, 8))
    facade.attach(model, "lora_a", r=4, alpha=8.0, dropout=0.0, target_layers=["0"])
    fill_lora(model[0], seed=2)
    x = torch.randn(3, 8)
    assert not torch.allclose(model(x), base_out, atol=1e-6)


def test_merge_folds_delta_into_base_weights():
    model = make_model().eval()
    facade = PEFTAdapter()
    facade.attach(model, "lora_a", r=4, alpha=8.0, dropout=0.0, target_layers=["0", "1"])
    fill_lora(model[0], seed=3)
    fill_lora(model[1], seed=3)
    base_weights_0 = model[0].original_linear.weight.detach().clone()
    base_weights_1 = model[1].original_linear.weight.detach().clone()
    deltas_0 = (model[0].lora_B @ model[0].lora_A) * model[0].scaling
    deltas_1 = (model[1].lora_B @ model[1].lora_A) * model[1].scaling
    x = torch.randn(4, 8)
    before = model(x)
    merged = facade.merge("lora_a")
    assert merged is model
    assert isinstance(model[0], nn.Linear)
    assert isinstance(model[1], nn.Linear)
    assert torch.allclose(model[0].weight, base_weights_0 + deltas_0, atol=1e-6)
    assert torch.allclose(model[1].weight, base_weights_1 + deltas_1, atol=1e-6)
    assert torch.allclose(model(x), before, atol=1e-5)
    assert facade.list_adapters() == []


def test_merge_unattached_raises():
    facade = PEFTAdapter()
    facade.attach(make_model(), "lora_a", target_layers=["0"])
    with pytest.raises(ValueError, match="not attached"):
        facade.merge("lora_b")


def test_multiple_adapters_share_base_weights():
    model = make_model().eval()
    facade = PEFTAdapter()
    facade.attach(model, "lora_a", r=4, alpha=8.0, dropout=0.0, target_layers=["0"])
    fill_lora(model[0], seed=11)
    facade.attach(model, "lora_b", r=4, alpha=8.0, dropout=0.0, target_layers=["0"])
    fill_lora(model[0], seed=22)
    assert facade.active == "lora_b"
    x = torch.randn(2, 8)
    out_b = model(x)
    facade.activate("lora_a")
    out_a = model(x)
    facade.deactivate()
    out_base = model(x)
    assert not torch.allclose(out_a, out_base, atol=1e-6)
    assert not torch.allclose(out_b, out_base, atol=1e-6)
    assert not torch.allclose(out_a, out_b, atol=1e-6)
    assert facade.active is None
    assert set(facade.list_adapters()) == {"lora_a", "lora_b"}


def test_activate_unknown_raises():
    facade = PEFTAdapter()
    with pytest.raises(ValueError, match="not attached"):
        facade.activate("ghost")
