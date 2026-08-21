"""tests/continual_learning/test_knowledge_anchor.py
==================================================
Unit tests for knowledge anchors.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cadgenesis.continual_learning.knowledge_anchor import KnowledgeAnchor


def _make_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_anchor_loss_zero_after_anchor():
    model = _make_model()
    anchor = KnowledgeAnchor()
    anchor.anchor(model)
    assert anchor.is_anchored
    assert len(anchor) == 4
    assert anchor.anchor_loss(model).item() == 0.0


def test_anchor_loss_grows_and_restore_is_exact():
    model = _make_model()
    anchor = KnowledgeAnchor()
    anchor.anchor(model)
    state_before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.1)
    loss_before_restore = anchor.anchor_loss(model).item()
    assert loss_before_restore > 0.0
    anchor.restore(model)
    assert anchor.anchor_loss(model).item() == 0.0
    for key, value in model.state_dict().items():
        assert torch.equal(value, state_before[key])


def test_anchor_loss_grows_with_perturbation_magnitude():
    model = _make_model()
    anchor = KnowledgeAnchor()
    anchor.anchor(model)
    with torch.no_grad():
        model[0].weight.mul_(1.001)
    small = anchor.anchor_loss(model).item()
    with torch.no_grad():
        model[0].weight.mul_(1.5)
    large = anchor.anchor_loss(model).item()
    assert large > small > 0.0


def test_move_anchor_re_anchors_to_current_weights():
    model = _make_model()
    anchor = KnowledgeAnchor()
    anchor.anchor(model)
    with torch.no_grad():
        for param in model.parameters():
            param.mul_(2.0)
    anchor.move_anchor(model)
    assert anchor.anchor_loss(model).item() == 0.0


def test_anchor_loss_zero_before_anchoring():
    model = _make_model()
    assert KnowledgeAnchor().anchor_loss(model).item() == 0.0


def test_anchor_loss_gradient_flows_to_params():
    model = _make_model()
    anchor = KnowledgeAnchor()
    anchor.anchor(model)
    loss = anchor.anchor_loss(model)
    loss.backward()
    assert model[0].weight.grad is not None
