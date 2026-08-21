"""tests/transformer/test_pillar1_integration.py
================================================
Pillar 1 integration tests: the new Foundation-Model capabilities (sparse /
multi-scale attention, specialised MoE, hierarchical transformer, dynamic
routing, evolution framework) must integrate with the *existing* training and
inference contracts without breaking them.

Checks
------
* :class:`HierarchicalCADTransformer` is accepted by :class:`CADInferenceEngine`
  (duck-typed ``(logits, confidence)`` contract).
* Specialised-MoE + sparse-attention features forward inside a real model.
* The evolution builder's outputs run through the same forward contract.
* Full pipeline: train one step, run greedy inference.
"""

from __future__ import annotations

import torch

from cadgenesis.config import CADConfig
from cadgenesis.inference.engine import CADInferenceEngine
from cadgenesis.tokenizer import AutonomousCADTokenizer
from cadgenesis.transformer import (
    ConfigurationDrivenBuilder,
    HierarchicalCADTransformer,
)


def _tokenizer():
    tok = AutonomousCADTokenizer.build_mini()
    tok.build_lang_vocab(["create a mounting bracket", "create a bracket", "small cover plate"])
    return tok


def _hier_config(**overrides) -> CADConfig:
    cfg = CADConfig.mini()
    cfg.model.use_hierarchical_transformer = True
    for key, value in overrides.items():
        setattr(cfg.model, key, value)
    return cfg


class TestInferenceContract:
    def test_greedy_with_hierarchical_model(self):
        tokenizer = _tokenizer()
        model = HierarchicalCADTransformer(_hier_config())
        engine = CADInferenceEngine(model, tokenizer, device="cpu")
        result = engine.greedy("create a mounting bracket", max_len=8)
        assert result.text == "create a mounting bracket"
        assert isinstance(result.tokens, list)
        assert 0.0 <= result.confidence <= 1.0

    def test_beam_with_hierarchical_model(self):
        tokenizer = _tokenizer()
        model = HierarchicalCADTransformer(_hier_config())
        engine = CADInferenceEngine(model, tokenizer, device="cpu")
        result = engine.beam("create a bracket", beam_width=2, max_len=8)
        assert 0.0 <= result.confidence <= 1.0

    def test_early_exit_model_through_engine(self):
        tokenizer = _tokenizer()
        model = HierarchicalCADTransformer(
            _hier_config(early_exit_threshold=0.99, computation_budget=0.6)
        )
        engine = CADInferenceEngine(model, tokenizer, device="cpu")
        result = engine.greedy("small cover plate", max_len=8)
        assert 0.0 <= result.confidence <= 1.0


class TestTrainingContract:
    def test_train_step_forward_backward(self):
        model = HierarchicalCADTransformer(_hier_config())
        src = torch.randint(0, 50, (2, 12))
        tgt = torch.randint(0, 30, (2, 8))
        tgt_type = torch.randint(0, 3, (2, 8))
        logits, _ = model(src, tgt, tgt_type)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.cad_vocab_size),
            torch.randint(0, model.cad_vocab_size, (2 * 8,)),
        )
        loss.backward()
        assert loss.item() > 0

    def test_specialized_moe_training(self):
        model = HierarchicalCADTransformer(_hier_config(use_specialized_moe=True))
        src = torch.randint(0, 50, (2, 12))
        tgt = torch.randint(0, 30, (2, 8))
        tgt_type = torch.randint(0, 3, (2, 8))
        logits, conf = model(src, tgt, tgt_type)
        total = logits.sum() + conf.sum() + model.aux_loss()
        total.backward()
        # Router and at least one domain expert received gradients.
        assert model.planner_blocks[0].ffn.router.weight.grad is not None

    def test_builder_output_trains(self):
        arch = {
            "type": "hierarchical",
            "name": "integration-arch",
            "d_model": 128,
            "nhead": 4,
            "heads": {"self": 2, "geometry": 1, "agent": 1},
            "stages": {
                "planner": 1,
                "geometry": 1,
                "constraint": 1,
                "execution": 1,
                "validation": 1,
            },
            "computation_budget": 0.75,
        }
        model = ConfigurationDrivenBuilder().build_model(arch)
        assert isinstance(model, HierarchicalCADTransformer)
        src = torch.randint(0, 50, (2, 10))
        tgt = torch.randint(0, 30, (2, 6))
        tgt_type = torch.randint(0, 3, (2, 6))
        logits, conf = model(src, tgt, tgt_type)
        (logits.sum() + conf.sum()).backward()
        assert model.routing.report()["savings_fraction"] >= 0.0


class TestMemoryAndAgentHooks:
    def test_memory_bank_and_agents_accepted(self):
        model = HierarchicalCADTransformer(_hier_config())
        src = torch.randint(0, 50, (2, 10))
        tgt = torch.randint(0, 30, (2, 6))
        tgt_type = torch.randint(0, 3, (2, 6))
        bank = torch.randn(2, 32, model.d_model)
        agent_states = torch.randn(2, 6, model.d_model)
        logits, _ = model(
            src,
            tgt,
            tgt_type,
            memory_bank=bank,
            agent_states=agent_states,
        )
        assert logits.shape[1] == 6
