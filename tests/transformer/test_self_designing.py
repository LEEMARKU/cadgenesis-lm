"""
Tests for the Self-Designing Transformer subsystem (NAS, routing, adaptive
heads, pruning, MoE growth, rollback) and the SparseMoEFFN.
"""

import pytest
import torch

from cadgenesis.config import CADConfig
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer
from cadgenesis.transformer.moe import SparseMoEFFN
from cadgenesis.transformer.self_designing import (
    AdaptiveAttentionHeadSelector,
    ArchitectureEvaluator,
    ArchitectureSearchSpace,
    ArchitectureSpec,
    AutomaticRollback,
    DynamicLayerRouter,
    LayerPruningController,
    NeuralArchitectureSearch,
    SelfDesigningTransformer,
)


@pytest.fixture
def mini_config() -> CADConfig:
    return CADConfig.mini()


def _mini_data():
    return [
        ([5, 12, 3, 8], [1, 10, 22, 7, 2]),
        ([9, 4, 15], [1, 5, 18, 2]),
        ([2, 11, 6, 13, 1], [1, 30, 4, 9, 2]),
    ]


class TestArchitectureSpec:
    def test_validate_heads(self):
        with pytest.raises(ValueError):
            ArchitectureSpec(nhead=4, self_attn_heads=4, geometry_attn_heads=1)

    def test_validate_divisibility(self):
        with pytest.raises(ValueError):
            ArchitectureSpec(d_model=100, nhead=8)

    def test_to_model_config_roundtrip(self):
        cfg = CADConfig.mini()
        spec = ArchitectureSpec.from_model_config(cfg.model)
        assert spec.to_model_config() == cfg.model

    def test_signature_stable(self):
        a = ArchitectureSpec(num_encoder_layers=2)
        b = ArchitectureSpec(num_encoder_layers=2)
        assert a.signature() == b.signature()
        assert a.signature() != ArchitectureSpec(num_encoder_layers=3).signature()

    def test_search_space_samples_valid(self):
        space = ArchitectureSearchSpace()
        for _ in range(20):
            space.sample().validate()


class TestSelfDesigningTransformer:
    def test_forward(self, mini_config):
        w = SelfDesigningTransformer(mini_config)
        src = torch.randint(0, 50, (2, 12))
        tgt_in = torch.randint(0, 30, (2, 6))
        tgt_type = torch.randint(0, 3, (2, 6))
        logits, conf = w(src, tgt_in, tgt_type)
        assert logits.shape == (2, 6, logits.shape[-1])
        assert conf.shape == (2, 6, 1)
        (logits.sum() + conf.sum()).backward()

    def test_evaluate_complexity(self, mini_config):
        w = SelfDesigningTransformer(mini_config)
        src = torch.randint(0, 50, (2, 12))
        assert w.evaluate_complexity(src).shape == (2, 1)

    def test_moe_forward_and_growth(self, mini_config):
        mini_config.model.use_moe = True
        mini_config.model.num_experts = 4
        mini_config.model.top_k_experts = 2
        w = SelfDesigningTransformer(mini_config)
        src = torch.randint(0, 50, (2, 10))
        tgt_in = torch.randint(0, 30, (2, 5))
        tgt_type = torch.randint(0, 3, (2, 5))
        logits, conf = w(src, tgt_in, tgt_type)
        (logits.sum() + conf.sum()).backward()

        grown = w.grow_experts(1)
        assert grown and all(v == 5 for v in grown.values())
        assert all(len(loads) == 5 for loads in w.expert_load().values())

        # forward after growth is still differentiable
        logits2, _ = w(src, tgt_in, tgt_type)
        logits2.sum().backward()

    def test_prune_and_unprune(self, mini_config):
        w = SelfDesigningTransformer(mini_config)
        pruned = w.prune_layers(0.25)
        assert len(pruned) >= 1
        report = w.pruning.report()
        assert len(report["pruned"]) >= 1
        w.unprune_layers()
        assert w.pruning.effective_layers() == (
            mini_config.model.num_encoder_layers,
            mini_config.model.num_decoder_layers,
        )

    def test_rollback(self, mini_config):
        w = SelfDesigningTransformer(mini_config)
        w.snapshot(2.0)
        w.snapshot(1.0)
        assert w.check_performance(1.0) is None
        assert w.rollback.report()["best_metric"] == 1.0

    def test_architecture_report(self, mini_config):
        w = SelfDesigningTransformer(mini_config)
        report = w.architecture_report()
        assert report["encoder_layers"] == mini_config.model.num_encoder_layers
        assert report["params"] > 0

    def test_search_architecture(self):
        w = SelfDesigningTransformer(CADConfig.mini())
        best, score, summary = w.search_architecture(_mini_data(), iterations=2, mode="random")
        assert isinstance(best, ArchitectureSpec)
        assert score != float("-inf")
        assert summary["history_size"] == 2
        # forward with the discovered architecture still works
        src = torch.randint(0, 50, (2, 8))
        tgt_in = torch.randint(0, 30, (2, 4))
        tgt_type = torch.randint(0, 3, (2, 4))
        logits, _ = w(src, tgt_in, tgt_type)
        assert logits.ndim == 3


class TestDynamicLayerRouter:
    def test_layer_gate_shape(self):
        router = DynamicLayerRouter(d_model=64, num_layers=4)
        x = torch.randn(2, 8, 64)
        mask = router(x)
        assert mask.shape == (2, 8, 4)
        gate = router.layer_gate(2, x)
        assert gate.shape == (2, 8, 1)

    def test_keep_ratio_bounds(self):
        router = DynamicLayerRouter(d_model=64, num_layers=4)
        ratio = router.keep_ratio(torch.randn(2, 8, 64))
        assert 0.0 <= ratio <= 1.0

    def test_out_of_range(self):
        router = DynamicLayerRouter(d_model=64, num_layers=2)
        with pytest.raises(IndexError):
            router.layer_gate(5, torch.randn(1, 4, 64))


class TestAdaptiveAttentionHeadSelector:
    def test_head_weights_shape(self):
        hs = AdaptiveAttentionHeadSelector(d_model=64, num_layers=4, num_active_heads=4)
        x = torch.randn(2, 8, 64)
        weights = hs(x)
        assert weights.shape == (2, 8, 4, 4)

    def test_active_head_ratio_bounds(self):
        hs = AdaptiveAttentionHeadSelector(d_model=64, num_layers=4, num_active_heads=4)
        ratio = hs.active_head_ratio(torch.randn(2, 8, 64))
        assert 0.0 <= ratio <= 1.0


class TestLayerPruningController:
    def test_resizes_to_model(self):
        cfg = CADConfig.mini()
        back = GeometryAwareTransformer(cfg)
        pc = LayerPruningController(2, 2)
        pc.compute_importance(back)
        assert pc.effective_layers() == (
            cfg.model.num_encoder_layers,
            cfg.model.num_decoder_layers,
        )

    def test_prune_unprune(self):
        pc = LayerPruningController(3, 3)
        pruned = pc.prune_layers(0.5)
        assert len(pruned) == 4
        assert pc.effective_layers() == (1, 1)
        pc.unprune_all()
        assert pc.effective_layers() == (3, 3)


class TestAutomaticRollback:
    def test_rolls_back_after_patience(self):
        from cadgenesis.config import CADConfig

        back = GeometryAwareTransformer(CADConfig.mini())
        rb = AutomaticRollback(back, tolerance=0.05, patience=2)
        rb.snapshot(10.0)
        assert rb.check_and_rollback(11.0) is None
        rolled = rb.check_and_rollback(11.5)
        assert rolled is not None
        assert rb.report()["rollback_count"] == 1

    def test_no_rollback_on_improvement(self):
        from cadgenesis.config import CADConfig

        back = GeometryAwareTransformer(CADConfig.mini())
        rb = AutomaticRollback(back, tolerance=0.05, patience=2)
        rb.snapshot(10.0)
        assert rb.check_and_rollback(8.0) is None
        rb.snapshot(8.0)
        assert rb.report()["best_metric"] == 8.0


class TestArchitectureEvaluator:
    def test_score_returns_finite(self):
        ev = ArchitectureEvaluator(train_steps=2, eval_batches=1)
        spec = ArchitectureSpec(num_encoder_layers=1, num_decoder_layers=1, d_model=64)
        score = ev.score(spec, _mini_data())
        assert score != float("-inf")


class TestNeuralArchitectureSearch:
    def test_random_search_best(self):
        def fake_eval(spec):
            return -float(spec.num_encoder_layers + spec.num_decoder_layers)

        nas = NeuralArchitectureSearch(evaluator=fake_eval, seed=0)
        _best, score = nas.random_search(iterations=6)
        assert score >= -8

    def test_evolutionary_returns_spec(self):
        def fake_eval(spec):
            return -float(spec.d_model)

        nas = NeuralArchitectureSearch(evaluator=fake_eval, seed=0)
        best, score = nas.evolutionary(generations=2, population_size=4)
        assert isinstance(best, ArchitectureSpec)
        assert score == -64.0  # lowest d_model wins
        assert nas.summary()["history_size"] > 0


class TestSparseMoEFFN:
    def test_forward_and_aux_loss(self):
        moe = SparseMoEFFN(d_model=64, num_experts=4, top_k=2)
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert out.shape == (2, 8, 64)
        loss = out.sum() + moe.get_aux_loss().sum()
        loss.backward()

    def test_add_remove_expert(self):
        moe = SparseMoEFFN(d_model=64, num_experts=3, top_k=2)
        moe.add_expert()
        assert moe.num_experts == 4
        assert moe.router.weight.shape[0] == 4
        moe.remove_expert(0)
        assert moe.num_experts == 3

    def test_expert_load(self):
        moe = SparseMoEFFN(d_model=64, num_experts=3, top_k=1)
        x = torch.randn(4, 6, 64)
        moe(x)
        loads = moe.expert_load()
        assert len(loads) == 3
        assert sum(loads) == 4 * 6

    def test_routing_balance(self):
        moe = SparseMoEFFN(d_model=64, num_experts=3, top_k=1)
        moe(torch.randn(4, 6, 64))
        balance = moe.routing_balance()
        assert balance >= 0.0
