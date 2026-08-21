"""
tests/transformer/test_modern_wiring.py
=======================================
End-to-end wiring of the 2025-2026 frontier features into the model:
hybrid SSM, BitNet, shared-expert MoE, and YaRN RoPE scaling.
"""

from __future__ import annotations

import pytest
import torch

from cadgenesis.config import CADConfig
from cadgenesis.quantization.bitnet import BitLinear
from cadgenesis.transformer.geometry_transformer import GeometryAwareTransformer
from cadgenesis.transformer.positional import RotaryEmbedding


@pytest.fixture(autouse=True)
def _restore_rope_defaults():
    yield
    RotaryEmbedding.configure_defaults(
        max_position_embeddings=4096,
        base=10000.0,
        scaling_factor=1.0,
        scaling_type="none",
    )


def _modern_config() -> CADConfig:
    cfg = CADConfig.mini()
    m = cfg.model
    # Keep d_model=128 so every head count divides the width.
    m.use_ssm = True
    m.ssm_every_n_blocks = 2
    m.ssm_heads = 4
    m.use_bitnet = True
    m.use_moe = True
    m.num_experts = 4
    m.num_shared_experts = 2
    m.shared_expert_dim = 32
    m.rope_scaling_type = "yarn"
    m.rope_scaling_factor = 4.0
    m.max_position_embeddings = 512
    return cfg


class TestModernWiring:
    def test_builds_with_all_features(self):
        torch.manual_seed(0)
        model = GeometryAwareTransformer(_modern_config())
        # SSM modules present (encoder + decoder, every 2nd block).
        assert sum(1 for s in model.encoder_ssm if s is not None) >= 1
        assert sum(1 for s in model.decoder_ssm if s is not None) >= 1
        # BitLinear swapped in.
        assert sum(1 for m in model.modules() if isinstance(m, BitLinear)) > 0
        # Shared expert wired into at least one MoE block.
        from cadgenesis.transformer.moe import SparseMoEFFN

        shared_seen = False
        for block in (*model.encoder_blocks, *model.decoder_blocks):
            for sub in block.modules():
                if isinstance(sub, SparseMoEFFN) and sub.shared_expert is not None:
                    shared_seen = True
        assert shared_seen

    def test_forward_backward(self):
        torch.manual_seed(0)
        model = GeometryAwareTransformer(_modern_config())
        src = torch.randint(0, 50, (2, 8))
        tgt = torch.randint(0, 30, (2, 4))
        tt = torch.randint(0, 3, (2, 4))
        logits, conf = model(src, tgt, tt)
        assert logits.shape == (2, 4, model.cad_vocab_size)
        assert torch.isfinite(logits).all()
        (logits.sum() + conf.sum()).backward()

    def test_cached_equivalence(self):
        torch.manual_seed(0)
        model = GeometryAwareTransformer(_modern_config())
        model.eval()
        B, S, T = 2, 8, 5
        src = torch.randint(0, 50, (B, S))
        tgt = torch.randint(0, 30, (B, T))
        tt = torch.randint(0, 3, (B, T))
        with torch.no_grad():
            full, _ = model(src, tgt, tt)
            cache = model.prepare_decoder_cache(src)
            step_ids, step_tt = tgt[:, :1], tt[:, :1]
            outs = []
            for i in range(T):
                step, _ = model.decode_step(step_ids, step_tt, cache)
                outs.append(step)
                if i + 1 < T:
                    step_ids, step_tt = tgt[:, i + 1 : i + 2], tt[:, i + 1 : i + 2]
            diff = (torch.cat(outs, 1) - full).abs().max().item()
        assert diff < 1e-2

    def test_yarn_defaults_inherited_by_backends(self):
        _ = GeometryAwareTransformer(_modern_config())
        # A backend-constructed RotaryEmbedding inherits YaRN scaling.
        r = RotaryEmbedding(16)
        assert r.scaling_type == "yarn"
        assert r.attn_factor == 2.0

    def test_greedy_decoding_runs(self):
        torch.manual_seed(0)
        from cadgenesis.inference.engine import CADInferenceEngine
        from cadgenesis.tokenizer import AutonomousCADTokenizer

        tok = AutonomousCADTokenizer.build_mini()
        tok.build_lang_vocab(["create a steel box"])
        model = GeometryAwareTransformer(_modern_config())
        engine = CADInferenceEngine(model, tok, device="cpu")
        result = engine.greedy("create a steel box", max_len=8, use_cache=True)
        assert result.ids
        assert all(0 <= i < model.cad_vocab_size for i in result.ids)
