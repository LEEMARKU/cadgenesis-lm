"""tests/transformer/test_mtp.py
================================
Unit tests for cadgenesis.transformer.mtp.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from cadgenesis.transformer.mtp import MultiTokenPredictionHead, mtp_loss


class TestMultiTokenPredictionHead:
    def test_shape(self):
        d_model, vocab_size, depth, B, T = 32, 100, 3, 2, 10
        head = MultiTokenPredictionHead(d_model, vocab_size, mtp_depth=depth)
        embed = nn.Embedding(vocab_size, d_model)
        hidden = torch.randn(B, T, d_model)
        targets = torch.randint(1, vocab_size, (B, T))
        logits_list = head(hidden, targets, embed)
        assert len(logits_list) == depth
        assert logits_list[0].shape == (B, T - 1, vocab_size)
        assert logits_list[1].shape == (B, T - 2, vocab_size)
        assert logits_list[2].shape == (B, T - 3, vocab_size)

    def test_weight_tied_logits(self):
        d_model, vocab_size, depth, B, T = 16, 20, 2, 2, 5
        head = MultiTokenPredictionHead(d_model, vocab_size, mtp_depth=depth, dropout=0.0)
        head.eval()
        embed = nn.Embedding(vocab_size, d_model)
        hidden = torch.randn(B, T, d_model)
        targets = torch.randint(1, vocab_size, (B, T))
        logits_list = head(hidden, targets, embed)
        with torch.no_grad():
            h1 = head.norms["1"](hidden + head.branches["1"](hidden))
            expected_1 = (h1 @ embed.weight.T)[:, : T - 1]
            h2 = head.norms["2"](embed(targets[:, 1:]) + head.branches["2"](h1[:, : T - 1]))
            expected_2 = (h2 @ embed.weight.T)[:, : T - 2]
        assert torch.allclose(logits_list[0], expected_1)
        assert torch.allclose(logits_list[1], expected_2)

    def test_single_depth(self):
        head = MultiTokenPredictionHead(16, 20, mtp_depth=1)
        embed = nn.Embedding(20, 16)
        B, T = 2, 5
        hidden = torch.randn(B, T, 16)
        targets = torch.randint(1, 20, (B, T))
        logits_list = head(hidden, targets, embed)
        assert len(logits_list) == 1
        assert logits_list[0].shape == (B, T - 1, 20)
        total, breakdown = mtp_loss(logits_list, targets, 1)
        assert torch.isfinite(total)
        assert breakdown["mtp_total"] == pytest.approx(breakdown["mtp_d1"])

    def test_invalid_depth(self):
        with pytest.raises(ValueError):
            MultiTokenPredictionHead(16, 20, mtp_depth=0)
        with pytest.raises(ValueError):
            MultiTokenPredictionHead(16, 20, mtp_depth=-3)


class TestMTPLoss:
    def test_loss_finite_and_decreases(self):
        torch.manual_seed(0)
        d_model, vocab_size, depth, B, T = 32, 50, 2, 4, 8
        head = MultiTokenPredictionHead(d_model, vocab_size, mtp_depth=depth, dropout=0.0)
        embed = nn.Embedding(vocab_size, d_model)
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
        hidden = torch.randn(B, T, d_model)
        targets = torch.randint(1, vocab_size, (B, T))
        logits_list = head(hidden, targets, embed)
        total, breakdown = mtp_loss(logits_list, targets, depth)
        assert torch.isfinite(total)
        assert set(breakdown) >= {"mtp_d1", "mtp_d2", "mtp_total"}
        start = total.item()
        for _ in range(10):
            optimizer.zero_grad()
            logits_list = head(hidden, targets, embed)
            total, _ = mtp_loss(logits_list, targets, depth)
            total.backward()
            optimizer.step()
        final = total.item()
        assert torch.isfinite(torch.tensor(final))
        assert final < start

    def test_pad_masking(self):
        d_model, vocab_size, depth, B, T = 16, 30, 2, 3, 6
        head = MultiTokenPredictionHead(d_model, vocab_size, mtp_depth=depth, dropout=0.0)
        head.eval()
        embed = nn.Embedding(vocab_size, d_model)
        hidden = torch.randn(B, T, d_model)
        targets = torch.randint(1, vocab_size, (B, T))
        targets[:, 3:] = 0
        logits_list = head(hidden, targets, embed)
        total, _breakdown = mtp_loss(logits_list, targets, depth, pad_id=0)
        assert torch.isfinite(total)
        expected = 0.0
        for d in range(1, depth + 1):
            tgt = targets[:, d:]
            mask = tgt != 0
            expected = expected + torch.nn.functional.cross_entropy(
                logits_list[d - 1][mask], tgt[mask]
            )
        expected = expected / depth
        assert total.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_custom_weights(self):
        d_model, vocab_size, depth, B, T = 16, 20, 2, 2, 5
        head = MultiTokenPredictionHead(d_model, vocab_size, mtp_depth=depth, dropout=0.0)
        head.eval()
        embed = nn.Embedding(vocab_size, d_model)
        hidden = torch.randn(B, T, d_model)
        targets = torch.randint(1, vocab_size, (B, T))
        logits_list = head(hidden, targets, embed)
        weights = [0.25, 0.75]
        total, breakdown = mtp_loss(logits_list, targets, depth, weights=weights)
        expected = 0.25 * breakdown["mtp_d1"] + 0.75 * breakdown["mtp_d2"]
        assert total.item() == pytest.approx(expected, rel=1e-5)
        with pytest.raises(ValueError):
            mtp_loss(logits_list, targets, depth, weights=[0.5])


class TestGradientFlow:
    def test_all_branches_receive_gradients(self):
        d_model, vocab_size, depth, B, T = 16, 20, 3, 2, 6
        head = MultiTokenPredictionHead(d_model, vocab_size, mtp_depth=depth, dropout=0.0)
        embed = nn.Embedding(vocab_size, d_model)
        hidden = torch.randn(B, T, d_model)
        targets = torch.randint(1, vocab_size, (B, T))
        logits_list = head(hidden, targets, embed)
        total, _ = mtp_loss(logits_list, targets, depth)
        total.backward()
        for d in range(1, depth + 1):
            linear = head.branches[str(d)][1]
            assert linear.weight.grad is not None
            assert torch.isfinite(linear.weight.grad).all()
            assert linear.weight.grad.abs().sum() > 0
