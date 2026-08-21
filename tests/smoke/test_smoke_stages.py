"""
tests/smoke/test_smoke_stages.py
================================
Fast CPU tests for the G15 smoke stages (mini preset, reduced sizes —
the full-size stages run via `scripts/smoke/run_all.py`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cadgenesis.smoke.runner import _render_markdown
from cadgenesis.smoke.stages import (
    has_updated_gradients,
    make_mini,
    parameter_count,
    stage1_forward_backward,
    stage2_tiny_dataset,
    stage3_overfit,
    stage4_dev_run,
)


class TestStage1:
    def test_pass_and_gradients(self):
        result = stage1_forward_backward(seed=42, batch_size=4)
        assert result["status"] == "PASS"
        assert torch.isfinite(torch.tensor(result["loss"]))
        assert result["gradients_updated"] is True
        assert result["parameters"] > 0
        assert result["duration_s"] >= 0

    def test_deterministic_loss_with_seed(self):
        a = stage1_forward_backward(seed=7, batch_size=4)["loss"]
        b = stage1_forward_backward(seed=7, batch_size=4)["loss"]
        assert a == pytest.approx(b, abs=1e-6)


class TestStage2:
    def test_loss_decreases(self):
        result = stage2_tiny_dataset(seed=42, n_records=20, epochs=1, batch_size=4)
        assert result["status"] == "PASS"
        assert result["final_train_loss"] < result["initial_val_loss"]
        assert torch.isfinite(torch.tensor(result["final_train_loss"]))


class TestStage3:
    def test_learning_happens(self):
        result = stage3_overfit(
            seed=42,
            n_records=4,
            max_steps=10,
            target_loss=-1.0,  # never reached: runs all 10 steps
            batch_size=4,
        )
        assert result["final_loss"] < result["initial_loss"]
        assert result["curve"]

    def test_fails_gracefully_when_target_unreached(self):
        result = stage3_overfit(
            seed=42,
            n_records=4,
            max_steps=1,
            target_loss=0.0,
            batch_size=4,
        )
        assert result["status"] == "FAIL"
        assert result["target_reached"] is False


class TestStage4:
    def test_persisted_artifacts(self, tmp_path: Path):
        result = stage4_dev_run(
            seed=42,
            n_records=24,
            epochs=1,
            batch_size=4,
            out_dir=tmp_path,
        )
        assert result["status"] == "PASS"
        assert result["epochs"] == 1
        assert (tmp_path / "metrics" / "metrics.jsonl").exists()
        assert (tmp_path / "last.pt").exists()
        rows = [
            json.loads(line)
            for line in (tmp_path / "metrics" / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert any(r["event"] == "validation" for r in rows)
        assert any(r["event"] == "checkpoint" for r in rows)
        assert result["checkpoint_epoch"] == 0


class TestHelpers:
    def test_parameter_count_positive(self):
        _config, _tok, model, _trainer = make_mini(seed=1)
        assert parameter_count(model) > 1000

    def test_has_updated_gradients_false_without_backward(self):
        _config, _tok, model, _trainer = make_mini(seed=1)
        assert has_updated_gradients(model) is False

    def test_render_markdown_verdict(self):
        text = _render_markdown(
            [
                {"stage": "stage1_forward_backward", "status": "PASS", "duration_s": 1.0,
                 "result": {"loss": 2.5}},
                {"stage": "stage2_tiny_dataset", "status": "PASS", "duration_s": 2.0,
                 "result": {"final_val_loss": 1.5}},
            ]
        )
        assert "ALL STAGES PASS" in text
        assert "stage1_forward_backward" in text

    def test_render_markdown_fail_verdict(self):
        text = _render_markdown(
            [{"stage": "stage3_overfit", "status": "FAIL", "duration_s": 1.0,
              "result": {"final_loss": 9.9}}]
        )
        assert "do NOT proceed" in text