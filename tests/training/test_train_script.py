"""End-to-end test of the checkpointable train entrypoint (tiny + fast)."""

import argparse
import sys
from pathlib import Path

import torch

from cadgenesis.datasets.cad_program_synth import write_synthetic_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cadgenesis.cli.generate import load_engine
from cadgenesis.config import CADConfig
from train import run_training


def _args(tmp_path, out_dir, epochs=2, seed=7, n=24, **overrides):
    data = write_synthetic_jsonl(tmp_path / "progs.jsonl", n=n, seed=seed)
    base = dict(
        data=str(data),
        out_dir=str(out_dir),
        epochs=epochs,
        batch_size=8,
        seed=seed,
        lr=None,
        schedule="wsd",
        packed=True,
        val_fraction=0.2,
        max_records=None,
        model="mini",
        warmup_steps=5,
        resume_from=None,
        bf16=False,
        fsdp=False,
        mu_transfer=False,
        d_model=None,
        optimizer="adamw",
        device=None,
        packed_max_src_len=None,
        packed_max_tgt_len=None,
        teacher=None,
        prompts=None,
        prompts_file=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_training_saves_checkpoints_with_digest(tmp_path):
    result = run_training(_args(tmp_path, tmp_path / "run1"))
    assert result["digest"]
    ckpt = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=False)
    assert ckpt["run_digest"] == result["digest"]
    assert ckpt["epoch"] >= 1
    assert "model_state_dict" in ckpt and "optimizer_state_dict" in ckpt
    assert ckpt["config"]["model"]["d_model"] == CADConfig.mini().model.d_model
    assert len(result["checkpoints"]) == 2
    assert result["train_loss"] >= 0.0


def test_identical_runs_produce_identical_digest(tmp_path):
    r1 = run_training(_args(tmp_path, tmp_path / "a"))
    r2 = run_training(_args(tmp_path, tmp_path / "b"))
    assert r1["digest"] == r2["digest"]


def test_different_seed_changes_digest(tmp_path):
    r1 = run_training(_args(tmp_path, tmp_path / "a", seed=7))
    r2 = run_training(_args(tmp_path, tmp_path / "b", seed=8))
    assert r1["digest"] != r2["digest"]


def test_training_loss_decreases(tmp_path):
    r_early = run_training(_args(tmp_path, tmp_path / "run", epochs=1, n=80, lr=1e-3))
    r_late = run_training(_args(tmp_path, tmp_path / "run2", epochs=8, n=80, lr=1e-3))
    assert r_late["train_loss"] < r_early["train_loss"]


def test_resume_continues_from_checkpoint(tmp_path):
    out = tmp_path / "resume_run"
    run_training(_args(tmp_path, out, epochs=2))
    resume_ckpt = out / "checkpoint_epoch_2.pt"
    r = run_training(_args(tmp_path, out, epochs=4, resume_from=str(resume_ckpt)))
    assert r["digest"]
    assert len(r["checkpoints"]) == 2  # epochs 3 and 4
    final = torch.load(out / "checkpoint_epoch_4.pt", map_location="cpu", weights_only=False)
    assert final["run_digest"] == r["digest"]


def test_resume_with_mismatched_config_refuses(tmp_path):
    out = tmp_path / "bad_resume"
    run_training(_args(tmp_path, out, epochs=2))
    resume_ckpt = out / "checkpoint_epoch_2.pt"
    import pytest

    with pytest.raises(SystemExit):
        run_training(_args(tmp_path, out, epochs=4, seed=999, resume_from=str(resume_ckpt)))


def test_resume_replays_exact_interrupted_trajectory(tmp_path):
    out = tmp_path / "interrupt"
    run_training(_args(tmp_path, out, epochs=2))
    run_training(_args(tmp_path, out, epochs=4, resume_from=str(out / "checkpoint_epoch_2.pt")))
    run_training(_args(tmp_path, tmp_path / "fresh", epochs=4))

    import torch

    r4 = torch.load(out / "checkpoint_epoch_4.pt", map_location="cpu", weights_only=False)
    f4 = torch.load(
        tmp_path / "fresh" / "checkpoint_epoch_4.pt", map_location="cpu", weights_only=False
    )
    assert r4["validation_loss"] == f4["validation_loss"]
    assert torch.equal(
        r4["model_state_dict"]["out_proj.weight"], f4["model_state_dict"]["out_proj.weight"]
    )


def test_trained_checkpoint_loads_and_decodes(tmp_path):
    """A saved checkpoint must reload via the CLI loader and generate tokens
    whose ids resolve in the restored vocab (incl. train-time extensions)."""
    run_training(_args(tmp_path, tmp_path / "run", epochs=2, n=60))
    ckpt = tmp_path / "run" / "best_checkpoint.pt"
    engine = load_engine(str(ckpt), device="cpu")
    result = engine.greedy("create a steel box", max_len=12)
    assert result.ids, "expected non-empty generation"
    id2tok = engine.tokenizer.vocab.to_id2tok()
    assert all(i in id2tok for i in result.ids)
    assert result.tokens, "tokens must decode"


def test_unpacked_path_also_works(tmp_path):
    r = run_training(_args(tmp_path, tmp_path / "unpacked", packed=False, epochs=1))
    assert r["digest"]
    assert r["val_loss"] >= 0.0
