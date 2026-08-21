"""tests/training/test_checkpoint.py"""

from __future__ import annotations

import os

from cadgenesis.training.checkpoint import (
    CheckpointManager,
    cleanup_checkpoints,
    move_checkpoint,
)


def test_manager_saves_checkpoint(tmp_path):
    saved: list[tuple[str, int, int, float | None]] = []

    def fake_save(path: str, epoch: int, step: int, val_loss: float | None) -> None:
        saved.append((path, epoch, step, val_loss))

    manager = CheckpointManager(str(tmp_path), save_checkpoint=fake_save, every_epochs=1)
    path = manager.save(epoch=0, step=10, metrics={"loss": 0.5}, validation_loss=0.5)
    assert path.endswith("checkpoint-0-10.pt")
    assert saved[0][1] == 0
    assert saved[0][2] == 10
    assert saved[0][3] == 0.5


def test_manager_tracks_best(tmp_path):
    manager = CheckpointManager(str(tmp_path), save_checkpoint=None, every_epochs=1)
    first = manager.save(epoch=0, step=1, validation_loss=1.0)
    second = manager.save(epoch=1, step=2, validation_loss=0.3)
    assert manager.best_loss == 0.3
    assert manager.best[0][0] == 0.3
    assert second in manager.best[0][1]
    assert os.path.exists(first)


def test_manager_writes_meta(tmp_path):
    manager = CheckpointManager(str(tmp_path), save_checkpoint=None, every_epochs=1)
    manager.save(epoch=2, step=50, validation_loss=0.7)
    meta = manager.load_meta()
    assert meta["last_epoch"] == 2
    assert meta["last_step"] == 50
    assert meta["best_loss"] == 0.7


def test_manager_resume_from(tmp_path):
    manager = CheckpointManager(str(tmp_path), save_checkpoint=None, every_epochs=1)
    path = manager.save(epoch=0, step=10)
    assert manager.resume_from() == path


def test_manager_should_checkpoint_by_steps(tmp_path):
    manager = CheckpointManager(str(tmp_path), every_steps=100)
    assert not manager.should_checkpoint(step=50, epoch=0)
    assert manager.should_checkpoint(step=100, epoch=0)


def test_manager_retains_best_only(tmp_path):
    manager = CheckpointManager(str(tmp_path), save_checkpoint=None, keep_best=2)
    for epoch in range(4):
        manager.save(epoch=epoch, step=epoch, validation_loss=float(epoch))
    remaining = [name for name in os.listdir(tmp_path) if name.startswith("checkpoint-")]
    assert len(remaining) <= 2


def test_cleanup_checkpoints(tmp_path):
    for epoch in range(5):
        path = os.path.join(tmp_path, f"checkpoint-{epoch}-{epoch}.pt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("x")
    with open(os.path.join(tmp_path, "meta.json"), "w", encoding="utf-8") as handle:
        handle.write("{}")
    removed = cleanup_checkpoints(str(tmp_path), keep=2)
    assert len(removed) == 3
    remaining = [name for name in os.listdir(tmp_path) if name.startswith("checkpoint-")]
    assert len(remaining) == 2


def test_move_checkpoint(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "checkpoint-0-0.pt"
    source.write_text("x", encoding="utf-8")
    destination = tmp_path / "dest"
    target = move_checkpoint(str(source), str(destination))
    assert os.path.exists(target)
    assert not os.path.exists(str(source))
