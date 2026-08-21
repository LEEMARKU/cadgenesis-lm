"""tests/training/test_callbacks.py"""

from __future__ import annotations

import pytest

from cadgenesis.training.callbacks import (
    CallbackRegistry,
    CheckpointCallback,
    EarlyStoppingCallback,
    StopTraining,
    TrainerCallback,
    TrainingEvent,
)


class RecordingCallback(TrainerCallback):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_train_begin(self, event: TrainingEvent) -> None:
        self.events.append("train_begin")

    def on_epoch_begin(self, event: TrainingEvent) -> None:
        self.events.append(f"epoch_begin:{event.epoch}")

    def on_step(self, event: TrainingEvent) -> None:
        self.events.append("step")

    def on_epoch_end(self, event: TrainingEvent) -> None:
        self.events.append("epoch_end")

    def on_train_end(self, event: TrainingEvent) -> None:
        self.events.append("train_end")


def test_registry_fires_in_order():
    registry = CallbackRegistry()
    recorder = RecordingCallback()
    registry.add(recorder)
    event = TrainingEvent(epoch=1, step=10)
    registry.on_train_begin(event)
    registry.on_epoch_begin(event)
    registry.on_step(event)
    registry.on_epoch_end(event)
    registry.on_train_end(event)
    assert recorder.events == [
        "train_begin",
        "epoch_begin:1",
        "step",
        "epoch_end",
        "train_end",
    ]


def test_registry_deduplicates_and_removes():
    registry = CallbackRegistry()
    callback = TrainerCallback()
    registry.add(callback)
    registry.add(callback)
    assert len(registry) == 1
    registry.remove(callback)
    assert len(registry) == 0
    registry.clear()


def test_registry_clear():
    registry = CallbackRegistry()
    registry.add(TrainerCallback())
    registry.clear()
    assert len(registry) == 0


def test_early_stopping_raises_after_patience():
    callback = EarlyStoppingCallback(patience=2, min_delta=0.1)
    event = TrainingEvent(epoch=0, validation_metrics={"loss": 1.0})
    callback.on_validation(event)
    with pytest.raises(StopTraining):
        for epoch in range(1, 4):
            event = TrainingEvent(
                epoch=epoch,
                validation_metrics={"loss": 1.5},
                best_validation_loss=callback.best_value,
            )
            callback.on_validation(event)


def test_early_stopping_improvement_resets_wait():
    callback = EarlyStoppingCallback(patience=3)
    event = TrainingEvent(epoch=0, validation_metrics={"loss": 1.0})
    callback.on_validation(event)
    event = TrainingEvent(epoch=1, validation_metrics={"loss": 0.5})
    callback.on_validation(event)
    assert callback.wait == 0
    assert callback.best_value == 0.5


def test_checkpoint_callback_saves_last_and_best():
    saved: list[tuple[str, int, int]] = []

    def fake_save(path: str, epoch: int, step: int, val_loss: float | None) -> None:
        saved.append((path, epoch, step))

    callback = CheckpointCallback(fake_save, "out", every_epochs=1, save_best=True)
    event = TrainingEvent(
        epoch=0,
        step=100,
        metrics={"loss": 1.0},
        validation_metrics={"loss": 1.0},
    )
    callback.on_epoch_end(event)
    assert saved[0][0].endswith("last.pt")
    assert saved[1][0].endswith("best.pt")
    assert callback.best_value == 1.0


def test_checkpoint_callback_skips_interval():
    saved: list[str] = []

    def fake_save(path: str, epoch: int, step: int, val_loss: float | None) -> None:
        saved.append(path)

    callback = CheckpointCallback(fake_save, "out", every_epochs=3, save_best=False)
    callback.on_epoch_end(TrainingEvent(epoch=0, step=0, metrics={"loss": 1.0}))
    callback.on_epoch_end(TrainingEvent(epoch=1, step=1, metrics={"loss": 1.0}))
    assert saved == []


def test_training_event_as_dict():
    event = TrainingEvent(epoch=2, step=5, metrics={"loss": 0.5})
    data = event.as_dict()
    assert data["epoch"] == 2
    assert data["metrics"]["loss"] == 0.5
