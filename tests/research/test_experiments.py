from __future__ import annotations

import pytest

from cadgenesis.research.experiments import ExperimentTracker, Hyperparams, new_experiment_id


class TestHyperparams:
    def test_defaults(self):
        hp = Hyperparams()
        assert hp.optimizer == "adamw"
        assert hp.learning_rate == 3e-4

    def test_roundtrip(self):
        hp = Hyperparams(learning_rate=1e-3, extra={"notes": "x"})
        restored = Hyperparams.from_dict(hp.to_dict())
        assert restored.learning_rate == 1e-3
        assert restored.extra == {"notes": "x"}

    def test_fingerprint_stable(self):
        assert Hyperparams(seed=1).fingerprint() == Hyperparams(seed=1).fingerprint()
        assert Hyperparams(seed=1).fingerprint() != Hyperparams(seed=2).fingerprint()

    def test_new_id_unique(self):
        assert new_experiment_id() != new_experiment_id()


class TestExperimentTracker:
    def test_create_and_get(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "experiments")
        record = tracker.create(name="run-1", hyperparams=Hyperparams(batch_size=16))
        assert tracker.get(record.id) is record
        assert record.status == "running"

    def test_create_accepts_mapping(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "experiments")
        record = tracker.create(name="run-2", hyperparams={"learning_rate": 1e-2})
        assert record.hyperparams.learning_rate == 1e-2

    def test_log_metric_and_best(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "experiments")
        record = tracker.create(name="run")
        tracker.log_metric(record.id, "loss", 0.5, step=1)
        tracker.log_metric(record.id, "loss", 0.3, step=2)
        tracker.set_status(record.id, "completed", metric="loss", minimize=True)
        assert record.best_metric == 0.3
        assert record.best_metric_name == "loss"
        assert record.status == "completed"

    def test_best_query(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "experiments")
        for name, loss in (("a", 0.4), ("b", 0.2)):
            record = tracker.create(name=name)
            tracker.log_metric(record.id, "loss", loss)
            tracker.set_status(record.id, "completed", metric="loss")
        best = tracker.best(metric="loss", minimize=True, limit=1)
        assert best[0].name == "b"

    def test_notes_and_artifacts(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "experiments")
        record = tracker.create(name="run")
        tracker.add_note(record.id, "looks good")
        artifact = tmp_path / "plot.png"
        artifact.write_bytes(b"fake-png")
        tracker.attach_artifact(record.id, str(artifact), copy=True)
        assert "looks good" in record.notes
        assert record.artifacts[0].endswith("plot.png")
        assert (tmp_path / "experiments" / record.id / "artifacts" / "plot.png").exists()

    def test_export_summary(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "experiments")
        tracker.create(name="run")
        summary = tracker.export_summary()
        assert summary["count"] == 1
        assert summary["running"] == 1
        assert summary["completed"] == 0

    def test_persistence_across_instances(self, tmp_path):
        root = tmp_path / "experiments"
        tracker = ExperimentTracker(root)
        record = tracker.create(name="persisted")
        tracker.log_metric(record.id, "loss", 0.1)
        reloaded = ExperimentTracker(root)
        assert reloaded.get(record.id).metrics["loss"][0]["value"] == 0.1

    def test_require_missing(self, tmp_path):
        tracker = ExperimentTracker(tmp_path / "experiments")
        with pytest.raises(KeyError):
            tracker.log_metric("nope", "loss", 1.0)
