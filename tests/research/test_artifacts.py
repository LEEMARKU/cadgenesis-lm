from __future__ import annotations

from cadgenesis.research.artifacts import ArtifactRegistry, file_sha256


class TestArtifactRegistry:
    def test_store_and_get(self, tmp_path):
        source = tmp_path / "model.pt"
        source.write_bytes(b"checkpoint-bytes")
        expected_hash = file_sha256(source)
        registry = ArtifactRegistry(tmp_path / "artifacts")
        record = registry.store("exp_1", str(source), name="model.pt")
        assert record.experiment_id == "exp_1"
        assert record.sha256 == expected_hash
        assert registry.get("exp_1", "model.pt") is record
        assert registry.path("exp_1", "model.pt") == record.path

    def test_store_moves_by_default(self, tmp_path):
        source = tmp_path / "model.pt"
        source.write_bytes(b"x")
        registry = ArtifactRegistry(tmp_path / "artifacts")
        registry.store("exp_1", str(source))
        assert not source.exists()

    def test_store_copy(self, tmp_path):
        source = tmp_path / "model.pt"
        source.write_bytes(b"x")
        registry = ArtifactRegistry(tmp_path / "artifacts")
        registry.store("exp_1", str(source), copy=True)
        assert source.exists()

    def test_store_bytes(self, tmp_path):
        registry = ArtifactRegistry(tmp_path / "artifacts")
        record = registry.store_bytes("exp_1", "plot.png", b"png-data", metadata={"type": "plot"})
        assert registry.read_bytes("exp_1", "plot.png") == b"png-data"
        assert record.metadata == {"type": "plot"}

    def test_list_and_delete(self, tmp_path):
        registry = ArtifactRegistry(tmp_path / "artifacts")
        registry.store_bytes("exp_1", "a.bin", b"1")
        registry.store_bytes("exp_2", "b.bin", b"2")
        assert len(registry.list()) == 2
        assert len(registry.list(experiment_id="exp_1")) == 1
        assert registry.delete("exp_1", "a.bin") is True
        assert registry.delete("exp_1", "a.bin") is False

    def test_verify(self, tmp_path):
        registry = ArtifactRegistry(tmp_path / "artifacts")
        registry.store_bytes("exp_1", "a.bin", b"content")
        assert registry.verify("exp_1", "a.bin") is True
        record = registry.get("exp_1", "a.bin")

        with open(record.path, "ab") as handle:
            handle.write(b"tampered")
        assert registry.verify("exp_1", "a.bin") is False

    def test_missing_source(self, tmp_path):
        registry = ArtifactRegistry(tmp_path / "artifacts")
        try:
            registry.store("exp_1", str(tmp_path / "missing"))
            raised = False
        except FileNotFoundError:
            raised = True
        assert raised

    def test_persistence(self, tmp_path):
        root = tmp_path / "artifacts"
        registry = ArtifactRegistry(root)
        registry.store_bytes("exp_1", "a.bin", b"data")
        reloaded = ArtifactRegistry(root)
        assert reloaded.get("exp_1", "a.bin") is not None
