from __future__ import annotations

from cadgenesis.serving.batching import BatchScheduler, DynamicBatcher


class TestBatchScheduler:
    def test_group_partitions(self):
        requests = [
            ("cad", 10),
            ("cad", 8),
            ("cad", 6),
            ("cad", 4),
            ("cad", 2),
            ("cad", 1),
            ("cad", 3),
        ]
        batches = BatchScheduler.group(requests, max_batch=4)
        assert [len(b) for b in batches] == [4, 3]

    def test_group_by_model(self):
        requests = [("a", 4), ("b", 4), ("a", 4), ("b", 4)]
        batches = BatchScheduler.group(requests, max_batch=2)
        models = {b[0][0] for b in batches}
        assert models == {"a", "b"}

    def test_group_handles_empty(self):
        assert BatchScheduler.group([], max_batch=4) == []

    def test_padded_lengths(self):
        batch = [("cad", 4), ("cad", 8), ("cad", 2)]
        assert BatchScheduler.padded_lengths(batch) == [8, 8, 8]


class TestDynamicBatcher:
    def test_batching_collects_requests(self):
        batch_sizes = []

        def dispatch(payloads):
            batch_sizes.append(len(payloads))
            return [f"ok-{p['i']}" for p in payloads]

        batcher = DynamicBatcher(dispatch, max_batch=4, max_wait_seconds=0.05)
        futures = [batcher.submit({"i": i}, max_len=8) for i in range(9)]
        results = [f.result(timeout=10) for f in futures]
        batcher.shutdown()
        assert results == [f"ok-{i}" for i in range(9)]
        assert batch_sizes == [4, 4, 1]

    def test_submit_after_shutdown_raises(self):
        batcher = DynamicBatcher(lambda p: ["x"] * len(p), max_batch=4)
        batcher.shutdown()
        try:
            batcher.submit({"i": 1}, max_len=8)
            raised = False
        except RuntimeError:
            raised = True
        assert raised

    def test_pending_count(self):
        batcher = DynamicBatcher(lambda p: ["x"] * len(p), max_batch=4, max_wait_seconds=0.2)
        batcher.submit({"i": 1}, max_len=8)
        assert batcher.pending_count() >= 0
        batcher.shutdown()
