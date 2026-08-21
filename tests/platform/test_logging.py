from __future__ import annotations

import json

import pytest

from cadgenesis.platform.logging import (
    DistributedLogClient,
    HttpSink,
    JsonFileSink,
    LogAggregator,
    structured_record,
    wire_aggregator,
)


class TestLogAggregator:
    def test_emit_and_snapshot(self):
        aggregator = LogAggregator()
        aggregator.emit(structured_record("info", "hello", fields={"step": 1}))
        snapshot = aggregator.snapshot()
        assert len(snapshot) == 1
        assert snapshot[0]["level"] == "INFO"
        assert snapshot[0]["message"] == "hello"

    def test_capacity_bound(self):
        aggregator = LogAggregator(capacity=100)
        for i in range(150):
            aggregator.emit(structured_record("info", f"msg {i}"))
        assert len(aggregator) == 100
        assert aggregator.snapshot()[0]["message"] == "msg 50"

    def test_json_file_sink(self, tmp_path):
        log_file = tmp_path / "logs.jsonl"
        sink = JsonFileSink(str(log_file))
        aggregator = LogAggregator()
        aggregator.add_sink(sink)
        aggregator.emit(structured_record("error", "boom", fields={"code": 500}))
        line = json.loads(log_file.read_text(encoding="utf-8").strip())
        assert line["level"] == "ERROR"
        assert line["code"] == 500

    def test_sink_isolation(self, tmp_path):
        def bad_sink(record):
            raise RuntimeError("sink broke")

        aggregator = LogAggregator()
        aggregator.add_sink(bad_sink)
        aggregator.emit(structured_record("info", "x"))  # must not raise


class TestHttpSink:
    def test_http_sink_connection_error(self):
        sink = HttpSink("http://127.0.0.1:1/logs", timeout=0.2)
        with pytest.raises(OSError):
            sink(structured_record("info", "x"))


class TestDistributedLogClient:
    def test_send_batch_unreachable(self):
        client = DistributedLogClient("http://127.0.0.1:1")
        assert client.send_batch([structured_record("info", "x")]) == -1


class TestWireAggregator:
    def test_wire(self):
        aggregator = LogAggregator()
        emitter = wire_aggregator(aggregator)
        emitter.emit("info", "wired", step=1)
        snapshot = aggregator.snapshot()
        assert snapshot and snapshot[0]["message"] == "wired"
