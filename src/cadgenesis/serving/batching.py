"""
cadgenesis.serving.batching
===========================
Dynamic batching for the CADGenesis serving stack.

- ``DynamicBatcher``: merges concurrent requests with similar max_len into
  batches (max_batch, max_wait_ms), returns futures; thread-safe
- ``BatchScheduler``: pad-and-group policy used by the HTTP/gRPC handlers
- All pure Python; engine dispatch is injected by the caller
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cadgenesis.serving.batching")

BatchFn = Callable[[list[Any]], list[Any]]


@dataclass(order=True)
class _Pending:
    """Priority by max_len (shortest first) for packing efficiency."""

    max_len: int
    created: float = field(compare=False)
    payload: Any = field(compare=False)
    future: Future = field(compare=False)


class DynamicBatcher:
    """Coalesces requests arriving within a small window into one engine call.

    ``dispatch`` receives the list of request payloads and must return a list
    of results aligned with the input order.
    """

    def __init__(
        self,
        dispatch: BatchFn,
        max_batch: int = 8,
        max_wait_seconds: float = 0.02,
        poll_seconds: float = 0.001,
    ) -> None:
        self.dispatch = dispatch
        self.max_batch = max(1, max_batch)
        self.max_wait = max(0.0, max_wait_seconds)
        self.poll = max(0.0001, poll_seconds)
        self._pending: list[_Pending] = []
        self._lock = threading.Condition()
        self._stopped = False
        self._worker = threading.Thread(target=self._run, name="dynamic-batcher", daemon=True)
        self._worker.start()

    def submit(self, payload: Any, max_len: int = 128) -> Future:
        if self._stopped:
            raise RuntimeError("batcher is shut down")
        future: Future = Future()
        with self._lock:
            self._pending.append(
                _Pending(max_len=max_len, created=time.monotonic(), payload=payload, future=future)
            )
            self._lock.notify_all()
        return future

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._stopped:
                    return
                if not self._pending:
                    self._lock.wait()
                    continue
                first = self._pending[0]
                elapsed = time.monotonic() - first.created
                while len(self._pending) < self.max_batch and elapsed < self.max_wait:
                    if self._lock.wait(timeout=self.poll) and self._stopped:
                        return
                    elapsed = time.monotonic() - first.created
                batch = self._pending[: self.max_batch]
                del self._pending[: len(batch)]
            self._execute(batch)

    def _execute(self, batch: list[_Pending]) -> None:
        error: BaseException | None = None
        results: list[Any] | None = None
        try:
            results = self.dispatch([p.payload for p in batch])
        except BaseException as exc:
            error = exc
        if error is None and (results is None or len(results) != len(batch)):
            error = RuntimeError(
                f"dispatch returned {len(results) if results is not None else 0} "
                f"results for {len(batch)} requests"
            )
        if error is None:
            assert results is not None
            for pending, result in zip(batch, results, strict=True):
                pending.future.set_result(result)
        else:
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(error)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def shutdown(self) -> None:
        """Stop the worker thread; pending futures are resolved with an error."""
        with self._lock:
            self._stopped = True
            pending = self._pending
            self._pending = []
            self._lock.notify_all()
        for item in pending:
            item.future.set_exception(RuntimeError("batcher shut down"))
        if self._worker is not None and self._worker is not threading.current_thread():
            self._worker.join(timeout=2.0)


class BatchScheduler:
    """Deterministic grouping policy: same-model requests padded to max_len."""

    @staticmethod
    def group(
        requests: Sequence[tuple[str, int]],
        max_batch: int = 8,
    ) -> list[list[tuple[str, int]]]:
        """Group ``(model_name, max_len)`` pairs into batches.

        Groups by model first, sorts by max_len, packs greedily up to
        ``max_batch`` items.
        """
        by_model: dict[str, list[tuple[str, int]]] = {}
        for name, length in requests:
            by_model.setdefault(name, []).append((name, length))
        batches: list[list[tuple[str, int]]] = []
        for group in by_model.values():
            group.sort(key=lambda item: item[1])
            batches.extend(
                group[index : index + max_batch] for index in range(0, len(group), max_batch)
            )
        return batches

    @staticmethod
    def padded_lengths(batch: Sequence[tuple[str, int]]) -> list[int]:
        """Pad every request in a batch to the batch's max_len."""
        target = max((length for _, length in batch), default=0)
        return [target for _ in batch]


__all__ = ["BatchScheduler", "DynamicBatcher"]
