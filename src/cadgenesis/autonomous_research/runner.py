"""
Automated Experiment Runner - Distributed execution, reproducible execution, isolation,
checkpoint recovery.
"""

from __future__ import annotations

import pickle
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class ExperimentExecution:
    """An executing experiment."""

    execution_id: str
    graph_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    node_results: dict[str, Any] = field(default_factory=dict)
    node_errors: dict[str, str] = field(default_factory=dict)
    node_status: dict[str, str] = field(default_factory=dict)
    started_at: float | None = None
    completed_at: float | None = None
    checkpoints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class AutomatedExperimentRunner:
    """Runs experiment graphs with isolation, checkpointing, and recovery."""

    def __init__(self, max_workers: int = 4, checkpoint_dir: str = "./experiment_checkpoints"):
        self.max_workers = max_workers
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._executions: dict[str, ExperimentExecution] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = RLock()

    def run_experiment(
        self,
        graph_id: str,
        graph: Any,  # ExperimentGraph
        node_functions: dict[str, Callable],
        resume_from: str | None = None,
    ) -> ExperimentExecution:
        """Run an experiment graph."""
        execution_id = str(uuid.uuid4())
        execution = ExperimentExecution(execution_id=execution_id, graph_id=graph_id)

        with self._lock:
            self._executions[execution_id] = execution

        # Submit to executor
        future = self._executor.submit(
            self._run_graph,
            execution,
            graph,
            node_functions,
            resume_from,
        )
        execution.metadata["future"] = future

        return execution

    def _run_graph(
        self,
        execution: ExperimentExecution,
        graph: Any,
        node_functions: dict[str, Callable],
        resume_from: str | None,
    ) -> None:
        execution.status = ExecutionStatus.RUNNING
        execution.started_at = time.time()

        try:
            # Get execution order
            order = graph.get_execution_order()

            # Resume from checkpoint if specified
            start_idx = 0
            if resume_from and resume_from in execution.checkpoints:
                # Find index of resume_from node
                for i, node_id in enumerate(order):
                    if node_id == resume_from:
                        start_idx = i + 1
                        break

            # Execute nodes in order
            for node_id in order[start_idx:]:
                if execution.status == ExecutionStatus.CANCELLED:
                    break

                node = graph.nodes[node_id]
                execution.node_status[node_id] = ExecutionStatus.RUNNING.value

                try:
                    # Get function for this node type
                    func = node_functions.get(node.node_type.value)
                    if not func:
                        raise ValueError(f"No function for node type: {node.node_type.value}")

                    # Prepare inputs from dependencies
                    inputs = {}
                    for dep_id in node.dependencies:
                        if dep_id in execution.node_results:
                            inputs[dep_id] = execution.node_results[dep_id]

                    # Run node
                    result = func(node.config, inputs)

                    execution.node_results[node_id] = result
                    execution.node_status[node_id] = ExecutionStatus.COMPLETED.value

                    # Checkpoint
                    self._save_checkpoint(execution, node_id, result)

                except Exception as e:
                    execution.node_status[node_id] = ExecutionStatus.FAILED.value
                    execution.node_errors[node_id] = str(e)
                    execution.status = ExecutionStatus.FAILED
                    break

            if execution.status != ExecutionStatus.FAILED:
                execution.status = ExecutionStatus.COMPLETED

        except Exception as e:
            execution.status = ExecutionStatus.FAILED
            execution.metadata["error"] = str(e)

        finally:
            execution.completed_at = time.time()

    def _save_checkpoint(self, execution: ExperimentExecution, node_id: str, result: Any) -> None:
        """Save checkpoint for recovery."""
        checkpoint_path = self.checkpoint_dir / f"{execution.execution_id}_{node_id}.pkl"
        checkpoint_data = {
            "execution_id": execution.execution_id,
            "node_id": node_id,
            "result": result,
            "timestamp": time.time(),
        }
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        execution.checkpoints[node_id] = str(checkpoint_path)

    def get_execution(self, execution_id: str) -> ExperimentExecution | None:
        with self._lock:
            return self._executions.get(execution_id)

    def cancel_execution(self, execution_id: str) -> bool:
        with self._lock:
            execution = self._executions.get(execution_id)
            if not execution:
                return False
            execution.status = ExecutionStatus.CANCELLED
            future = execution.metadata.get("future")
            if future:
                future.cancel()
            return True

    def pause_execution(self, execution_id: str) -> bool:
        with self._lock:
            execution = self._executions.get(execution_id)
            if not execution or execution.status != ExecutionStatus.RUNNING:
                return False
            execution.status = ExecutionStatus.PAUSED
            return True

    def resume_execution(self, execution_id: str) -> bool:
        with self._lock:
            execution = self._executions.get(execution_id)
            if not execution or execution.status != ExecutionStatus.PAUSED:
                return False
            execution.status = ExecutionStatus.RUNNING
            return True

    def list_executions(self, status: ExecutionStatus | None = None) -> list[ExperimentExecution]:
        with self._lock:
            executions = list(self._executions.values())
            if status:
                executions = [e for e in executions if e.status == status]
            return executions

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
