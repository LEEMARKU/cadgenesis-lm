"""Experiment Planner - Experiment graph, execution plan, scheduling, dependency resolution."""

from __future__ import annotations

import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class ExperimentNodeType(str, Enum):
    SETUP = "setup"
    TRAIN = "train"
    EVALUATE = "evaluate"
    ANALYZE = "analyze"
    COMPARE = "compare"
    REPORT = "report"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ExperimentNode:
    """A node in the experiment graph."""

    node_id: str
    name: str
    node_type: ExperimentNodeType
    function: Callable | None = None
    config: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)  # node_ids
    estimated_duration: float = 0.0  # seconds
    required_resources: dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.PENDING
    result: Any | None = None
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None


@dataclass
class ExperimentGraph:
    """A directed acyclic graph of experiment nodes."""

    graph_id: str
    name: str
    nodes: dict[str, ExperimentNode] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)  # (from, to)

    def add_node(self, node: ExperimentNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, from_node: str, to_node: str) -> None:
        self.edges.append((from_node, to_node))
        self.nodes[to_node].dependencies.append(from_node)

    def get_execution_order(self) -> list[str]:
        """Return nodes in topological order."""
        # Kahn's algorithm
        in_degree: defaultdict[str, int] = defaultdict(int)
        for u, v in self.edges:
            in_degree[v] += 1
            if u not in in_degree:
                in_degree[u] = 0

        queue = deque([n for n, d in in_degree.items() if d == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for u, v in self.edges:
                if u == node:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)

        if len(order) != len(self.nodes):
            raise ValueError("Graph has cycles")

        return order

    def get_parallelizable_groups(self) -> list[list[str]]:
        """Group nodes that can run in parallel."""
        order = self.get_execution_order()
        groups: list[list[str]] = []
        current_group: list[str] = []

        for node_id in order:
            node = self.nodes[node_id]
            # Check if all dependencies are in previous groups
            deps_done = all(any(dep in g for g in groups) for dep in node.dependencies)
            if deps_done and current_group:
                groups.append(current_group)
                current_group = []
            current_group.append(node_id)

        if current_group:
            groups.append(current_group)

        return groups


class ExperimentPlanner:
    """Plans experiment execution graphs and schedules."""

    def __init__(self):
        self._graphs: dict[str, ExperimentGraph] = {}
        self._lock = RLock()

    def create_graph(self, name: str) -> ExperimentGraph:
        graph = ExperimentGraph(graph_id=str(uuid.uuid4()), name=name)
        with self._lock:
            self._graphs[graph.graph_id] = graph
        return graph

    def get_graph(self, graph_id: str) -> ExperimentGraph | None:
        with self._lock:
            return self._graphs.get(graph_id)

    def create_training_experiment(
        self,
        name: str,
        model_config: dict[str, Any],
        train_config: dict[str, Any],
        eval_config: dict[str, Any],
    ) -> ExperimentGraph:
        graph = self.create_graph(name)

        # Setup node
        setup = ExperimentNode(
            node_id=str(uuid.uuid4()),
            name="setup",
            node_type=ExperimentNodeType.SETUP,
            config={"model": model_config},
            estimated_duration=30,
        )
        graph.add_node(setup)

        # Train node
        train = ExperimentNode(
            node_id=str(uuid.uuid4()),
            name="train",
            node_type=ExperimentNodeType.TRAIN,
            config=train_config,
            dependencies=[setup.node_id],
            estimated_duration=train_config.get("estimated_time", 3600),
            required_resources={"gpu": train_config.get("gpus", 1)},
        )
        graph.add_node(train)

        # Evaluate node
        evaluate = ExperimentNode(
            node_id=str(uuid.uuid4()),
            name="evaluate",
            node_type=ExperimentNodeType.EVALUATE,
            config=eval_config,
            dependencies=[train.node_id],
            estimated_duration=300,
        )
        graph.add_node(evaluate)

        # Report node
        report = ExperimentNode(
            node_id=str(uuid.uuid4()),
            name="report",
            node_type=ExperimentNodeType.REPORT,
            config={},
            dependencies=[evaluate.node_id],
            estimated_duration=60,
        )
        graph.add_node(report)

        return graph

    def create_comparison_experiment(
        self,
        name: str,
        model_configs: list[dict[str, Any]],
        train_config: dict[str, Any],
        eval_config: dict[str, Any],
    ) -> ExperimentGraph:
        graph = self.create_graph(name)

        setup_nodes = []
        train_nodes = []
        eval_nodes = []

        for i, model_config in enumerate(model_configs):
            setup = ExperimentNode(
                node_id=str(uuid.uuid4()),
                name=f"setup_{i}",
                node_type=ExperimentNodeType.SETUP,
                config={"model": model_config, "variant": i},
                estimated_duration=30,
            )
            graph.add_node(setup)
            setup_nodes.append(setup)

            train = ExperimentNode(
                node_id=str(uuid.uuid4()),
                name=f"train_{i}",
                node_type=ExperimentNodeType.TRAIN,
                config={**train_config, "variant": i},
                dependencies=[setup.node_id],
                estimated_duration=train_config.get("estimated_time", 3600),
                required_resources={"gpu": train_config.get("gpus", 1)},
            )
            graph.add_node(train)
            train_nodes.append(train)

            evaluate = ExperimentNode(
                node_id=str(uuid.uuid4()),
                name=f"evaluate_{i}",
                node_type=ExperimentNodeType.EVALUATE,
                config={**eval_config, "variant": i},
                dependencies=[train.node_id],
                estimated_duration=300,
            )
            graph.add_node(evaluate)
            eval_nodes.append(evaluate)

        # Compare node
        compare = ExperimentNode(
            node_id=str(uuid.uuid4()),
            name="compare",
            node_type=ExperimentNodeType.COMPARE,
            config={"variants": len(model_configs)},
            dependencies=[e.node_id for e in eval_nodes],
            estimated_duration=120,
        )
        graph.add_node(compare)

        # Report node
        report = ExperimentNode(
            node_id=str(uuid.uuid4()),
            name="report",
            node_type=ExperimentNodeType.REPORT,
            config={},
            dependencies=[compare.node_id],
            estimated_duration=60,
        )
        graph.add_node(report)

        return graph

    def estimate_total_time(self, graph: ExperimentGraph) -> float:
        """Estimate total execution time considering parallelization."""
        groups = graph.get_parallelizable_groups()
        total = 0.0
        for group in groups:
            group_time = max(graph.nodes[n].estimated_duration for n in group)
            total += group_time
        return total
