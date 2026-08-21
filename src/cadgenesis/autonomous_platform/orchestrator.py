"""
Unified Workflow Orchestrator - Scheduling, dependency graph, event orchestration, rollback,
checkpointing, monitoring.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class WorkflowStage(str, Enum):
    MULTIMODAL_UNDERSTANDING = "multimodal_understanding"
    INTENT_EXTRACTION = "intent_extraction"
    REQUIREMENT_GRAPH = "requirement_graph"
    WORLD_MODEL = "world_model"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    MEMORY_RETRIEVAL = "memory_retrieval"
    PLANNER_AGENT = "planner_agent"
    TASK_GRAPH = "task_graph"
    MULTI_AGENT_COLLABORATION = "multi_agent_collaboration"
    NEURO_SYMBOLIC_REASONING = "neuro_symbolic_reasoning"
    CAD_GENERATION = "cad_generation"
    GEOMETRY_VALIDATION = "geometry_validation"
    CONSTRAINT_VALIDATION = "constraint_validation"
    SIMULATION = "simulation"
    MANUFACTURING_ANALYSIS = "manufacturing_analysis"
    OPTIMIZATION = "optimization"
    RELIABILITY_VERIFICATION = "reliability_verification"
    DOCUMENTATION_GENERATION = "documentation_generation"
    DIGITAL_TWIN_VALIDATION = "digital_twin_validation"
    HUMAN_REVIEW = "human_review"
    FINAL_PACKAGE = "final_package"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    ROLLED_BACK = "rolled_back"
    AWAITING_HUMAN = "awaiting_human"


@dataclass
class WorkflowState:
    """State of a workflow execution."""

    workflow_id: str
    prompt: str
    current_stage: WorkflowStage
    status: WorkflowStatus = WorkflowStatus.PENDING
    stage_results: dict[str, Any] = field(default_factory=dict)
    stage_errors: dict[str, str] = field(default_factory=dict)
    checkpoints: dict[str, Any] = field(default_factory=dict)
    human_feedback: dict[str, Any] | None = None
    started_at: float | None = None
    completed_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowStageConfig:
    """Configuration for a workflow stage."""

    stage: WorkflowStage
    function: Callable[[WorkflowState], Any]
    dependencies: list[WorkflowStage] = field(default_factory=list)
    required: bool = True
    timeout: float = 300.0  # seconds
    retry_count: int = 0
    rollback_on_failure: bool = False


class UnifiedWorkflowOrchestrator:
    """Orchestrates the complete autonomous engineering workflow."""

    def __init__(self, checkpoint_dir: str = "./workflow_checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        self._stages: dict[WorkflowStage, WorkflowStageConfig] = {}
        self._workflows: dict[str, WorkflowState] = {}
        self._lock = RLock()
        self._initialize_default_stages()

    def _initialize_default_stages(self) -> None:
        """Initialize the default 22-stage workflow from the specification."""
        stages = [
            WorkflowStage.MULTIMODAL_UNDERSTANDING,
            WorkflowStage.INTENT_EXTRACTION,
            WorkflowStage.REQUIREMENT_GRAPH,
            WorkflowStage.WORLD_MODEL,
            WorkflowStage.KNOWLEDGE_RETRIEVAL,
            WorkflowStage.MEMORY_RETRIEVAL,
            WorkflowStage.PLANNER_AGENT,
            WorkflowStage.TASK_GRAPH,
            WorkflowStage.MULTI_AGENT_COLLABORATION,
            WorkflowStage.NEURO_SYMBOLIC_REASONING,
            WorkflowStage.CAD_GENERATION,
            WorkflowStage.GEOMETRY_VALIDATION,
            WorkflowStage.CONSTRAINT_VALIDATION,
            WorkflowStage.SIMULATION,
            WorkflowStage.MANUFACTURING_ANALYSIS,
            WorkflowStage.OPTIMIZATION,
            WorkflowStage.RELIABILITY_VERIFICATION,
            WorkflowStage.DOCUMENTATION_GENERATION,
            WorkflowStage.DIGITAL_TWIN_VALIDATION,
            WorkflowStage.HUMAN_REVIEW,
            WorkflowStage.FINAL_PACKAGE,
        ]

        for i, stage in enumerate(stages):
            deps = [stages[i - 1]] if i > 0 else []
            self.register_stage(
                WorkflowStageConfig(
                    stage=stage,
                    function=self._default_stage_function(stage),
                    dependencies=deps,
                )
            )

    def _default_stage_function(self, stage: WorkflowStage) -> Callable[[WorkflowState], Any]:
        """Default no-op function for a stage."""

        def fn(state: WorkflowState) -> dict[str, Any]:
            return {"stage": stage.value, "status": "completed", "timestamp": time.time()}

        return fn

    def register_stage(self, config: WorkflowStageConfig) -> None:
        with self._lock:
            self._stages[config.stage] = config

    def execute_workflow(
        self,
        prompt: str,
        initial_context: dict[str, Any] | None = None,
    ) -> WorkflowState:
        """Execute the complete workflow."""
        workflow_id = str(uuid.uuid4())
        state = WorkflowState(
            workflow_id=workflow_id,
            prompt=prompt,
            current_stage=WorkflowStage.MULTIMODAL_UNDERSTANDING,
            status=WorkflowStatus.RUNNING,
            started_at=time.time(),
            metadata=initial_context or {},
        )

        with self._lock:
            self._workflows[workflow_id] = state

        try:
            self._run_workflow(state)
        except Exception as e:
            state.status = WorkflowStatus.FAILED
            state.metadata["error"] = str(e)

        state.completed_at = time.time()
        return state

    def _run_workflow(self, state: WorkflowState) -> None:
        """Run workflow stages in dependency order."""
        # Topological sort of stages
        stage_order = self._get_execution_order()

        for stage in stage_order:
            config = self._stages[stage]
            state.current_stage = stage

            try:
                # Check dependencies
                for dep in config.dependencies:
                    if dep.value not in state.stage_results:
                        raise RuntimeError(f"Dependency {dep.value} not satisfied")

                # Execute stage
                result = config.function(state)
                state.stage_results[stage.value] = result

                # Checkpoint
                self._save_checkpoint(state, stage)

            except Exception as e:
                state.stage_errors[stage.value] = str(e)
                if config.required:
                    state.status = WorkflowStatus.FAILED
                    if config.rollback_on_failure:
                        self._rollback(state, stage)
                    raise
                else:
                    # Optional stage - continue
                    state.stage_results[stage.value] = {"error": str(e), "skipped": True}

        state.status = WorkflowStatus.COMPLETED

    def _get_execution_order(self) -> list[WorkflowStage]:
        """Get stages in dependency order."""
        in_degree: defaultdict[WorkflowStage, int] = defaultdict(int)
        graph = defaultdict(list)

        for stage, config in self._stages.items():
            for dep in config.dependencies:
                graph[dep].append(stage)
                in_degree[stage] += 1
            if stage not in in_degree:
                in_degree[stage] = 0

        queue = deque([s for s, d in in_degree.items() if d == 0])
        order = []

        while queue:
            stage = queue.popleft()
            order.append(stage)
            for next_stage in graph[stage]:
                in_degree[next_stage] -= 1
                if in_degree[next_stage] == 0:
                    queue.append(next_stage)

        if len(order) != len(self._stages):
            raise ValueError("Workflow has circular dependencies")

        return order

    def _save_checkpoint(self, state: WorkflowState, stage: WorkflowStage) -> None:
        checkpoint = {
            "workflow_id": state.workflow_id,
            "stage": stage.value,
            "stage_results": state.stage_results.copy(),
            "timestamp": time.time(),
        }
        state.checkpoints[stage.value] = checkpoint

    def _rollback(self, state: WorkflowState, failed_stage: WorkflowStage) -> None:
        """Rollback to last successful checkpoint."""
        state.status = WorkflowStatus.ROLLED_BACK
        # In production, would restore from checkpoint and retry

    def pause_workflow(self, workflow_id: str) -> bool:
        with self._lock:
            state = self._workflows.get(workflow_id)
            if not state or state.status != WorkflowStatus.RUNNING:
                return False
            state.status = WorkflowStatus.PAUSED
            return True

    def resume_workflow(self, workflow_id: str) -> bool:
        with self._lock:
            state = self._workflows.get(workflow_id)
            if not state or state.status != WorkflowStatus.PAUSED:
                return False
            state.status = WorkflowStatus.RUNNING
            # Would resume from last checkpoint
            return True

    def await_human_review(self, workflow_id: str, feedback: dict[str, Any]) -> bool:
        """Provide human feedback and continue."""
        with self._lock:
            state = self._workflows.get(workflow_id)
            if not state or state.status != WorkflowStatus.AWAITING_HUMAN:
                return False
            state.human_feedback = feedback
            state.status = WorkflowStatus.RUNNING
            return True

    def get_workflow(self, workflow_id: str) -> WorkflowState | None:
        with self._lock:
            return self._workflows.get(workflow_id)

    def list_workflows(self, status: WorkflowStatus | None = None) -> list[WorkflowState]:
        with self._lock:
            workflows = list(self._workflows.values())
            if status:
                workflows = [w for w in workflows if w.status == status]
            return workflows
