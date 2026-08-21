"""
Safe Promotion Pipeline - Experimental Module → Benchmark → Validation → Regression Tests →
Human Approval → Production.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class PromotionStage(str, Enum):
    EXPERIMENTAL = "experimental"
    BENCHMARK = "benchmark"
    VALIDATION = "validation"
    REGRESSION_TEST = "regression_test"
    HUMAN_APPROVAL = "human_approval"
    PRODUCTION = "production"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class PromotionDecision(str, Enum):
    PROMOTE = "promote"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"
    NEEDS_MORE_INFO = "needs_more_info"


@dataclass
class PromotionCriteria:
    """Criteria for promotion at each stage."""

    min_benchmark_score: float = 0.8
    max_regression_tolerance: float = 0.05
    required_validations: list[str] = field(default_factory=list)
    required_approvals: int = 1
    approvers: list[str] = field(default_factory=list)


@dataclass
class PromotionRecord:
    """Record of a promotion decision."""

    record_id: str
    experiment_id: str
    stage: PromotionStage
    decision: PromotionDecision
    decided_by: str
    timestamp: float
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromotionPipelineState:
    """Current state of an experiment in the promotion pipeline."""

    experiment_id: str
    current_stage: PromotionStage
    stage_history: list[PromotionRecord] = field(default_factory=list)
    criteria: PromotionCriteria = field(default_factory=PromotionCriteria)
    benchmark_results: dict[str, float] = field(default_factory=dict)
    validation_results: dict[str, bool] = field(default_factory=dict)
    regression_results: dict[str, bool] = field(default_factory=dict)
    approvals: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class SafePromotionPipeline:
    """Manages the safe promotion of experimental modules to production."""

    def __init__(self):
        self._pipelines: dict[str, PromotionPipelineState] = {}
        self._lock = RLock()

    def create_pipeline(
        self,
        experiment_id: str,
        criteria: PromotionCriteria | None = None,
    ) -> PromotionPipelineState:
        """Initialize a promotion pipeline for an experiment."""
        with self._lock:
            state = PromotionPipelineState(
                experiment_id=experiment_id,
                current_stage=PromotionStage.EXPERIMENTAL,
                criteria=criteria or PromotionCriteria(),
            )
            self._pipelines[experiment_id] = state
            return state

    def get_pipeline(self, experiment_id: str) -> PromotionPipelineState | None:
        with self._lock:
            return self._pipelines.get(experiment_id)

    def advance_stage(
        self,
        experiment_id: str,
        target_stage: PromotionStage,
        decision: PromotionDecision,
        decided_by: str,
        rationale: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[bool, PromotionPipelineState | None]:
        """Advance experiment to the next stage with a decision."""
        with self._lock:
            state = self._pipelines.get(experiment_id)
            if not state:
                return False, None

            # Validate stage progression
            stages = [
                PromotionStage.EXPERIMENTAL,
                PromotionStage.BENCHMARK,
                PromotionStage.VALIDATION,
                PromotionStage.REGRESSION_TEST,
                PromotionStage.HUMAN_APPROVAL,
                PromotionStage.PRODUCTION,
            ]

            try:
                current_idx = stages.index(state.current_stage)
                target_idx = stages.index(target_stage)
            except ValueError:
                return False, None

            if target_idx <= current_idx and target_stage not in (
                PromotionStage.REJECTED,
                PromotionStage.ROLLED_BACK,
            ):
                return False, None

            # Record decision
            record = PromotionRecord(
                record_id=str(uuid.uuid4()),
                experiment_id=experiment_id,
                stage=target_stage,
                decision=decision,
                decided_by=decided_by,
                timestamp=time.time(),
                rationale=rationale,
                metadata=metadata or {},
            )
            state.stage_history.append(record)
            state.current_stage = target_stage
            state.updated_at = time.time()

            return True, state

    def submit_benchmark_results(
        self,
        experiment_id: str,
        results: dict[str, float],
    ) -> bool:
        with self._lock:
            state = self._pipelines.get(experiment_id)
            if not state:
                return False
            state.benchmark_results = results
            state.updated_at = time.time()
            return True

    def submit_validation_results(
        self,
        experiment_id: str,
        results: dict[str, bool],
    ) -> bool:
        with self._lock:
            state = self._pipelines.get(experiment_id)
            if not state:
                return False
            state.validation_results = results
            state.updated_at = time.time()
            return True

    def submit_regression_results(
        self,
        experiment_id: str,
        results: dict[str, bool],
    ) -> bool:
        with self._lock:
            state = self._pipelines.get(experiment_id)
            if not state:
                return False
            state.regression_results = results
            state.updated_at = time.time()
            return True

    def submit_approval(
        self,
        experiment_id: str,
        approver: str,
        approved: bool,
        comments: str = "",
    ) -> bool:
        with self._lock:
            state = self._pipelines.get(experiment_id)
            if not state:
                return False
            state.approvals.append(
                {
                    "approver": approver,
                    "approved": approved,
                    "comments": comments,
                    "timestamp": time.time(),
                }
            )
            state.updated_at = time.time()
            return True

    def check_promotion_ready(self, experiment_id: str) -> tuple[bool, list[str]]:
        """Check if experiment meets all criteria for promotion to production."""
        with self._lock:
            state = self._pipelines.get(experiment_id)
            if not state:
                return False, ["Pipeline not found"]

            errors = []

            # Check benchmark criteria
            if state.current_stage == PromotionStage.BENCHMARK:
                for metric, value in state.benchmark_results.items():
                    if value < state.criteria.min_benchmark_score:
                        errors.append(
                            f"Benchmark {metric}={value:.3f} below threshold "
                            f"{state.criteria.min_benchmark_score}"
                        )

            # Check validation criteria
            if state.current_stage == PromotionStage.VALIDATION:
                for val in state.criteria.required_validations:
                    if not state.validation_results.get(val, False):
                        errors.append(f"Required validation '{val}' not passed")

            # Check regression criteria
            if state.current_stage == PromotionStage.REGRESSION_TEST:
                for metric, passed in state.regression_results.items():
                    if not passed:
                        errors.append(f"Regression test '{metric}' failed")

            # Check approval criteria
            if state.current_stage == PromotionStage.HUMAN_APPROVAL:
                approval_count = sum(1 for a in state.approvals if a["approved"])
                if approval_count < state.criteria.required_approvals:
                    errors.append(
                        f"Insufficient approvals: {approval_count}/"
                        f"{state.criteria.required_approvals}"
                    )

            return len(errors) == 0, errors

    def list_pipelines(self, stage: PromotionStage | None = None) -> list[PromotionPipelineState]:
        with self._lock:
            pipelines = list(self._pipelines.values())
            if stage:
                pipelines = [p for p in pipelines if p.current_stage == stage]
            return pipelines

    def get_promotion_history(self, experiment_id: str) -> list[PromotionRecord]:
        with self._lock:
            state = self._pipelines.get(experiment_id)
            return state.stage_history if state else []
