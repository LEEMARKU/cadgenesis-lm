"""
Human Approval Pipeline - Research Idea → Experiment Plan → Execution → Benchmark → Statistical
Validation → Peer Review → Human Approval → Production Promotion.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class ApprovalStage(str, Enum):
    RESEARCH_IDEA = "research_idea"
    EXPERIMENT_PLAN = "experiment_plan"
    EXECUTION = "execution"
    BENCHMARK = "benchmark"
    STATISTICAL_VALIDATION = "statistical_validation"
    PEER_REVIEW = "peer_review"
    HUMAN_APPROVAL = "human_approval"
    PRODUCTION_PROMOTION = "production_promotion"
    REJECTED = "rejected"


class ApprovalDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"
    NEED_MORE_INFO = "need_more_info"


@dataclass
class ApprovalRequest:
    """A request for approval at a specific stage."""

    request_id: str
    experiment_id: str
    stage: ApprovalStage
    title: str
    description: str
    payload: dict[str, Any]  # Stage-specific data
    requested_by: str
    requested_at: float = field(default_factory=time.time)
    reviewers: list[str] = field(default_factory=list)
    required_approvals: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ApprovalRecord:
    """Record of an approval decision."""

    record_id: str
    request_id: str
    reviewer: str
    decision: ApprovalDecision
    comments: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class ApprovalState:
    """Current state of an experiment in the approval pipeline."""

    experiment_id: str
    current_stage: ApprovalStage
    stage_history: list[ApprovalRecord] = field(default_factory=list)
    pending_request: ApprovalRequest | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class HumanApprovalPipeline:
    """Manages the human approval pipeline for research experiments."""

    def __init__(self):
        self._states: dict[str, ApprovalState] = {}
        self._requests: dict[str, ApprovalRequest] = {}
        self._lock = RLock()

        # Define stage progression
        self._stages = [
            ApprovalStage.RESEARCH_IDEA,
            ApprovalStage.EXPERIMENT_PLAN,
            ApprovalStage.EXECUTION,
            ApprovalStage.BENCHMARK,
            ApprovalStage.STATISTICAL_VALIDATION,
            ApprovalStage.PEER_REVIEW,
            ApprovalStage.HUMAN_APPROVAL,
            ApprovalStage.PRODUCTION_PROMOTION,
        ]

    def create_experiment(self, experiment_id: str) -> ApprovalState:
        """Initialize approval pipeline for an experiment."""
        with self._lock:
            state = ApprovalState(
                experiment_id=experiment_id,
                current_stage=ApprovalStage.RESEARCH_IDEA,
            )
            self._states[experiment_id] = state
            return state

    def get_state(self, experiment_id: str) -> ApprovalState | None:
        with self._lock:
            return self._states.get(experiment_id)

    def submit_for_approval(
        self,
        experiment_id: str,
        stage: ApprovalStage,
        title: str,
        description: str,
        payload: dict[str, Any],
        requested_by: str,
        reviewers: list[str] | None = None,
        required_approvals: int = 1,
    ) -> ApprovalRequest:
        """Submit experiment for approval at a specific stage."""
        with self._lock:
            state = self._states.get(experiment_id)
            if not state:
                raise ValueError(f"Experiment {experiment_id} not in pipeline")

            # Validate stage progression
            if self._stages.index(stage) < self._stages.index(state.current_stage):
                raise ValueError("Cannot go back to earlier stage")

            request = ApprovalRequest(
                request_id=str(uuid.uuid4()),
                experiment_id=experiment_id,
                stage=stage,
                title=title,
                description=description,
                payload=payload,
                requested_by=requested_by,
                reviewers=reviewers or [],
                required_approvals=required_approvals,
            )

            self._requests[request.request_id] = request
            state.pending_request = request
            state.updated_at = time.time()

            return request

    def record_decision(
        self,
        request_id: str,
        reviewer: str,
        decision: ApprovalDecision,
        comments: str = "",
    ) -> tuple[bool, ApprovalState | None]:
        """Record a reviewer's decision."""
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return False, None

            record = ApprovalRecord(
                record_id=str(uuid.uuid4()),
                request_id=request_id,
                reviewer=reviewer,
                decision=decision,
                comments=comments,
            )

            state = self._states.get(request.experiment_id)
            if not state:
                return False, None

            state.stage_history.append(record)

            # Check if we have enough approvals
            approvals = sum(
                1
                for r in state.stage_history
                if r.request_id == request_id and r.decision == ApprovalDecision.APPROVE
            )
            rejections = sum(
                1
                for r in state.stage_history
                if r.request_id == request_id and r.decision == ApprovalDecision.REJECT
            )

            if rejections > 0:
                # Any rejection rejects the request
                state.current_stage = ApprovalStage.REJECTED
                state.pending_request = None
                state.updated_at = time.time()
                return True, state

            if approvals >= request.required_approvals:
                # Advance to next stage
                current_idx = self._stages.index(state.current_stage)
                if current_idx + 1 < len(self._stages):
                    state.current_stage = self._stages[current_idx + 1]
                state.pending_request = None
                state.updated_at = time.time()
                return True, state

            return True, state

    def request_changes(
        self,
        request_id: str,
        reviewer: str,
        comments: str,
    ) -> bool:
        """Request changes to the submission."""
        with self._lock:
            request = self._requests.get(request_id)
            if not request:
                return False

            record = ApprovalRecord(
                record_id=str(uuid.uuid4()),
                request_id=request_id,
                reviewer=reviewer,
                decision=ApprovalDecision.REQUEST_CHANGES,
                comments=comments,
            )

            state = self._states.get(request.experiment_id)
            if not state:
                return False

            state.stage_history.append(record)
            state.updated_at = time.time()
            return True

    def is_approved_for_production(self, experiment_id: str) -> bool:
        """Check if experiment has been approved for production."""
        with self._lock:
            state = self._states.get(experiment_id)
            return state.current_stage == ApprovalStage.PRODUCTION_PROMOTION if state else False

    def get_pending_requests(self, reviewer: str | None = None) -> list[ApprovalRequest]:
        with self._lock:
            requests = list(self._requests.values())
            if reviewer:
                requests = [r for r in requests if reviewer in r.reviewers]
            return [
                r
                for r in requests
                if r.request_id
                in [
                    s.pending_request.request_id for s in self._states.values() if s.pending_request
                ]
            ]

    def get_approval_history(self, experiment_id: str) -> list[ApprovalRecord]:
        with self._lock:
            state = self._states.get(experiment_id)
            return state.stage_history if state else []
