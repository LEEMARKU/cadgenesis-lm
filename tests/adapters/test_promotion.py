"""Tests for cadgenesis.adapters.promotion."""

import pytest

from cadgenesis.adapters.manager import AdapterMetadata
from cadgenesis.adapters.promotion import (
    PROMOTED_STATUS,
    AdapterPromotion,
    PromotionCriteria,
)


def test_approve_when_meets_thresholds():
    meta = AdapterMetadata(adapter_id="aero_v1", domain="aerospace")
    decision = AdapterPromotion().evaluate(
        meta, {"accuracy": 0.9, "stability": 0.95, "samples": 1.0}
    )
    assert decision.approved
    assert decision.score == pytest.approx(0.5 * 0.9 + 0.3 * 0.95)
    assert len(decision.reasons) == 3


def test_reject_low_accuracy():
    meta = AdapterMetadata(adapter_id="aero_v1", domain="aerospace")
    decision = AdapterPromotion().evaluate(meta, {"accuracy": 0.8, "stability": 0.95})
    assert not decision.approved
    assert any("accuracy" in reason for reason in decision.reasons)
    assert any("stability" in reason for reason in decision.reasons)


def test_reject_low_stability():
    meta = AdapterMetadata(adapter_id="aero_v1", domain="aerospace")
    decision = AdapterPromotion().evaluate(meta, {"accuracy": 0.9, "stability": 0.8})
    assert not decision.approved


def test_reject_excessive_drift():
    meta = AdapterMetadata(adapter_id="aero_v1", domain="aerospace")
    decision = AdapterPromotion().evaluate(meta, {"accuracy": 0.9, "stability": 0.95, "drift": 0.2})
    assert not decision.approved
    assert any("drift" in reason for reason in decision.reasons)


def test_drift_within_tolerance_approved():
    meta = AdapterMetadata(adapter_id="aero_v1", domain="aerospace")
    decision = AdapterPromotion().evaluate(
        meta, {"accuracy": 0.9, "stability": 0.95, "drift": 0.05, "samples": 1.0}
    )
    assert decision.approved


def test_falls_back_to_metadata_scores():
    meta = AdapterMetadata(
        adapter_id="aero_v1", domain="aerospace", accuracy_score=0.9, stability_score=0.95
    )
    decision = AdapterPromotion().evaluate(meta, {"samples": 1.0})
    assert decision.approved


def test_default_metadata_scores_reject():
    meta = AdapterMetadata(adapter_id="aero_v1", domain="aerospace")
    decision = AdapterPromotion().evaluate(meta, {})
    assert not decision.approved


def test_min_samples_gate():
    meta = AdapterMetadata(adapter_id="aero_v1", domain="aerospace")
    promp = AdapterPromotion()
    missing = promp.evaluate(meta, {"accuracy": 0.9, "stability": 0.95})
    assert not missing.approved
    decision = promp.evaluate(meta, {"accuracy": 0.9, "stability": 0.95, "samples": 2.0})
    assert decision.approved


def test_custom_criteria():
    meta = AdapterMetadata(adapter_id="aero_v1", domain="aerospace")
    criteria = PromotionCriteria(min_accuracy=0.95, min_stability=0.98, min_samples=5)
    decision = AdapterPromotion().evaluate(
        meta,
        {"accuracy": 0.9, "stability": 0.95, "samples": 10.0},
        criteria=criteria,
    )
    assert not decision.approved
    assert any("0.95" in reason for reason in decision.reasons)


def test_promote_updates_status_when_approved():
    meta = AdapterMetadata(adapter_id="aero_v1", domain="aerospace")
    promp = AdapterPromotion()
    status = promp.promote(meta, {"accuracy": 0.9, "stability": 0.95, "samples": 1.0})
    assert status == PROMOTED_STATUS
    assert meta.status == PROMOTED_STATUS


def test_promote_keeps_status_when_rejected():
    meta = AdapterMetadata(adapter_id="aero_v1", domain="aerospace")
    promp = AdapterPromotion()
    status = promp.promote(meta, {"accuracy": 0.5, "stability": 0.5})
    assert status == "candidate"
    assert meta.status == "candidate"
