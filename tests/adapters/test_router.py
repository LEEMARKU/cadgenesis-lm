"""Tests for cadgenesis.adapters.router."""

import pytest

from cadgenesis.adapters.router import AdapterRouter


def test_exact_domain_match():
    router = AdapterRouter()
    router.register("aero_v1", ["aerospace"], "wing and spar design")
    decision = router.route("design a wing", domain="aerospace")
    assert decision.adapter_id == "aero_v1"
    assert decision.strategy == "exact"
    assert decision.score == 1.0


def test_exact_match_case_insensitive():
    router = AdapterRouter()
    router.register("aero_v1", ["aerospace"], "")
    decision = router.route("anything", domain="Aerospace")
    assert decision.adapter_id == "aero_v1"
    assert decision.strategy == "exact"


def test_exact_match_unknown_domain_falls_back():
    router = AdapterRouter(default_adapter_id="base")
    router.register("aero_v1", ["aerospace"], "wing design")
    decision = router.route("design a wing", domain="unknown_domain")
    assert decision.strategy == "similarity"
    assert decision.adapter_id == "aero_v1"


def test_similarity_fallback_uses_domains_and_description():
    router = AdapterRouter()
    router.register("aero_v1", ["aerospace"], "wing spar aircraft design")
    router.register("mold_v1", ["injection_molding"], "plastic mold cavity")
    decision = router.route("design a wing spar for an aircraft")
    assert decision.adapter_id == "aero_v1"
    assert decision.strategy == "similarity"
    assert decision.score > 0.0


def test_default_fallback():
    router = AdapterRouter(default_adapter_id="base_weights")
    decision = router.route("zzzz qqqq wwww")
    assert decision.adapter_id == "base_weights"
    assert decision.strategy == "default"
    assert decision.score == 0.0


def test_no_match_returns_none():
    router = AdapterRouter()
    router.register("aero_v1", ["aerospace"], "wing design")
    decision = router.route("zzzz qqqq wwww")
    assert decision.adapter_id is None
    assert decision.strategy == "default"
    assert decision.score == 0.0


def test_deterministic_tie_break_exact():
    router = AdapterRouter()
    router.register("zeta", ["aerospace"], "")
    router.register("alpha", ["aerospace"], "")
    decision = router.route("whatever", domain="aerospace")
    assert decision.adapter_id == "alpha"


def test_deterministic_tie_break_similarity():
    router = AdapterRouter()
    router.register("zeta", ["wing"], "spar aircraft")
    router.register("alpha", ["wing"], "spar aircraft")
    decision = router.route("wing spar aircraft design")
    assert decision.adapter_id == "alpha"


def test_register_duplicate_raises():
    router = AdapterRouter()
    router.register("aero_v1", ["aerospace"])
    with pytest.raises(ValueError):
        router.register("aero_v1", ["automotive"])
