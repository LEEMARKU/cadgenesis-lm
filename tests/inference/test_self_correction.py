"""Tests for the M8 self-correction family: the bounded retry loop, repair
heuristics, risk assessment and the best-result tracking fix."""

from __future__ import annotations

from cadgenesis.inference.self_correction import SelfCorrectingInference


class _Scripted(SelfCorrectingInference):
    """SelfCorrectingInference with scripted validation/risk/repair."""

    def __init__(
        self,
        outcomes: list[tuple[bool, str, float]],
        repairs: dict[str, list[str]] | None = None,
        max_attempts: int = 5,
    ) -> None:
        super().__init__(max_attempts=max_attempts)
        self.outcomes = list(outcomes)
        self.calls = 0
        self.repairs = repairs or {}

    def _validate_program(self, tokens: list[str]) -> tuple[bool, str]:
        self.calls += 1
        if self.calls > len(self.outcomes):
            return False, "out of script"
        valid, reason, _ = self.outcomes[self.calls - 1]
        return valid, reason

    def _assess_risk(self, tokens: list[str], prompt: str | None = None) -> float:
        if self.calls <= len(self.outcomes):
            return self.outcomes[self.calls - 1][2]
        return 0.9

    def _attempt_repair(self, tokens: list[str], prompt: str, attempt: int) -> list[str] | None:
        return self.repairs.get("".join(tokens), None)


def test_valid_first_attempt() -> None:
    loop = _Scripted([(True, "valid", 0.4)])
    result = loop.correct("make a box", ["BOX", "NUM_80"])
    assert result.success
    assert result.attempt == 1
    assert result.cad_tokens == ["BOX", "NUM_80"]
    assert result.error is None
    assert result.risk_score == 0.4


def test_repair_then_valid() -> None:
    loop = _Scripted(
        [(False, "missing base solid operation", 0.9), (True, "valid", 0.5)],
        repairs={"EXTRUDE": ["EXTRUDE", "NUM_5"]},
    )
    result = loop.correct("extrude something", ["EXTRUDE"])
    assert result.success
    assert result.attempt == 2
    assert result.cad_tokens == ["EXTRUDE", "NUM_5"]


def test_all_invalid_returns_least_risk_fallback() -> None:
    loop = _Scripted(
        [
            (False, "geometry validation failed (analytic kernel)", 0.9),
            (False, "geometry validation failed (analytic kernel)", 0.6),
            (False, "geometry validation failed (analytic kernel)", 0.3),
        ]
    )
    result = loop.correct("bad", ["BOX", "NUM_80", "NUM_40", "NUM_20"])
    assert not result.success
    assert result.risk_score == 0.3
    assert result.error is not None


def test_valid_never_shadowed_by_lower_risk_invalid() -> None:
    """Regression: ``best_risk`` was only updated on the invalid branch, so
    a later lower-risk invalid attempt overwrote a found valid program."""
    loop = _Scripted(
        [
            (False, "geometry validation failed (analytic kernel)", 0.9),
            (True, "valid", 0.5),
            (False, "geometry validation failed (analytic kernel)", 0.3),
        ],
        repairs={"BOX": ["BOX", "NUM_80", "NUM_40", "NUM_20"]},
    )
    result = loop.correct("box", ["BOX"])
    assert result.success
    assert result.attempt == 2
    assert result.cad_tokens == ["BOX", "NUM_80", "NUM_40", "NUM_20"]
    assert result.risk_score == 0.5


def test_lower_risk_valid_beats_higher_risk_valid() -> None:
    loop = _Scripted(
        [
            (True, "valid", 0.7),
            (True, "valid", 0.4),
        ],
        repairs={"A": ["B"]},
    )
    result = loop.correct("p", ["A"])
    assert result.success
    assert result.attempt == 2
    assert result.cad_tokens == ["B"]
    assert result.risk_score == 0.4


def test_max_attempts_budget_respected() -> None:
    loop = _Scripted(
        [(False, "geometry validation failed (analytic kernel)", 0.9)] * 3, max_attempts=3
    )
    result = loop.correct("p", ["NUM_1"])
    assert not result.success
    assert loop.calls == 3


def test_result_bool() -> None:
    loop = _Scripted([(True, "valid", 0.4)])
    assert bool(loop.correct("p", ["BOX"]))
    loop = _Scripted([(False, "geometry validation failed (analytic kernel)", 0.9)])
    assert not bool(loop.correct("p", ["BOX"]))


def test_real_validation_path_valid_program() -> None:
    loop = SelfCorrectingInference(max_attempts=3)
    result = loop.correct("a box", ["BOX", "NUM_80", "NUM_40", "NUM_20"])
    assert result.success
    assert result.error is None


def test_real_validation_path_rejects_nonsense() -> None:
    loop = SelfCorrectingInference(max_attempts=2)
    result = loop.correct("nonsense", ["<eos>"])
    assert not result.success
    assert result.error is not None


def test_risk_heuristics() -> None:
    loop = SelfCorrectingInference()
    assert loop._assess_risk(["BOX", "NUM_80"], "p") == 0.67
    assert loop._assess_risk(["NUM_1"], "p") == 0.9
    many = ["BOX"] + [f"NUM_{i}" for i in range(10)]
    assert loop._assess_risk(many, "p") >= 0.1
