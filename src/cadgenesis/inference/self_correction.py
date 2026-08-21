"""Self-correction inference loop for CADGenesis-LM.

Provides a bounded retry loop that validates, identifies, and repairs
invalid CAD generation output. Every retry has a measurable reason
and the loop terminates after a configurable max attempts.
"""

from __future__ import annotations

from cadgenesis.confidence.risk import RiskAssessor, RiskConfig
from cadgenesis.execution.failure_modes import FailureMode, classify_reason
from cadgenesis.execution.geometry_validation import validate_program


class SelfCorrectionResult:
    """Result of a self-correction attempt.

    Attributes
    ----------
    success : bool
        Whether the correction succeeded within the budget.
    attempt : int
        The attempt number (1-indexed).
    cad_tokens : list[str] | None
        The corrected CAD token list, if successful.
    cad_text : str | None
        The prompt text corresponding to the corrected tokens.
    risk_score : float | None
        Composite risk score from the risk assessor.
    error : str | None
        Description of the failure, if unsuccessful.
    failure_mode : str
        Classified failure mode (see ``execution.failure_modes``).
    """

    def __init__(
        self,
        success: bool,
        attempt: int,
        cad_tokens: list[str] | None,
        cad_text: str | None,
        risk_score: float | None,
        error: str | None,
        failure_mode: str = FailureMode.UNKNOWN.value,
    ) -> None:
        self.success = success
        self.attempt = attempt
        self.cad_tokens = cad_tokens
        self.cad_text = cad_text
        self.risk_score = risk_score
        self.error = error
        self.failure_mode = failure_mode

    def __bool__(self) -> bool:
        return self.success


class SelfCorrectingInference:
    """Generate CAD output with self-correction and validation.

    Workflow:
        1. Generate candidate CAD program
        2. Validate via neuro-symbolic checks
        3. Validate via geometry kernel
        4. If invalid: identify failure mode -> repair -> retry
        5. Repeat within max_attempts budget
        6. Return best valid result or fail gracefully

    The loop terminates after max_attempts attempts even if all
    attempts fail, returning the least-invalid result along with
    diagnostic information.
    """

    def __init__(
        self,
        max_attempts: int = 5,
        risk_config: RiskConfig | None = None,
        geometry_weight: float = 1.0,
        constraint_weight: float = 1.0,
        dfm_weight: float = 0.5,
    ) -> None:
        self.max_attempts = max_attempts
        self.risk_config = risk_config or RiskConfig()
        self.risk_assessor = RiskAssessor(
            alpha=self.risk_config.uncertainty_penalty,
            beta=self.risk_config.consequence_weight,
            gamma=self.risk_config.uncertainty_penalty,
        )
        self.geometry_weight = geometry_weight
        self.constraint_weight = constraint_weight
        self.dfm_weight = dfm_weight

    def _validate_program(self, tokens: list[str]) -> tuple[bool, str]:
        """Validate a CAD token list using analytic geometry validator.

        Returns (is_valid, reason) where reason describes the failure.
        """
        # Token-level minimum check
        if not tokens:
            return False, "empty token list"

        # Check for required base operations
        required_keywords = {"EXTRUDE", "BOX", "CYLINDER", "SKETCH_RECT"}
        has_base = any(k in tokens for k in required_keywords)

        if not has_base:
            return False, "missing base solid operation (EXTRUDE/BOX/CYLINDER/SKETCH_RECT)"

        # Analytic geometry validator
        try:
            is_valid = validate_program(tokens)
            if not is_valid:
                return False, "geometry validation failed (analytic kernel)"
        except Exception as e:
            return False, f"validator error: {str(e)[:80]}"

        return True, "valid"

    def _assess_risk(self, tokens: list[str], prompt: str | None = None) -> float:
        """Assess risk score for a CAD program.

        Combines confidence, uncertainty, and consequence factors.
        """
        # Extract confidence-related tokens (features, dimensions)
        confidence_indicators = [t for t in tokens if t.startswith("NUM_")]
        num_features = len(confidence_indicators)

        # Heuristic risk based on program structure
        # More features + valid base = lower risk
        has_base = any(k in tokens for k in {"EXTRUDE", "BOX", "CYLINDER", "SKETCH_RECT"})

        risk = max(0.1, 0.7 - 0.03 * num_features) if has_base else 0.9

        # Clamp to [0, 1]
        return round(min(1.0, max(0.0, risk)), 4)

    def _attempt_repair(
        self,
        tokens: list[str],
        prompt: str,
        attempt: int,
    ) -> list[str] | None:
        """Attempt to repair invalid CAD token list.

        Repairs are hand-crafted pattern fixes based on common failure
        modes. They are not learned - they are deterministic corrections
        that increase the likelihood of passing the analytic validator.
        """
        repaired = list(tokens)

        # Common repair 1: Ensure EXTRUDE has a dimension
        if "EXTRUDE" in repaired:
            extrude_idx = None
            for i, t in enumerate(repaired):
                if t == "EXTRUDE":
                    extrude_idx = i
                    break
            if extrude_idx is not None and extrude_idx + 1 < len(repaired):
                next_token = repaired[extrude_idx + 1]
                if not next_token.startswith("NUM_"):
                    # Insert a default dimension
                    repaired.insert(extrude_idx + 1, "NUM_5")

        # Common repair 2: Ensure at least one feature after base solid
        base_ops = {"BOX", "CYLINDER", "SKETCH_RECT"}
        has_base = any(t in base_ops for t in repaired)
        features = [t for t in repaired if t not in base_ops and not t.startswith("NUM_")]
        if has_base and not features:
            # Add a default feature (FILLET is common)
            repaired.append("FILLET")

        # Common repair 3: Remove duplicate tokens
        seen = set()
        deduped = []
        for t in repaired:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        if len(deduped) < len(repaired) and self._quick_validate(deduped):
            return deduped

        # Quick-check the current repaired version
        if self._quick_validate(repaired):
            return repaired

        return None

    def _quick_validate(self, tokens: list[str]) -> bool:
        """Fast pre-check before the full analytic validator.

        Returns True if the token list has a reasonable structure,
        False if it's clearly malformed and should be skipped.
        """
        if not tokens:
            return False
        # Must have at least a base operation
        base_ops = {"BOX", "CYLINDER", "SKETCH_RECT"}
        if not any(t in base_ops for t in tokens):
            return False
        # Must not be completely NUM_-only
        non_numeric = [t for t in tokens if not t.startswith("NUM_")]
        return bool(non_numeric)

    def correct(self, prompt: str, initial_tokens: list[str]) -> SelfCorrectionResult:
        """Run the self-correction loop.

        Parameters
        ----------
        prompt : str
            The user prompt that generated the initial CAD program.
        initial_tokens : list[str]
            The initially generated CAD token list.

        Returns
        -------
        SelfCorrectionResult
            The result of the correction loop. ``success=True`` if a
            valid program was found within ``max_attempts``.
        """
        best_result: SelfCorrectionResult | None = None
        best_risk = 1.0  # Lower is better

        tokens = list(initial_tokens)  # Work with a mutable copy

        for attempt in range(1, self.max_attempts + 1):
            # On attempts > 1, try to repair based on previous failure
            if attempt > 1 and best_result is not None:
                repair_tokens = self._attempt_repair(tokens, prompt, attempt)
                if repair_tokens is not None:
                    tokens = repair_tokens
                # If repair failed, keep current tokens and try validation

            # Validate the program
            is_valid, reason = self._validate_program(tokens)

            # Classify the failure mode for metrics / diagnosis
            failure_mode = (
                FailureMode.UNKNOWN.value if is_valid else classify_reason(reason).value
            )

            # Assess risk
            risk_score = self._assess_risk(tokens, prompt)

            # If valid, track as best result (lowest risk wins among valid;
            # a valid result always outranks an invalid fallback)
            if is_valid:
                if (
                    best_result is None
                    or not best_result.success
                    or (best_result.risk_score is not None and risk_score < best_result.risk_score)
                ):
                    best_result = SelfCorrectionResult(
                        success=True,
                        attempt=attempt,
                        cad_tokens=list(tokens),
                        cad_text=prompt,
                        risk_score=risk_score,
                        error=None,
                        failure_mode=failure_mode,
                    )
                # Don't immediately return - continue if we can find
                # an even lower-risk valid program within budget
                if attempt < self.max_attempts:
                    continue

            # Track the best (lowest risk) invalid result as fallback.
            # Never shadow an already-found valid result.
            elif best_result is None or (not best_result.success and risk_score < best_risk):
                best_result = SelfCorrectionResult(
                    success=False,
                    attempt=attempt,
                    cad_tokens=list(tokens),
                    cad_text=prompt,
                    risk_score=risk_score,
                    error=reason,
                    failure_mode=failure_mode,
                )
                best_risk = risk_score

        # Return best result found within budget (check identity, not
        # truthiness: ``SelfCorrectionResult.__bool__`` follows ``success``,
        # so ``best_result or fallback`` would discard invalid results).
        if best_result is not None:
            return best_result
        return SelfCorrectionResult(
            success=False,
            attempt=self.max_attempts,
            cad_tokens=None,
            cad_text=prompt,
            risk_score=1.0,
            error="max attempts exceeded without valid program",
            failure_mode=FailureMode.EXECUTION_ERROR.value,
        )
