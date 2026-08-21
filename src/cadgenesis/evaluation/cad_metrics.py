"""cadgenesis.evaluation.cad_metrics
=================================
CAD-generation metrics (validity, fidelity).

Metrics over the TOON serialization format and CAD token sequences: parse
validity via ``sdk.toon_extended.from_toon``, token-level fidelity,
normalized edit distance, primitive feature coverage, and plan/execution
feature agreement.  All methods are pure and operate on plain data.
"""

from __future__ import annotations

from sdk import toon_extended


class CADMetrics:
    """CAD-generation quality metrics (validity, fidelity, coverage)."""

    _FEATURE_KEYWORDS: tuple[str, ...] = ("BOX", "CYLINDER", "SPHERE", "EXTRUDE_PROFILE")
    _MAX_EDIT_LENGTH = 1000

    @staticmethod
    def validity_rate(toon_strings: list[str]) -> float:
        """Fraction of TOON strings that parse to at least one object."""
        if not toon_strings:
            return 0.0
        valid = sum(1 for s in toon_strings if toon_extended.from_toon(s))
        return valid / len(toon_strings)

    @staticmethod
    def token_accuracy(
        predicted: list[list[str]],
        reference: list[list[str]],
    ) -> float:
        """Mean exact-token match rate over aligned token sequences."""
        accuracies: list[float] = []
        for pred, ref in zip(predicted, reference, strict=False):
            if not ref:
                accuracies.append(0.0)
                continue
            matches = sum(1 for p, r in zip(pred, ref, strict=False) if p == r)
            accuracies.append(matches / len(ref))
        return sum(accuracies) / len(accuracies) if accuracies else 0.0

    @staticmethod
    def edit_distance_similarity(predicted: list[str], reference: list[str]) -> float:
        """1 - normalized Levenshtein over space-joined token lists.

        Sequences are joined with a single space and compared as strings.
        Inputs longer than ``_MAX_EDIT_LENGTH`` chars are truncated to keep
        the O(n*m) DP bounded.
        """
        a = " ".join(predicted)
        b = " ".join(reference)
        max_len = max(len(a), len(b))
        if max_len == 0:
            return 1.0
        distance = CADMetrics._levenshtein(a, b)
        return 1.0 - distance / max_len

    @staticmethod
    def primitive_coverage(toon_strings: list[str]) -> dict[str, float]:
        """Fraction of samples whose parsed objects contain each feature.

        ``EXTRUDE_PROFILE`` matches samples whose objects include the
        ``SKETCH_RECT`` / ``EXTRUDE`` profile-extrusion pattern.
        """
        counts = {keyword: 0 for keyword in CADMetrics._FEATURE_KEYWORDS}
        total = len(toon_strings)
        if total == 0:
            return {keyword: 0.0 for keyword in counts}
        for toon_str in toon_strings:
            values: set[str] = set()
            for obj in toon_extended.from_toon(toon_str):
                values.update(str(value) for value in obj.values())
            if "BOX" in values:
                counts["BOX"] += 1
            if "CYLINDER" in values:
                counts["CYLINDER"] += 1
            if "SPHERE" in values:
                counts["SPHERE"] += 1
            if values & {"SKETCH_RECT", "EXTRUDE"}:
                counts["EXTRUDE_PROFILE"] += 1
        return {keyword: count / total for keyword, count in counts.items()}

    @staticmethod
    def plan_accuracy(
        executed_features: list[list[str]],
        planned_features: list[list[str]],
    ) -> float:
        """Mean fraction of planned features present in the executed set."""
        accuracies: list[float] = []
        for executed, planned in zip(executed_features, planned_features, strict=False):
            planned_set = set(planned)
            if not planned_set:
                accuracies.append(0.0)
                continue
            accuracies.append(len(planned_set & set(executed)) / len(planned_set))
        return sum(accuracies) / len(accuracies) if accuracies else 0.0

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        """Classic two-row DP edit distance (O(n*m) time, O(m) memory)."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        a = a[: CADMetrics._MAX_EDIT_LENGTH]
        b = b[: CADMetrics._MAX_EDIT_LENGTH]
        previous = list(range(len(b) + 1))
        for i, char_a in enumerate(a, start=1):
            current = [i]
            for j, char_b in enumerate(b, start=1):
                cost = 0 if char_a == char_b else 1
                current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
            previous = current
        return previous[-1]


__all__ = ["CADMetrics"]
