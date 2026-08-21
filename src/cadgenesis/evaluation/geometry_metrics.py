"""cadgenesis.evaluation.geometry_metrics
======================================
Geometric accuracy metrics.

Dimension-relative error, axis-aligned bounding-box IoU, TOON-based validity
via the CAD execution engine, and simple rule-based symmetry / perpendicularity
checks over parsed TOON objects.
"""

from __future__ import annotations

from math import prod
from typing import Any

from cadgenesis.execution.execution_engine import CADExecutionEngine
from sdk import toon_extended

_EPSILON = 1e-6


class GeometryMetrics:
    """Geometric accuracy metrics for CAD generations."""

    _DIM_KEYS: tuple[str, ...] = ("w", "h", "d")

    @staticmethod
    def dimension_relative_error(
        predicted: dict[str, float],
        reference: dict[str, float],
    ) -> float:
        """Mean relative error ``|p - r| / |r|`` over shared keys.

        Zero reference: error is 1.0 when the prediction is non-zero
        (fully wrong) and 0.0 when both are zero; results stay in [0, 1].
        """
        errors: list[float] = []
        for key in sorted(set(predicted) & set(reference)):
            p = float(predicted[key])
            r = float(reference[key])
            if abs(r) <= _EPSILON:
                errors.append(0.0 if abs(p) <= _EPSILON else 1.0)
            else:
                errors.append(abs(p - r) / abs(r))
        return sum(errors) / len(errors) if errors else 0.0

    @staticmethod
    def bbox_iou(
        predicted: tuple[float, float, float],
        reference: tuple[float, float, float],
    ) -> float:
        """IoU of axis-aligned boxes given as (width, height, depth).

        Boxes share an origin, so intersection is the elementwise min and
        union the elementwise max of the dimensions.  Zero-volume boxes
        (any non-positive dimension) score 0.0.
        """
        inter_dims: list[float] = []
        union_dims: list[float] = []
        for p, r in zip(predicted, reference, strict=True):
            inter_dims.append(min(max(p, 0.0), max(r, 0.0)))
            union_dims.append(max(max(p, 0.0), max(r, 0.0)))
        if any(d <= 0.0 for d in inter_dims) or any(d <= 0.0 for d in union_dims):
            return 0.0
        return prod(inter_dims) / prod(union_dims)

    @staticmethod
    def validity_via_execution(toon_strings: list[str]) -> tuple[float, dict[str, int]]:
        """Fraction of TOON strings that pass the execution engine.

        Each TOON is parsed and flattened to feature tokens, which are fed
        to ``CADExecutionEngine.execute_and_evaluate``.  Returns
        ``(valid_fraction, error_counts)`` where ``error_counts`` keys are
        ``"parse_failed"`` plus any engine error messages.
        """
        engine = CADExecutionEngine()
        valid = 0
        error_counts: dict[str, int] = {}
        for toon_str in toon_strings:
            parsed = toon_extended.from_toon(toon_str)
            if not parsed:
                error_counts["parse_failed"] = error_counts.get("parse_failed", 0) + 1
                continue
            tokens = [str(value) for obj in parsed for value in obj.values()]
            result = engine.execute_and_evaluate(tokens)
            if result.errors:
                for error in result.errors:
                    error_counts[error] = error_counts.get(error, 0) + 1
            else:
                valid += 1
        total = len(toon_strings)
        return (valid / total if total else 0.0), error_counts

    @staticmethod
    def symmetry_error(objects: list[dict[str, Any]], tolerance: float = 0.01) -> float:
        """Fraction of parsed objects with asymmetric width/height/depth.

        An object is symmetric when its w/h/d span is within ``tolerance``
        (relative to the largest absolute dimension).  Objects without
        numeric ``w``/``h``/``d`` values are skipped.
        """
        checked = 0
        asymmetric = 0
        for obj in objects:
            dims = GeometryMetrics._numeric_dims(obj)
            if dims is None:
                continue
            checked += 1
            span = max(dims) - min(dims)
            scale = max(abs(d) for d in dims)
            if span > tolerance * max(scale, _EPSILON):
                asymmetric += 1
        return asymmetric / checked if checked else 0.0

    @staticmethod
    def perpendicularity_error(objects: list[dict[str, Any]], tolerance: float = 1.0) -> float:
        """Fraction of parsed objects whose ``angle`` deviates from 90°.

        Angles are compared modulo 180° (circular distance).  Objects
        without a numeric ``angle`` value are skipped.
        """
        checked = 0
        violated = 0
        for obj in objects:
            raw_angle = obj.get("angle")
            if raw_angle is None:
                continue
            try:
                angle = float(raw_angle)
            except (TypeError, ValueError):
                continue
            checked += 1
            residual = abs((angle % 180.0) - 90.0)
            residual = min(residual, 180.0 - residual)
            if residual > tolerance:
                violated += 1
        return violated / checked if checked else 0.0

    @staticmethod
    def _numeric_dims(obj: dict[str, Any]) -> tuple[float, float, float] | None:
        """Extract (w, h, d) from an object dict; None when not numeric."""
        dims: list[float] = []
        for key in GeometryMetrics._DIM_KEYS:
            value = obj.get(key)
            if value is None:
                return None
            try:
                dims.append(float(value))
            except (TypeError, ValueError):
                return None
        return dims[0], dims[1], dims[2]


__all__ = ["GeometryMetrics"]
