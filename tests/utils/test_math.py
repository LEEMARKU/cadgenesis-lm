"""tests/utils/test_math.py"""

from __future__ import annotations

import math

import pytest

from cadgenesis.utils.math import (
    ExponentialMovingAverage,
    RunningStats,
    aspect_ratio,
    bbox_volume,
    clamp,
    euclidean_distance,
    lerp,
    mean,
    median,
    norm_angle_deg,
    pct_change,
    percentile,
    round_to,
    safe_div,
    smoothstep,
)


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10


def test_lerp():
    assert lerp(0, 10, 0.5) == 5
    assert lerp(0, 10, 0) == 0
    assert lerp(0, 10, 1) == 10


def test_safe_div():
    assert safe_div(10, 2) == 5
    assert safe_div(10, 0) == 0
    assert safe_div(10, 0, default=-1) == -1
    assert safe_div(10, float("nan")) == 0


def test_smoothstep():
    assert smoothstep(0, 1, 0) == 0
    assert smoothstep(0, 1, 1) == 1
    assert smoothstep(0, 1, 0.5) == pytest.approx(0.5)


def test_round_to():
    assert round_to(0.126, 0.01) == pytest.approx(0.13)
    assert round_to(1.5, 1.0) == pytest.approx(2.0)


def test_mean_nan_safe():
    assert mean([1, 2, float("nan"), 3]) == pytest.approx(2.0)
    assert mean([]) == 0.0


def test_running_stats():
    stats = RunningStats()
    stats.update_many([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats.count == 5
    assert stats.mean == pytest.approx(3.0)
    assert stats.variance == pytest.approx(2.5)
    assert stats.stddev == pytest.approx(math.sqrt(2.5))


def test_running_stats_merge():
    a = RunningStats()
    a.update_many([1.0, 2.0, 3.0])
    b = RunningStats()
    b.update_many([4.0, 5.0])
    a.merge(b)
    assert a.count == 5
    assert a.mean == pytest.approx(3.0)


def test_ema_warmup():
    ema = ExponentialMovingAverage(alpha=0.5)
    assert ema.update(10.0) == pytest.approx(10.0)
    assert ema.update(20.0) == pytest.approx(15.0)


def test_percentile():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 100) == 4.0
    assert percentile(values, 50) == pytest.approx(2.5)
    assert percentile(values, 25) == pytest.approx(1.75)
    with pytest.raises(ValueError):
        percentile([], 50)


def test_median():
    assert median([1, 3, 2]) == pytest.approx(2)
    assert median([1, 2, 3, 4]) == pytest.approx(2.5)


def test_pct_change():
    assert pct_change(110, 100) == pytest.approx(10.0)
    assert pct_change(90, 100) == pytest.approx(-10.0)


def test_norm_angle_deg():
    assert norm_angle_deg(370) == pytest.approx(10.0)
    assert norm_angle_deg(-190) == pytest.approx(170.0)


def test_euclidean_distance():
    assert euclidean_distance([0, 0], [3, 4]) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        euclidean_distance([1], [1, 2])


def test_bbox_volume():
    assert bbox_volume([0, 0, 0], [10, 10, 10]) == pytest.approx(1000.0)
    assert bbox_volume([0, 0], [10, 0]) == pytest.approx(0.0)


def test_aspect_ratio():
    assert aspect_ratio(200, 100) == pytest.approx(2.0)
    assert aspect_ratio(0, 0) == 1.0
