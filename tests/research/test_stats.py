from __future__ import annotations

from cadgenesis.research.stats import (
    bootstrap_ci,
    cohens_d,
    describe,
    mean_ci_normal,
    welch_t_test,
)


class TestMeanCI:
    def test_normal_ci_contains_mean(self):
        samples = [1.0, 1.1, 0.9, 1.05, 0.95]
        ci = mean_ci_normal(samples)
        assert ci.estimate == 1.0
        assert ci.lower <= 1.0 <= ci.upper
        assert ci.lower < ci.upper

    def test_contains(self):
        ci = mean_ci_normal([1, 2, 3])
        assert ci.contains(2.0)
        assert not ci.contains(100.0)


class TestBootstrapCI:
    def test_bounds_cover_sample_range(self):
        samples = [3.0, 4.0, 5.0, 6.0, 7.0]
        ci = bootstrap_ci(samples, level=0.95, resamples=500, seed=42)
        assert ci.lower <= ci.estimate <= ci.upper
        assert ci.lower < ci.upper

    def test_seeded_determinism(self):
        samples = [1.0, 2.0, 3.0, 4.0]
        first = bootstrap_ci(samples, resamples=200, seed=7)
        second = bootstrap_ci(samples, resamples=200, seed=7)
        assert first.lower == second.lower and first.upper == second.upper


class TestWelchTTest:
    def test_significant_difference(self):
        group_a = [1.0, 1.1, 0.9, 1.0, 1.05]
        group_b = [2.0, 2.2, 1.8, 2.1, 1.9]
        result = welch_t_test(group_a, group_b)
        assert result.significant is True
        assert result.p_value < 0.05
        assert result.statistic < 0

    def test_insignificant_difference(self):
        group_a = [1.0, 1.01, 0.99, 1.0, 1.02]
        group_b = [1.0, 1.02, 0.98, 1.01, 1.0]
        result = welch_t_test(group_a, group_b)
        assert result.significant is False

    def test_identical_groups(self):
        group = [1.0, 2.0, 3.0]
        result = welch_t_test(group, group)
        assert result.statistic == 0.0
        assert result.p_value > 0.9


class TestCohenD:
    def test_positive_direction(self):
        a = [1.0, 1.1, 0.9]
        b = [2.0, 2.1, 1.9]
        assert cohens_d(a, b) < 0

    def test_zero_when_identical(self):
        a = [1.0, 2.0, 3.0]
        assert cohens_d(a, list(a)) == 0.0


class TestDescribe:
    def test_summary(self):
        summary = describe([1, 2, 3, 4, 5])
        assert summary["n"] == 5
        assert summary["mean"] == 3.0
        assert summary["median"] == 3.0
        assert summary["min"] == 1
        assert summary["max"] == 5
