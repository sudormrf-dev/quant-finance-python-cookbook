"""Tests for returns.py."""

from __future__ import annotations

import math

from patterns.returns import (
    AnnualizationPeriod,
    ReturnSeries,
    annualize_return,
    compute_returns,
)


class TestComputeReturns:
    def test_simple_returns(self):
        prices = [100.0, 110.0, 99.0]
        r = compute_returns(prices)
        assert len(r) == 2
        assert abs(r[0] - 0.1) < 1e-9

    def test_log_returns(self):
        prices = [100.0, math.e * 100]
        r = compute_returns(prices, log_returns=True)
        assert abs(r[0] - 1.0) < 1e-6

    def test_empty_prices(self):
        assert compute_returns([]) == []

    def test_single_price(self):
        assert compute_returns([100.0]) == []

    def test_negative_return(self):
        prices = [100.0, 90.0]
        r = compute_returns(prices)
        assert abs(r[0] - (-0.1)) < 1e-9


class TestAnnualizeReturn:
    def test_daily_annualized(self):
        daily = 0.001
        ann = annualize_return(daily, AnnualizationPeriod.DAILY)
        assert ann > 0.25  # ~29% per year

    def test_zero_return(self):
        assert annualize_return(0.0, 252) == 0.0

    def test_monthly_annualized(self):
        monthly = 0.01
        ann = annualize_return(monthly, AnnualizationPeriod.MONTHLY)
        assert abs(ann - ((1.01**12) - 1)) < 1e-9


class TestReturnSeries:
    def setup_method(self):
        self.returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        self.rs = ReturnSeries("test", self.returns)

    def test_len(self):
        assert len(self.rs) == 5

    def test_mean_positive(self):
        assert self.rs.mean() > 0

    def test_std_positive(self):
        assert self.rs.std() > 0

    def test_cumulative(self):
        cum = self.rs.cumulative()
        expected = 1.01 * 0.98 * 1.03 * 0.99 * 1.02 - 1
        assert abs(cum - expected) < 1e-9

    def test_stats_count(self):
        stats = self.rs.stats()
        assert stats.count == 5

    def test_stats_min_max(self):
        stats = self.rs.stats()
        assert stats.min_return == -0.02
        assert stats.max_return == 0.03

    def test_rolling_mean_length(self):
        rm = self.rs.rolling_mean(3)
        assert len(rm) == 3

    def test_subset(self):
        sub = self.rs.subset(1, 4)
        assert len(sub) == 3

    def test_values_copy(self):
        vals = self.rs.values
        vals.append(99.0)
        assert len(self.rs) == 5

    def test_empty_series_stats(self):
        empty = ReturnSeries("empty", [])
        stats = empty.stats()
        assert stats.count == 0

    def test_annualized_return(self):
        stats = self.rs.stats()
        ann = stats.annualized_return(252)
        assert isinstance(ann, float)

    def test_annualized_vol(self):
        stats = self.rs.stats()
        vol = stats.annualized_vol(252)
        assert vol > 0

    def test_to_dict(self):
        d = self.rs.stats().to_dict()
        assert "mean" in d
        assert "std" in d
