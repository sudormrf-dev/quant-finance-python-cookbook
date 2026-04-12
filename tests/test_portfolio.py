"""Tests for portfolio.py."""

from __future__ import annotations

from patterns.portfolio import (
    Asset,
    Portfolio,
    PortfolioWeights,
    RebalanceRule,
)


class TestAsset:
    def test_mean_return_empty(self):
        a = Asset("AAPL")
        assert a.mean_return() == 0.0

    def test_mean_return(self):
        a = Asset("AAPL", returns=[0.01, 0.02, 0.03])
        assert abs(a.mean_return() - 0.02) < 1e-9

    def test_volatility_empty(self):
        a = Asset("AAPL")
        assert a.volatility() == 0.0

    def test_volatility_positive(self):
        a = Asset("AAPL", returns=[0.01, -0.02, 0.03])
        assert a.volatility() > 0

    def test_set_returns_returns_self(self):
        a = Asset("AAPL")
        assert a.set_returns([0.01]) is a


class TestPortfolioWeights:
    def test_normalizes_on_init(self):
        w = PortfolioWeights({"A": 1.0, "B": 1.0})
        assert abs(w.get("A") - 0.5) < 1e-9

    def test_is_valid(self):
        w = PortfolioWeights({"A": 0.6, "B": 0.4})
        assert w.is_valid() is True

    def test_get_missing_zero(self):
        w = PortfolioWeights({"A": 1.0})
        assert w.get("Z") == 0.0

    def test_equal_weight(self):
        w = PortfolioWeights.equal_weight(["A", "B", "C"])
        assert abs(w.get("A") - 1 / 3) < 1e-9

    def test_equal_weight_empty(self):
        w = PortfolioWeights.equal_weight([])
        assert w.weights == {}

    def test_top_n(self):
        w = PortfolioWeights({"A": 3.0, "B": 1.0, "C": 2.0})
        top2 = w.top_n(2)
        assert "B" not in top2.weights

    def test_normalize_zero_total(self):
        w = PortfolioWeights.__new__(PortfolioWeights)
        w.weights = {}
        result = w.normalize()
        assert result.weights == {}


class TestPortfolio:
    def setup_method(self):
        self.p = Portfolio("test_portfolio")
        self.a1 = Asset("AAPL", sector="Tech", returns=[0.01, 0.02, -0.01])
        self.a2 = Asset("MSFT", sector="Tech", returns=[0.02, -0.01, 0.03])
        self.p.add_asset(self.a1, 0.6)
        self.p.add_asset(self.a2, 0.4)

    def test_asset_count(self):
        assert self.p.asset_count() == 2

    def test_tickers_sorted(self):
        tickers = self.p.tickers()
        assert tickers == sorted(tickers)

    def test_weight_of(self):
        w = self.p.weight_of("AAPL")
        assert 0 < w <= 1

    def test_portfolio_return(self):
        r = self.p.portfolio_return()
        assert isinstance(r, float)

    def test_portfolio_volatility(self):
        v = self.p.portfolio_volatility()
        assert v > 0

    def test_concentration_hhi(self):
        hhi = self.p.concentration_hhi()
        assert 0 < hhi <= 1.0

    def test_effective_n(self):
        n = self.p.effective_n()
        assert n > 0

    def test_sector_weights(self):
        sw = self.p.sector_weights()
        assert "Tech" in sw
        assert abs(sw["Tech"] - 1.0) < 1e-9

    def test_add_asset_returns_self(self):
        p = Portfolio("p")
        a = Asset("X")
        assert p.add_asset(a) is p

    def test_set_weights(self):
        w = PortfolioWeights({"AAPL": 0.7, "MSFT": 0.3})
        result = self.p.set_weights(w)
        assert result is self.p
        assert abs(self.p.weight_of("AAPL") - 0.7) < 1e-9

    def test_rebalance_rule(self):
        p = Portfolio("p", RebalanceRule.MONTHLY)
        assert p.rebalance_rule == RebalanceRule.MONTHLY
