"""Tests for risk_metrics.py."""

from __future__ import annotations

from patterns.risk_metrics import (
    RiskReport,
    VaRMethod,
    compute_cvar,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_var,
)

RETURNS = [0.01, -0.02, 0.03, -0.04, 0.02, -0.01, 0.015, -0.025, 0.005, 0.02]


class TestComputeVaR:
    def test_historical_var_positive(self):
        var = compute_var(RETURNS, 0.95, VaRMethod.HISTORICAL)
        assert var >= 0

    def test_parametric_var_positive(self):
        var = compute_var(RETURNS, 0.95, VaRMethod.PARAMETRIC)
        assert var >= 0

    def test_cornish_fisher_var(self):
        var = compute_var(RETURNS, 0.95, VaRMethod.CORNISH_FISHER)
        assert isinstance(var, float)

    def test_empty_returns(self):
        assert compute_var([], 0.95) == 0.0

    def test_higher_confidence_higher_var(self):
        var_90 = compute_var(RETURNS, 0.90, VaRMethod.HISTORICAL)
        var_99 = compute_var(RETURNS, 0.99, VaRMethod.HISTORICAL)
        assert var_99 >= var_90


class TestComputeCVaR:
    def test_cvar_positive(self):
        cvar = compute_cvar(RETURNS, 0.95)
        assert cvar >= 0

    def test_cvar_geq_var(self):
        var = compute_var(RETURNS, 0.95, VaRMethod.HISTORICAL)
        cvar = compute_cvar(RETURNS, 0.95)
        assert cvar >= var - 1e-9

    def test_empty(self):
        assert compute_cvar([], 0.95) == 0.0


class TestComputeSharpe:
    def test_positive_returns_positive_sharpe(self):
        positive = [0.01 + 0.001 * i for i in range(50)]
        sharpe = compute_sharpe(positive, 0.0, 252)
        assert sharpe > 0

    def test_zero_std_returns_zero(self):
        # All returns same → std = 0
        flat = [0.0] * 10
        assert compute_sharpe(flat, 0.0, 252) == 0.0

    def test_single_return(self):
        assert compute_sharpe([0.01], 0.0, 252) == 0.0

    def test_with_risk_free_rate(self):
        r = [0.001 + 0.0001 * i for i in range(100)]
        s_no_rf = compute_sharpe(r, 0.0, 252)
        s_with_rf = compute_sharpe(r, 0.05, 252)
        assert s_no_rf > s_with_rf


class TestComputeSortino:
    def test_positive_returns(self):
        # mix of positive with small negatives so downside_std > 0 but mean > 0
        mixed = [0.02, -0.005, 0.03, -0.003, 0.025] * 10
        sortino = compute_sortino(mixed, 0.0, 252)
        assert sortino > 0

    def test_no_downside(self):
        pos = [0.01] * 20
        s = compute_sortino(pos, 0.0, 252)
        assert s == 0.0  # downside_std = 0 → returns 0

    def test_single_return(self):
        assert compute_sortino([0.01]) == 0.0


class TestComputeMaxDrawdown:
    def test_all_positive_returns(self):
        assert compute_max_drawdown([0.01, 0.02, 0.03]) == 0.0

    def test_simple_drawdown(self):
        returns = [0.0, -0.5, 0.0]
        dd = compute_max_drawdown(returns)
        assert abs(dd - 0.5) < 1e-9

    def test_empty(self):
        assert compute_max_drawdown([]) == 0.0

    def test_between_zero_and_one(self):
        dd = compute_max_drawdown(RETURNS)
        assert 0.0 <= dd <= 1.0


class TestRiskReport:
    def setup_method(self):
        self.report = RiskReport(
            name="strategy",
            var_95=0.03,
            cvar_95=0.04,
            sharpe=1.2,
            sortino=1.5,
            max_drawdown=0.15,
            volatility=0.12,
            annualized_return=0.18,
        )

    def test_calmar(self):
        assert abs(self.report.calmar() - 0.18 / 0.15) < 1e-9

    def test_calmar_zero_drawdown(self):
        r = RiskReport("x", 0.0, 0.0, 1.0, 1.0, 0.0, 0.1, 0.1)
        assert r.calmar() == 0.0

    def test_is_acceptable_true(self):
        assert self.report.is_acceptable(min_sharpe=1.0, max_drawdown=0.2) is True

    def test_is_acceptable_false_sharpe(self):
        assert self.report.is_acceptable(min_sharpe=2.0) is False

    def test_is_acceptable_false_drawdown(self):
        assert self.report.is_acceptable(max_drawdown=0.1) is False

    def test_to_dict(self):
        d = self.report.to_dict()
        assert d["name"] == "strategy"
        assert "calmar" in d
