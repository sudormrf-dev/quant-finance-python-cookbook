"""Quantitative finance patterns in Python."""

from __future__ import annotations

from patterns.portfolio import (
    Asset,
    Portfolio,
    PortfolioStats,
    PortfolioWeights,
    RebalanceRule,
)
from patterns.pricing import (
    BlackScholesInputs,
    OptionContract,
    OptionStyle,
    OptionType,
    black_scholes_greeks,
    black_scholes_price,
)
from patterns.returns import (
    AnnualizationPeriod,
    ReturnSeries,
    ReturnStats,
    annualize_return,
    compute_returns,
)
from patterns.risk_metrics import (
    RiskMetric,
    RiskReport,
    VaRMethod,
    compute_cvar,
    compute_max_drawdown,
    compute_sharpe,
    compute_sortino,
    compute_var,
)

__all__ = [
    "AnnualizationPeriod",
    "Asset",
    "BlackScholesInputs",
    "OptionContract",
    "OptionStyle",
    "OptionType",
    "Portfolio",
    "PortfolioStats",
    "PortfolioWeights",
    "RebalanceRule",
    "ReturnSeries",
    "ReturnStats",
    "RiskMetric",
    "RiskReport",
    "VaRMethod",
    "annualize_return",
    "black_scholes_greeks",
    "black_scholes_price",
    "compute_cvar",
    "compute_max_drawdown",
    "compute_returns",
    "compute_sharpe",
    "compute_sortino",
    "compute_var",
]
