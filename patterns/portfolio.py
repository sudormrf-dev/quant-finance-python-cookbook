"""Portfolio construction and management patterns."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RebalanceRule(str, Enum):
    """Portfolio rebalancing rules."""

    NONE = "none"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    THRESHOLD = "threshold"
    ANNUAL = "annual"


@dataclass
class Asset:
    """A single portfolio asset."""

    ticker: str
    name: str = ""
    sector: str = ""
    currency: str = "USD"
    returns: list[float] = field(default_factory=list)

    def mean_return(self) -> float:
        if not self.returns:
            return 0.0
        return sum(self.returns) / len(self.returns)

    def volatility(self) -> float:
        if len(self.returns) < 2:
            return 0.0
        m = self.mean_return()
        var = sum((r - m) ** 2 for r in self.returns) / (len(self.returns) - 1)
        return math.sqrt(var)

    def set_returns(self, returns: list[float]) -> Asset:
        self.returns = list(returns)
        return self


@dataclass
class PortfolioWeights:
    """Asset weights in a portfolio."""

    weights: dict[str, float]

    def __post_init__(self) -> None:
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def get(self, ticker: str) -> float:
        return self.weights.get(ticker, 0.0)

    def normalize(self) -> PortfolioWeights:
        total = sum(self.weights.values())
        if total == 0:
            return self
        return PortfolioWeights({k: v / total for k, v in self.weights.items()})

    def is_valid(self, tol: float = 1e-6) -> bool:
        return abs(sum(self.weights.values()) - 1.0) < tol

    @staticmethod
    def equal_weight(tickers: list[str]) -> PortfolioWeights:
        n = len(tickers)
        if n == 0:
            return PortfolioWeights({})
        w = 1.0 / n
        return PortfolioWeights({t: w for t in tickers})

    def top_n(self, n: int) -> PortfolioWeights:
        sorted_items = sorted(self.weights.items(), key=lambda x: x[1], reverse=True)
        return PortfolioWeights(dict(sorted_items[:n]))


@dataclass
class PortfolioStats:
    """Portfolio performance statistics."""

    total_return: float
    annualized_return: float
    volatility: float
    sharpe: float
    max_drawdown: float
    asset_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "asset_count": self.asset_count,
        }


class Portfolio:
    """A collection of assets with weights."""

    def __init__(
        self, name: str, rebalance: RebalanceRule = RebalanceRule.NONE
    ) -> None:
        self.name = name
        self.rebalance_rule = rebalance
        self._assets: dict[str, Asset] = {}
        self._weights: PortfolioWeights = PortfolioWeights({})

    def add_asset(self, asset: Asset, weight: float = 0.0) -> Portfolio:
        self._assets[asset.ticker] = asset
        new_weights = dict(self._weights.weights)
        new_weights[asset.ticker] = weight
        self._weights = PortfolioWeights(new_weights)
        return self

    def set_weights(self, weights: PortfolioWeights) -> Portfolio:
        self._weights = weights
        return self

    def asset_count(self) -> int:
        return len(self._assets)

    def tickers(self) -> list[str]:
        return sorted(self._assets.keys())

    def weight_of(self, ticker: str) -> float:
        return self._weights.get(ticker)

    def portfolio_return(self) -> float:
        """Compute weighted mean return across all assets."""
        total = 0.0
        for ticker, asset in self._assets.items():
            total += self._weights.get(ticker) * asset.mean_return()
        return total

    def portfolio_volatility(self) -> float:
        """Approximate portfolio vol as weighted avg of individual vols (no correlation)."""
        total = 0.0
        for ticker, asset in self._assets.items():
            total += self._weights.get(ticker) * asset.volatility()
        return total

    def concentration_hhi(self) -> float:
        """Herfindahl-Hirschman Index: 1 = fully concentrated, 1/n = equal weight."""
        return sum(w**2 for w in self._weights.weights.values())

    def effective_n(self) -> float:
        """Effective number of assets (inverse HHI)."""
        hhi = self.concentration_hhi()
        return 1.0 / hhi if hhi > 0 else 0.0

    def sector_weights(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for ticker, asset in self._assets.items():
            sector = asset.sector or "Unknown"
            result[sector] = result.get(sector, 0.0) + self._weights.get(ticker)
        return result
