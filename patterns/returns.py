"""Return computation and statistics patterns."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class AnnualizationPeriod(int, Enum):
    """Trading periods per year for annualization."""

    DAILY = 252
    WEEKLY = 52
    MONTHLY = 12
    QUARTERLY = 4
    ANNUAL = 1


def compute_returns(prices: list[float], log_returns: bool = False) -> list[float]:
    """Compute simple or log returns from a price series."""
    if len(prices) < 2:
        return []
    if log_returns:
        return [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
    return [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]


def annualize_return(period_return: float, periods_per_year: int) -> float:
    """Annualize a periodic return."""
    return (1 + period_return) ** periods_per_year - 1


@dataclass
class ReturnStats:
    """Descriptive statistics for a return series."""

    count: int
    mean: float
    std: float
    min_return: float
    max_return: float
    skewness: float
    kurtosis: float
    total_return: float

    def annualized_return(
        self, periods_per_year: int = AnnualizationPeriod.DAILY
    ) -> float:
        return annualize_return(self.mean, periods_per_year)

    def annualized_vol(
        self, periods_per_year: int = AnnualizationPeriod.DAILY
    ) -> float:
        return self.std * math.sqrt(periods_per_year)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_return,
            "max": self.max_return,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "total_return": self.total_return,
        }


class ReturnSeries:
    """A named return series with statistics."""

    def __init__(self, name: str, returns: list[float]) -> None:
        self.name = name
        self._returns = list(returns)

    @property
    def values(self) -> list[float]:
        return list(self._returns)

    def __len__(self) -> int:
        return len(self._returns)

    def mean(self) -> float:
        if not self._returns:
            return 0.0
        return sum(self._returns) / len(self._returns)

    def variance(self) -> float:
        if len(self._returns) < 2:
            return 0.0
        m = self.mean()
        return sum((r - m) ** 2 for r in self._returns) / (len(self._returns) - 1)

    def std(self) -> float:
        return math.sqrt(self.variance())

    def cumulative(self) -> float:
        result = 1.0
        for r in self._returns:
            result *= 1 + r
        return result - 1.0

    def stats(self) -> ReturnStats:
        n = len(self._returns)
        if n == 0:
            return ReturnStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        m = self.mean()
        s = self.std()
        sorted_r = sorted(self._returns)
        skew = 0.0
        kurt = 0.0
        if s > 0 and n >= 3:
            skew = sum((r - m) ** 3 for r in self._returns) / (n * s**3)
            kurt = sum((r - m) ** 4 for r in self._returns) / (n * s**4) - 3
        return ReturnStats(
            count=n,
            mean=m,
            std=s,
            min_return=sorted_r[0],
            max_return=sorted_r[-1],
            skewness=skew,
            kurtosis=kurt,
            total_return=self.cumulative(),
        )

    def rolling_mean(self, window: int) -> list[float]:
        result = []
        for i in range(window - 1, len(self._returns)):
            window_vals = self._returns[i - window + 1 : i + 1]
            result.append(sum(window_vals) / window)
        return result

    def subset(self, start: int, end: int) -> ReturnSeries:
        return ReturnSeries(self.name, self._returns[start:end])
