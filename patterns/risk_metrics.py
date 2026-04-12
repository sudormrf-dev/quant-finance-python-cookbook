"""Risk metric computations: VaR, CVaR, Sharpe, Sortino, drawdown."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class RiskMetric(str, Enum):
    """Supported risk metrics."""

    VAR = "value_at_risk"
    CVAR = "conditional_var"
    SHARPE = "sharpe_ratio"
    SORTINO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR = "calmar_ratio"
    VOLATILITY = "volatility"


class VaRMethod(str, Enum):
    """VaR calculation methods."""

    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    CORNISH_FISHER = "cornish_fisher"


def compute_var(
    returns: list[float],
    confidence: float = 0.95,
    method: VaRMethod = VaRMethod.HISTORICAL,
) -> float:
    """Compute Value at Risk (loss as positive number)."""
    if not returns:
        return 0.0
    if method == VaRMethod.HISTORICAL:
        sorted_r = sorted(returns)
        idx = int((1 - confidence) * len(sorted_r))
        return -sorted_r[max(0, idx)]
    if method == VaRMethod.PARAMETRIC:
        m = sum(returns) / len(returns)
        variance = sum((r - m) ** 2 for r in returns) / max(1, len(returns) - 1)
        sigma = math.sqrt(variance)
        # z-score for confidence level (approx)
        z = _normal_quantile(confidence)
        return -(m - z * sigma)
    # Cornish-Fisher approximation
    m = sum(returns) / len(returns)
    variance = sum((r - m) ** 2 for r in returns) / max(1, len(returns) - 1)
    sigma = math.sqrt(variance)
    n = len(returns)
    skew = sum((r - m) ** 3 for r in returns) / (n * sigma**3) if sigma > 0 else 0.0
    z = _normal_quantile(confidence)
    z_cf = z + (z**2 - 1) * skew / 6
    return -(m - z_cf * sigma)


def compute_cvar(returns: list[float], confidence: float = 0.95) -> float:
    """Compute Conditional VaR (Expected Shortfall)."""
    if not returns:
        return 0.0
    sorted_r = sorted(returns)
    cutoff = int((1 - confidence) * len(sorted_r))
    tail = sorted_r[: max(1, cutoff)]
    return -sum(tail) / len(tail)


def compute_sharpe(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    excess = [r - risk_free_rate / periods_per_year for r in returns]
    m = sum(excess) / len(excess)
    variance = sum((r - m) ** 2 for r in excess) / (len(excess) - 1)
    sigma = math.sqrt(variance)
    if sigma == 0:
        return 0.0
    return (m / sigma) * math.sqrt(periods_per_year)


def compute_sortino(
    returns: list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sortino ratio (downside deviation)."""
    if len(returns) < 2:
        return 0.0
    target = risk_free_rate / periods_per_year
    excess_mean = sum(r - target for r in returns) / len(returns)
    downside = [min(0.0, r - target) for r in returns]
    downside_var = sum(d**2 for d in downside) / max(1, len(downside) - 1)
    downside_std = math.sqrt(downside_var)
    if downside_std == 0:
        return 0.0
    return (excess_mean / downside_std) * math.sqrt(periods_per_year)


def compute_max_drawdown(returns: list[float]) -> float:
    """Maximum drawdown from a return series (returns positive number)."""
    if not returns:
        return 0.0
    cumulative = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        cumulative *= 1 + r
        peak = max(peak, cumulative)
        dd = (peak - cumulative) / peak
        max_dd = max(max_dd, dd)
    return max_dd


def _normal_quantile(p: float) -> float:
    """Approximate inverse normal CDF (rational approximation)."""
    # Beasley-Springer-Moro approximation
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511349,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    ]
    y = p - 0.5
    if abs(y) < 0.42:
        r = y * y
        x = (
            y
            * (((a[3] * r + a[2]) * r + a[1]) * r + a[0])
            / ((((b[3] * r + b[2]) * r + b[1]) * r + b[0]) * r + 1)
        )
        return x
    r = p if y < 0 else 1 - p
    r = math.log(-math.log(r))
    x = c[0] + r * (
        c[1]
        + r
        * (
            c[2]
            + r * (c[3] + r * (c[4] + r * (c[5] + r * (c[6] + r * (c[7] + r * c[8])))))
        )
    )
    return x if y > 0 else -x


@dataclass
class RiskReport:
    """Aggregated risk metrics for a return series."""

    name: str
    var_95: float
    cvar_95: float
    sharpe: float
    sortino: float
    max_drawdown: float
    volatility: float
    annualized_return: float

    def calmar(self) -> float:
        if self.max_drawdown == 0:
            return 0.0
        return self.annualized_return / self.max_drawdown

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "annualized_return": self.annualized_return,
            "calmar": self.calmar(),
        }

    def is_acceptable(self, min_sharpe: float = 0.5, max_drawdown: float = 0.2) -> bool:
        return self.sharpe >= min_sharpe and self.max_drawdown <= max_drawdown
