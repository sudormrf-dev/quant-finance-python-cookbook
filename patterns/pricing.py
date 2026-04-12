"""Options pricing: Black-Scholes and Greeks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class OptionType(str, Enum):
    """Call or put option."""

    CALL = "call"
    PUT = "put"


class OptionStyle(str, Enum):
    """European or American option."""

    EUROPEAN = "european"
    AMERICAN = "american"


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass
class BlackScholesInputs:
    """Inputs for the Black-Scholes formula."""

    spot: float
    strike: float
    time_to_expiry: float  # years
    risk_free_rate: float
    volatility: float
    dividend_yield: float = 0.0

    def d1(self) -> float:
        t = self.time_to_expiry
        if t <= 0 or self.volatility <= 0:
            return 0.0
        return (
            math.log(self.spot / self.strike)
            + (self.risk_free_rate - self.dividend_yield + 0.5 * self.volatility**2) * t
        ) / (self.volatility * math.sqrt(t))

    def d2(self) -> float:
        return self.d1() - self.volatility * math.sqrt(self.time_to_expiry)

    def is_valid(self) -> bool:
        return (
            self.spot > 0
            and self.strike > 0
            and self.time_to_expiry > 0
            and self.volatility > 0
        )


@dataclass
class OptionContract:
    """An option contract specification."""

    option_type: OptionType
    style: OptionStyle
    spot: float
    strike: float
    time_to_expiry: float
    volatility: float
    risk_free_rate: float
    dividend_yield: float = 0.0
    multiplier: int = 100

    def inputs(self) -> BlackScholesInputs:
        return BlackScholesInputs(
            spot=self.spot,
            strike=self.strike,
            time_to_expiry=self.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            volatility=self.volatility,
            dividend_yield=self.dividend_yield,
        )

    def intrinsic_value(self) -> float:
        if self.option_type == OptionType.CALL:
            return max(0.0, self.spot - self.strike)
        return max(0.0, self.strike - self.spot)

    def is_itm(self) -> bool:
        return self.intrinsic_value() > 0

    def is_otm(self) -> bool:
        return self.intrinsic_value() == 0 and self.spot != self.strike

    def is_atm(self, tol: float = 0.01) -> bool:
        return abs(self.spot - self.strike) / self.spot < tol


def black_scholes_price(inputs: BlackScholesInputs, option_type: OptionType) -> float:
    """Black-Scholes option price."""
    if not inputs.is_valid():
        return 0.0
    s = inputs.spot
    k = inputs.strike
    t = inputs.time_to_expiry
    r = inputs.risk_free_rate
    q = inputs.dividend_yield
    d1 = inputs.d1()
    d2 = inputs.d2()
    if option_type == OptionType.CALL:
        return s * math.exp(-q * t) * _norm_cdf(d1) - k * math.exp(-r * t) * _norm_cdf(
            d2
        )
    return k * math.exp(-r * t) * _norm_cdf(-d2) - s * math.exp(-q * t) * _norm_cdf(-d1)


def black_scholes_greeks(
    inputs: BlackScholesInputs, option_type: OptionType
) -> dict[str, Any]:
    """Compute option Greeks: delta, gamma, vega, theta, rho."""
    if not inputs.is_valid():
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    s = inputs.spot
    k = inputs.strike
    t = inputs.time_to_expiry
    r = inputs.risk_free_rate
    q = inputs.dividend_yield
    sigma = inputs.volatility
    d1 = inputs.d1()
    d2 = inputs.d2()
    nd1 = _norm_pdf(d1)
    if option_type == OptionType.CALL:
        delta = math.exp(-q * t) * _norm_cdf(d1)
        theta = (
            -s * math.exp(-q * t) * nd1 * sigma / (2 * math.sqrt(t))
            - r * k * math.exp(-r * t) * _norm_cdf(d2)
            + q * s * math.exp(-q * t) * _norm_cdf(d1)
        ) / 365
        rho = k * t * math.exp(-r * t) * _norm_cdf(d2) / 100
    else:
        delta = -math.exp(-q * t) * _norm_cdf(-d1)
        theta = (
            -s * math.exp(-q * t) * nd1 * sigma / (2 * math.sqrt(t))
            + r * k * math.exp(-r * t) * _norm_cdf(-d2)
            - q * s * math.exp(-q * t) * _norm_cdf(-d1)
        ) / 365
        rho = -k * t * math.exp(-r * t) * _norm_cdf(-d2) / 100
    gamma = math.exp(-q * t) * nd1 / (s * sigma * math.sqrt(t))
    vega = s * math.exp(-q * t) * nd1 * math.sqrt(t) / 100
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
