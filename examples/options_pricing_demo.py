"""Options pricing demo: Black-Scholes pricing and Greeks for a synthetic portfolio."""

from __future__ import annotations

import math
import os
import random
import sys

# Allow running from repo root or examples/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from patterns.pricing import (
    OptionContract,
    OptionStyle,
    OptionType,
    black_scholes_greeks,
    black_scholes_price,
)


def generate_synthetic_options(n: int = 20, seed: int = 42) -> list[OptionContract]:
    """Generate *n* synthetic option contracts spanning calls, puts, ITM/OTM/ATM."""
    random.seed(seed)
    contracts: list[OptionContract] = []
    spots = [90.0, 95.0, 100.0, 105.0, 110.0]
    strikes = [85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0]
    option_types = [OptionType.CALL, OptionType.PUT]

    for i in range(n):
        spot = random.choice(spots)
        strike = random.choice(strikes)
        otype = option_types[i % 2]
        tte = round(random.uniform(0.05, 1.0), 4)  # 18 days to 1 year
        vol = round(random.uniform(0.12, 0.55), 4)  # 12% to 55% IV
        rfr = round(random.uniform(0.02, 0.06), 4)  # 2% to 6%
        div = round(random.uniform(0.0, 0.03), 4)  # 0% to 3%
        contracts.append(
            OptionContract(
                option_type=otype,
                style=OptionStyle.EUROPEAN,
                spot=spot,
                strike=strike,
                time_to_expiry=tte,
                volatility=vol,
                risk_free_rate=rfr,
                dividend_yield=div,
            )
        )
    return contracts


def _moneyness_label(contract: OptionContract) -> str:
    """Return ATM / ITM / OTM label."""
    if contract.is_atm():
        return "ATM"
    if contract.is_itm():
        return "ITM"
    return "OTM"


def print_options_table(contracts: list[OptionContract]) -> None:
    """Print a formatted table of option prices and Greeks."""
    header = (
        f"{'#':>3}  {'Type':>4}  {'Spot':>6}  {'Strike':>6}  {'TTE(y)':>6}  "
        f"{'IV%':>5}  {'Status':>3}  {'Price':>7}  {'Delta':>7}  "
        f"{'Gamma':>7}  {'Vega':>6}  {'Theta':>7}"
    )
    sep = "-" * len(header)
    print(sep)
    print("  OPTIONS PRICING DEMO — 20 Synthetic European Contracts")
    print(sep)
    print(header)
    print(sep)

    total_value = 0.0
    total_delta = 0.0
    total_gamma = 0.0

    for idx, c in enumerate(contracts, start=1):
        inp = c.inputs()
        price = black_scholes_price(inp, c.option_type)
        greeks = black_scholes_greeks(inp, c.option_type)
        notional_value = price * c.multiplier
        total_value += notional_value
        total_delta += greeks["delta"]
        total_gamma += greeks["gamma"]

        print(
            f"{idx:>3}  {c.option_type.value.upper()[:4]:>4}  {c.spot:>6.1f}  "
            f"{c.strike:>6.1f}  {c.time_to_expiry:>6.3f}  "
            f"{c.volatility * 100:>5.1f}  {_moneyness_label(c):>3}  "
            f"{price:>7.3f}  {greeks['delta']:>7.4f}  "
            f"{greeks['gamma']:>7.4f}  {greeks['vega']:>6.3f}  {greeks['theta']:>7.4f}"
        )

    print(sep)
    print(
        f"  Portfolio notional value (x{contracts[0].multiplier} multiplier): ${total_value:,.2f}"
    )
    print(f"  Net delta: {total_delta:+.4f}  |  Net gamma: {total_gamma:.4f}")
    print(sep)


def greeks_sensitivity_demo(base_spot: float = 100.0) -> None:
    """Show how delta and gamma change as the spot moves ±20%."""
    print("\n  DELTA / GAMMA PROFILE — ATM Call as spot moves")
    print("  (Strike=100, TTE=0.25yr, IV=25%, RFR=4%)\n")
    print(
        f"  {'Spot':>6}  {'Moneyness':>10}  {'Price':>7}  {'Delta':>7}  {'Gamma':>7}  {'Vega':>6}"
    )
    print("  " + "-" * 55)

    from patterns.pricing import BlackScholesInputs

    for offset in range(-20, 22, 5):
        spot = base_spot + offset
        inp = BlackScholesInputs(
            spot=spot,
            strike=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.04,
            volatility=0.25,
        )
        price = black_scholes_price(inp, OptionType.CALL)
        g = black_scholes_greeks(inp, OptionType.CALL)
        diff = spot - 100.0
        label = f"{diff:+.0f}%" if diff != 0 else "  ATM"
        print(
            f"  {spot:>6.1f}  {label:>10}  {price:>7.3f}  "
            f"{g['delta']:>7.4f}  {g['gamma']:>7.4f}  {g['vega']:>6.3f}"
        )


def put_call_parity_check(contracts: list[OptionContract]) -> None:
    """Verify put-call parity holds for matching pairs within the synthetic set."""
    from patterns.pricing import BlackScholesInputs

    print("\n  PUT-CALL PARITY VERIFICATION (sample of 5 pairs)")
    print("  C - P = S*e^(-q*T) - K*e^(-r*T)  (should be near 0 difference)\n")
    print(
        f"  {'Spot':>6}  {'Strike':>6}  {'TTE':>6}  {'Call':>7}  {'Put':>6}  {'LHS-RHS':>9}"
    )
    print("  " + "-" * 55)

    random.seed(0)
    for _ in range(5):
        spot = random.choice([90.0, 95.0, 100.0, 105.0, 110.0])
        strike = random.choice([90.0, 100.0, 110.0])
        tte = round(random.uniform(0.1, 0.5), 3)
        vol, rfr, div = 0.20, 0.04, 0.01
        inp = BlackScholesInputs(
            spot=spot,
            strike=strike,
            time_to_expiry=tte,
            risk_free_rate=rfr,
            volatility=vol,
            dividend_yield=div,
        )
        call_p = black_scholes_price(inp, OptionType.CALL)
        put_p = black_scholes_price(inp, OptionType.PUT)
        lhs = call_p - put_p
        rhs = spot * math.exp(-div * tte) - strike * math.exp(-rfr * tte)
        diff = lhs - rhs
        print(
            f"  {spot:>6.1f}  {strike:>6.1f}  {tte:>6.3f}  {call_p:>7.3f}  {put_p:>6.3f}  {diff:>+9.6f}"
        )


def main() -> None:
    """Entry point: run the full options pricing demo."""
    print("\n" + "=" * 80)
    print("  QUANT FINANCE PYTHON COOKBOOK — Options Pricing Demo")
    print("=" * 80)

    contracts = generate_synthetic_options(n=20)
    print_options_table(contracts)
    greeks_sensitivity_demo()
    put_call_parity_check(contracts)

    print("\n  Done. All computations use pure Python stdlib (math, random).")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
