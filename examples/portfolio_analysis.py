"""Portfolio analysis demo: risk metrics, Markowitz optimization, efficient frontier."""

from __future__ import annotations

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from patterns.portfolio import Asset, Portfolio, PortfolioWeights, RebalanceRule
from patterns.returns import ReturnSeries
from patterns.risk_metrics import (
    RiskReport,
    compute_cvar,
    compute_max_drawdown,
    compute_sharpe,
    compute_var,
)

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

ASSETS_META: list[tuple[str, str, float, float]] = [
    ("AAPL", "Technology", 0.00065, 0.016),
    ("MSFT", "Technology", 0.00060, 0.014),
    ("GOOGL", "Technology", 0.00055, 0.015),
    ("JPM", "Financials", 0.00040, 0.018),
    ("GS", "Financials", 0.00038, 0.020),
    ("JNJ", "Healthcare", 0.00030, 0.010),
    ("PFE", "Healthcare", 0.00028, 0.011),
    ("XOM", "Energy", 0.00035, 0.019),
    ("NEE", "Utilities", 0.00025, 0.009),
    ("BND", "Fixed Income", 0.00010, 0.004),
]

TRADING_DAYS = 252


def generate_returns(
    mean_daily: float, daily_vol: float, n_days: int = TRADING_DAYS, seed: int = 0
) -> list[float]:
    """Simulate daily log-normal returns using Box-Muller transform (stdlib only)."""
    random.seed(seed)
    returns: list[float] = []
    for i in range(n_days):
        # Box-Muller: two uniform draws → standard normal
        u1 = random.random() or 1e-15
        u2 = random.random() or 1e-15
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        r = mean_daily + daily_vol * z
        returns.append(r)
    return returns


def build_assets(seed_offset: int = 0) -> list[Asset]:
    """Create Asset objects with synthetic 1-year daily returns."""
    assets = []
    for i, (ticker, sector, mu, sigma) in enumerate(ASSETS_META):
        rets = generate_returns(mu, sigma, n_days=TRADING_DAYS, seed=i + seed_offset)
        a = Asset(ticker=ticker, name=ticker, sector=sector)
        a.set_returns(rets)
        assets.append(a)
    return assets


# ---------------------------------------------------------------------------
# Portfolio metric helpers
# ---------------------------------------------------------------------------


def portfolio_combined_returns(
    assets: list[Asset], weights: dict[str, float]
) -> list[float]:
    """Compute weighted daily portfolio returns (simple weighted sum)."""
    n = len(assets[0].returns)
    asset_map = {a.ticker: a.returns for a in assets}
    combined = []
    for day in range(n):
        r = sum(
            weights.get(ticker, 0.0) * rets[day] for ticker, rets in asset_map.items()
        )
        combined.append(r)
    return combined


def portfolio_vol(assets: list[Asset], weights: dict[str, float]) -> float:
    """Annualised portfolio volatility (weighted average of individual vols, no covariance)."""
    asset_map = {a.ticker: a for a in assets}
    vol = sum(weights.get(t, 0.0) * asset_map[t].volatility() for t in weights)
    return vol * math.sqrt(TRADING_DAYS)


def build_risk_report(name: str, rets: list[float]) -> RiskReport:
    """Construct a RiskReport from a daily return series."""
    ann_ret = (1 + sum(rets) / len(rets)) ** TRADING_DAYS - 1
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / max(1, len(rets) - 1)
    sigma_ann = math.sqrt(var * TRADING_DAYS)
    return RiskReport(
        name=name,
        var_95=compute_var(rets, 0.95),
        cvar_95=compute_cvar(rets, 0.95),
        sharpe=compute_sharpe(rets, risk_free_rate=0.04, periods_per_year=TRADING_DAYS),
        sortino=0.0,  # not needed for demo
        max_drawdown=compute_max_drawdown(rets),
        volatility=sigma_ann,
        annualized_return=ann_ret,
    )


# ---------------------------------------------------------------------------
# Simplified Markowitz optimisation (grid search over 2-asset frontier)
# ---------------------------------------------------------------------------


def markowitz_min_vol(assets: list[Asset]) -> dict[str, float]:
    """Return weights that minimise portfolio volatility (equal-vol inverse weighting)."""
    inv_vol = {a.ticker: 1.0 / max(a.volatility(), 1e-9) for a in assets}
    total = sum(inv_vol.values())
    return {t: v / total for t, v in inv_vol.items()}


def markowitz_max_sharpe(
    assets: list[Asset], rfr_daily: float = 0.04 / 252
) -> dict[str, float]:
    """Return weights that maximise Sharpe (Sharpe-score weighting heuristic)."""
    scores: dict[str, float] = {}
    for a in assets:
        vol = a.volatility()
        excess = a.mean_return() - rfr_daily
        scores[a.ticker] = max(excess / vol, 1e-9) if vol > 0 else 1e-9
    total = sum(scores.values())
    return {t: v / total for t, v in scores.items()}


def target_return_weights(
    assets: list[Asset], target_annual: float = 0.15
) -> dict[str, float]:
    """Blend min-vol and max-Sharpe weights to hit a target annualised return."""
    target_daily = (1 + target_annual) ** (1.0 / TRADING_DAYS) - 1
    min_v = markowitz_min_vol(assets)
    max_s = markowitz_max_sharpe(assets)

    best_weights = min_v
    best_gap = float("inf")
    for alpha in [i / 20 for i in range(21)]:
        blended = {
            t: alpha * max_s.get(t, 0.0) + (1 - alpha) * min_v.get(t, 0.0)
            for t in min_v
        }
        asset_map = {a.ticker: a for a in assets}
        port_ret = sum(blended[t] * asset_map[t].mean_return() for t in blended)
        gap = abs(port_ret - target_daily)
        if gap < best_gap:
            best_gap = gap
            best_weights = blended
    return best_weights


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_individual_risk_table(assets: list[Asset]) -> None:
    """Print per-asset Sharpe, VaR, CVaR, max drawdown."""
    print("\n  PER-ASSET RISK METRICS (1-year daily returns, annualised)")
    cols = (
        f"{'Ticker':>6}  {'Sector':>12}  {'Ann.Ret%':>8}  {'Vol%':>6}  "
        f"{'Sharpe':>6}  {'VaR95%':>7}  {'CVaR95%':>8}  {'MaxDD%':>7}"
    )
    hdr = '  ' + cols
    print("  " + "-" * (len(hdr) - 2))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for a in assets:
        rs = ReturnSeries(a.ticker, a.returns)
        stats = rs.stats()
        ann_ret = stats.annualized_return(TRADING_DAYS) * 100
        ann_vol = stats.annualized_vol(TRADING_DAYS) * 100
        sharpe = compute_sharpe(
            a.returns, risk_free_rate=0.04, periods_per_year=TRADING_DAYS
        )
        var95 = compute_var(a.returns, 0.95) * 100
        cvar95 = compute_cvar(a.returns, 0.95) * 100
        mdd = compute_max_drawdown(a.returns) * 100
        print(
            f"  {a.ticker:>6}  {a.sector:>12}  {ann_ret:>8.2f}  {ann_vol:>6.2f}  "
            f"{sharpe:>6.2f}  {var95:>7.3f}  {cvar95:>8.3f}  {mdd:>7.2f}"
        )
    print()


def print_efficient_frontier(
    assets: list[Asset],
    portfolios: list[tuple[str, dict[str, float]]],
) -> None:
    """Print three frontier portfolios and their risk/return profiles."""
    print("  EFFICIENT FRONTIER — 3 Key Portfolios")
    hdr = f"  {'Portfolio':>20}  {'Ann.Ret%':>8}  {'Ann.Vol%':>9}  {'Sharpe':>7}  {'VaR95%':>7}  {'MaxDD%':>7}"
    print("  " + "-" * (len(hdr) - 2))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for label, weights in portfolios:
        rets = portfolio_combined_returns(assets, weights)
        report = build_risk_report(label, rets)
        print(
            f"  {label:>20}  {report.annualized_return * 100:>8.2f}  "
            f"{report.volatility * 100:>9.2f}  {report.sharpe:>7.2f}  "
            f"{report.var_95 * 100:>7.3f}  {report.max_drawdown * 100:>7.2f}"
        )
    print()


def print_sector_weights(
    label: str, weights: dict[str, float], assets: list[Asset]
) -> None:
    """Print sector-level weight breakdown for a given portfolio."""
    sector_map: dict[str, float] = {}
    asset_map = {a.ticker: a.sector for a in assets}
    for ticker, w in weights.items():
        sector = asset_map.get(ticker, "Unknown")
        sector_map[sector] = sector_map.get(sector, 0.0) + w

    print(f"  Sector breakdown — {label}")
    for sector, w in sorted(sector_map.items(), key=lambda x: -x[1]):
        bar = "#" * int(w * 40)
        print(f"    {sector:>14}  {w * 100:>5.1f}%  {bar}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: full portfolio analysis pipeline."""
    print("\n" + "=" * 80)
    print("  QUANT FINANCE PYTHON COOKBOOK — Portfolio Analysis Demo")
    print("  10 assets × 252 trading days | stdlib only (math, random, statistics)")
    print("=" * 80)

    assets = build_assets()

    # 1. Individual asset risk metrics
    print_individual_risk_table(assets)

    # 2. Portfolio Construction — Portfolio class
    p = Portfolio("Demo Portfolio", RebalanceRule.QUARTERLY)
    eq_weights = PortfolioWeights.equal_weight([a.ticker for a in assets])
    for a in assets:
        p.add_asset(a)
    p.set_weights(eq_weights)

    print(
        f"  Equal-weight portfolio  |  assets: {p.asset_count()}  |  HHI: {p.concentration_hhi():.4f}"
        f"  |  Eff-N: {p.effective_n():.1f}"
    )
    eq_rets = portfolio_combined_returns(assets, eq_weights.weights)
    eq_report = build_risk_report("Equal Weight", eq_rets)
    print(
        f"  Ann.Return: {eq_report.annualized_return * 100:.2f}%  |  Ann.Vol: {eq_report.volatility * 100:.2f}%"
        f"  |  Sharpe: {eq_report.sharpe:.2f}\n"
    )

    # 3. Markowitz frontier
    min_v_w = markowitz_min_vol(assets)
    max_s_w = markowitz_max_sharpe(assets)
    tgt_w = target_return_weights(assets, target_annual=0.15)

    portfolios = [
        ("Min Volatility", min_v_w),
        ("Max Sharpe", max_s_w),
        ("Target 15% Ret", tgt_w),
    ]
    print_efficient_frontier(assets, portfolios)

    # 4. Sector allocation breakdown for Max Sharpe
    print_sector_weights("Max Sharpe", max_s_w, assets)

    print("  Done. All computations use pure Python stdlib.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
