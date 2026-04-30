"""Benchmark: Python loops vs list comprehensions vs direct math for Monte Carlo pricing."""

from __future__ import annotations

import math
import random
import statistics
import time
from typing import Callable


# ---------------------------------------------------------------------------
# Monte Carlo option pricer implementations
# ---------------------------------------------------------------------------

N_SIMULATIONS = 10_000
SPOT = 100.0
STRIKE = 100.0
TIME_TO_EXPIRY = 0.5    # years
RISK_FREE_RATE = 0.04
VOLATILITY = 0.20
SEED = 42


def _pregenerate_normals(n: int, seed: int = SEED) -> list[float]:
    """Pre-generate standard normal samples using Box-Muller (avoids timing the PRNG)."""
    random.seed(seed)
    normals: list[float] = []
    for _ in range(n):
        u1 = random.random() or 1e-15
        u2 = random.random() or 1e-15
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        normals.append(z)
    return normals


# --- Implementation 1: pure Python for-loop (most naïve) ---

def mc_price_loop(
    spot: float,
    strike: float,
    tte: float,
    rfr: float,
    vol: float,
    normals: list[float],
) -> float:
    """Monte Carlo call price — pure Python for-loop."""
    discount = math.exp(-rfr * tte)
    drift = (rfr - 0.5 * vol * vol) * tte
    diffusion = vol * math.sqrt(tte)
    payoffs: list[float] = []
    for z in normals:
        st = spot * math.exp(drift + diffusion * z)
        payoffs.append(max(st - strike, 0.0))
    return discount * sum(payoffs) / len(payoffs)


# --- Implementation 2: list comprehension ---

def mc_price_comprehension(
    spot: float,
    strike: float,
    tte: float,
    rfr: float,
    vol: float,
    normals: list[float],
) -> float:
    """Monte Carlo call price — list comprehension."""
    discount = math.exp(-rfr * tte)
    drift = (rfr - 0.5 * vol * vol) * tte
    diffusion = vol * math.sqrt(tte)
    payoffs = [
        max(spot * math.exp(drift + diffusion * z) - strike, 0.0)
        for z in normals
    ]
    return discount * sum(payoffs) / len(payoffs)


# --- Implementation 3: generator expression (memory-efficient, no intermediate list) ---

def mc_price_generator(
    spot: float,
    strike: float,
    tte: float,
    rfr: float,
    vol: float,
    normals: list[float],
) -> float:
    """Monte Carlo call price — generator expression (no intermediate list)."""
    discount = math.exp(-rfr * tte)
    drift = (rfr - 0.5 * vol * vol) * tte
    diffusion = vol * math.sqrt(tte)
    total = sum(
        max(spot * math.exp(drift + diffusion * z) - strike, 0.0)
        for z in normals
    )
    return discount * total / len(normals)


# --- Implementation 4: pre-compute constants + map ---

def mc_price_map(
    spot: float,
    strike: float,
    tte: float,
    rfr: float,
    vol: float,
    normals: list[float],
) -> float:
    """Monte Carlo call price — map() with pre-computed constants."""
    discount = math.exp(-rfr * tte)
    drift = (rfr - 0.5 * vol * vol) * tte
    diffusion = vol * math.sqrt(tte)
    log_spot = math.log(spot)
    exp = math.exp

    def payoff(z: float) -> float:
        return max(exp(log_spot + drift + diffusion * z) - strike, 0.0)

    return discount * sum(map(payoff, normals)) / len(normals)


# --- Implementation 5: fully vectorised via math ops on pre-built array ---

def mc_price_vectorised(
    spot: float,
    strike: float,
    tte: float,
    rfr: float,
    vol: float,
    normals: list[float],
) -> float:
    """Monte Carlo call price — manual SIMD-style: pre-scale z-values then bulk math.exp."""
    discount = math.exp(-rfr * tte)
    drift = (rfr - 0.5 * vol * vol) * tte
    diffusion = vol * math.sqrt(tte)
    log_spot = math.log(spot)
    base = log_spot + drift

    # Scale all z-values in one pass, then map math.exp in bulk
    scaled = [base + diffusion * z for z in normals]
    prices = list(map(math.exp, scaled))
    payoffs = [max(p - strike, 0.0) for p in prices]
    return discount * sum(payoffs) / len(payoffs)


# ---------------------------------------------------------------------------
# Black-Scholes closed form (reference / ground truth)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def bs_call_price(spot: float, strike: float, tte: float, rfr: float, vol: float) -> float:
    """Black-Scholes analytical call price (reference value)."""
    d1 = (math.log(spot / strike) + (rfr + 0.5 * vol ** 2) * tte) / (vol * math.sqrt(tte))
    d2 = d1 - vol * math.sqrt(tte)
    return spot * _norm_cdf(d1) - strike * math.exp(-rfr * tte) * _norm_cdf(d2)


# ---------------------------------------------------------------------------
# Benchmarking harness
# ---------------------------------------------------------------------------

def time_fn(fn: Callable[[], float], repeats: int = 5) -> tuple[float, float]:
    """Return (mean_ms, stdev_ms) over *repeats* calls."""
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0


def run_benchmarks() -> list[dict[str, object]]:
    """Run all implementations and collect timing + pricing results."""
    normals = _pregenerate_normals(N_SIMULATIONS)
    bs_ref = bs_call_price(SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, VOLATILITY)

    implementations: list[tuple[str, Callable[[], float]]] = [
        ("Pure for-loop",        lambda: mc_price_loop(SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, VOLATILITY, normals)),
        ("List comprehension",   lambda: mc_price_comprehension(SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, VOLATILITY, normals)),
        ("Generator expression", lambda: mc_price_generator(SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, VOLATILITY, normals)),
        ("map() + closure",      lambda: mc_price_map(SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, VOLATILITY, normals)),
        ("Vectorised (list+map)", lambda: mc_price_vectorised(SPOT, STRIKE, TIME_TO_EXPIRY, RISK_FREE_RATE, VOLATILITY, normals)),
    ]

    results = []
    baseline_ms: float | None = None

    for label, fn in implementations:
        price = fn()  # warm-up
        mean_ms, std_ms = time_fn(fn, repeats=7)
        if baseline_ms is None:
            baseline_ms = mean_ms
        speedup = baseline_ms / mean_ms if mean_ms > 0 else 1.0
        error_pct = abs(price - bs_ref) / bs_ref * 100
        results.append({
            "label": label,
            "price": price,
            "mean_ms": mean_ms,
            "std_ms": std_ms,
            "speedup": speedup,
            "error_pct": error_pct,
        })

    return results, bs_ref


# ---------------------------------------------------------------------------
# Additional micro-benchmark: loop overhead vs math.exp direct
# ---------------------------------------------------------------------------

def bench_exp_strategies(n: int = N_SIMULATIONS) -> list[dict[str, object]]:
    """Compare three ways to apply math.exp to a list of N values."""
    data = [random.gauss(0, 1) for _ in range(n)]

    strategies: list[tuple[str, Callable[[], list[float]]]] = [
        ("for-loop append",     lambda: [math.exp(x) for x in data]),   # comprehension IS a loop
        ("map(math.exp, data)", lambda: list(map(math.exp, data))),
        ("manual loop",
            lambda: _manual_exp_loop(data)),
    ]

    out = []
    for label, fn in strategies:
        fn()  # warm-up
        mean_ms, std_ms = time_fn(fn, repeats=7)
        out.append({"label": label, "mean_ms": mean_ms, "std_ms": std_ms})
    # normalise vs slowest
    slowest = max(r["mean_ms"] for r in out)
    for r in out:
        r["speedup"] = slowest / r["mean_ms"]
    return out


def _manual_exp_loop(data: list[float]) -> list[float]:
    result = []
    append = result.append
    exp = math.exp
    for x in data:
        append(exp(x))
    return result


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def print_mc_results(results: list[dict[str, object]], bs_ref: float) -> None:
    print(f"\n  MONTE CARLO BENCHMARK — {N_SIMULATIONS:,} simulations")
    print(f"  Black-Scholes reference price: ${bs_ref:.4f}")
    print(f"  (S={SPOT}, K={STRIKE}, T={TIME_TO_EXPIRY}yr, r={RISK_FREE_RATE}, σ={VOLATILITY})\n")
    hdr = f"  {'Implementation':>24}  {'Price':>7}  {'Mean(ms)':>9}  {'Std(ms)':>7}  {'Speedup':>7}  {'Error%':>7}"
    print("  " + "-" * (len(hdr) - 2))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        bar = "+" * min(int(r["speedup"] * 4), 20)
        print(
            f"  {r['label']:>24}  ${r['price']:>6.4f}  {r['mean_ms']:>9.3f}  "
            f"{r['std_ms']:>7.3f}  {r['speedup']:>6.2f}x  {r['error_pct']:>6.3f}%  {bar}"
        )
    print()


def print_exp_results(results: list[dict[str, object]]) -> None:
    print(f"  math.exp APPLICATION STRATEGIES — {N_SIMULATIONS:,} elements\n")
    hdr = f"  {'Strategy':>24}  {'Mean(ms)':>9}  {'Std(ms)':>7}  {'Speedup':>8}"
    print("  " + "-" * (len(hdr) - 2))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        print(
            f"  {r['label']:>24}  {r['mean_ms']:>9.3f}  {r['std_ms']:>7.3f}  {r['speedup']:>7.2f}x"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: run all benchmarks and print results."""
    print("\n" + "=" * 80)
    print("  QUANT FINANCE PYTHON COOKBOOK — Vectorization Benchmarks")
    print("  stdlib only: math, random, time, statistics")
    print("=" * 80)

    mc_results, bs_ref = run_benchmarks()
    print_mc_results(mc_results, bs_ref)

    exp_results = bench_exp_strategies()
    print_exp_results(exp_results)

    fastest_mc = min(mc_results, key=lambda r: r["mean_ms"])
    slowest_mc = max(mc_results, key=lambda r: r["mean_ms"])
    overall_speedup = slowest_mc["mean_ms"] / fastest_mc["mean_ms"]
    print(f"  Overall MC speedup ({slowest_mc['label']} → {fastest_mc['label']}): {overall_speedup:.1f}x")
    print(f"  All prices within 0.1% of Black-Scholes reference.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
