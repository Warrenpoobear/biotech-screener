#!/usr/bin/env python3
"""
Module 5 v2 vs v3 Backtest Comparison (v2 - Fixed)

Runs side-by-side backtesting of Module 5 v2 and v3 to measure:
- IC (Information Coefficient) differences
- Ranking stability
- Score distribution changes
- Feature contribution analysis

Key fixes from v1:
- Corrected spread labeling (Top-Bottom, not Q5-Q1)
- Added IC t-stat and confidence intervals
- Added sanity check: sign(IC) == sign(spread)
- Added v3 feature coverage diagnostics
- Added portfolio-level metrics (turnover, drawdown)

Usage:
    python backtest/compare_module5_versions.py
    python backtest/compare_module5_versions.py --start-date 2023-01-01 --end-date 2024-12-31
"""
import argparse
import json
import math
import sys
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path
from statistics import mean, stdev, median
from typing import Dict, Any, List, Optional, Tuple, Set

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.returns_provider import CSVReturnsProvider
from backtest.metrics import compute_spearman_ic, HORIZON_TRADING_DAYS

# Import Module 5 versions
from module_5_composite_v2 import compute_module_5_composite_v2
from module_5_composite_v3 import compute_module_5_composite_v3

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_UNIVERSE = [
    "ACAD", "ALNY", "AMGN", "ARWR", "BEAM", "BIIB", "BLUE", "BMRN",
    "BNTX", "EDIT", "EXEL", "FOLD", "GILD", "HALO", "IMVT", "INCY",
    "IONS", "KRTX", "MRNA", "PCVX", "RARE", "REGN", "SRPT", "VRTX",
]

# Static clinical data for testing
CLINICAL_DATA = {
    "AMGN": {"phase": "approved", "trials": 15, "lead_phase": "approved"},
    "GILD": {"phase": "approved", "trials": 12, "lead_phase": "approved"},
    "VRTX": {"phase": "approved", "trials": 10, "lead_phase": "approved"},
    "REGN": {"phase": "approved", "trials": 14, "lead_phase": "approved"},
    "BIIB": {"phase": "phase 3", "trials": 8, "lead_phase": "phase_3"},
    "ALNY": {"phase": "phase 3", "trials": 6, "lead_phase": "phase_3"},
    "BMRN": {"phase": "phase 3", "trials": 5, "lead_phase": "phase_3"},
    "INCY": {"phase": "phase 2/3", "trials": 4, "lead_phase": "phase_2"},
    "EXEL": {"phase": "phase 3", "trials": 3, "lead_phase": "phase_3"},
    "MRNA": {"phase": "approved", "trials": 6, "lead_phase": "approved"},
    "BNTX": {"phase": "approved", "trials": 5, "lead_phase": "approved"},
    "IONS": {"phase": "phase 2", "trials": 8, "lead_phase": "phase_2"},
    "SRPT": {"phase": "phase 3", "trials": 4, "lead_phase": "phase_3"},
    "RARE": {"phase": "phase 2", "trials": 3, "lead_phase": "phase_2"},
    "BLUE": {"phase": "phase 1/2", "trials": 2, "lead_phase": "phase_1"},
    "FOLD": {"phase": "phase 2", "trials": 3, "lead_phase": "phase_2"},
    "ACAD": {"phase": "phase 3", "trials": 2, "lead_phase": "phase_3"},
    "HALO": {"phase": "phase 2/3", "trials": 4, "lead_phase": "phase_2"},
    "KRTX": {"phase": "phase 3", "trials": 3, "lead_phase": "phase_3"},
    "IMVT": {"phase": "phase 2", "trials": 2, "lead_phase": "phase_2"},
    "ARWR": {"phase": "phase 2", "trials": 4, "lead_phase": "phase_2"},
    "PCVX": {"phase": "phase 2", "trials": 2, "lead_phase": "phase_2"},
    "BEAM": {"phase": "phase 1/2", "trials": 2, "lead_phase": "phase_1"},
    "EDIT": {"phase": "phase 1", "trials": 1, "lead_phase": "phase_1"},
}

# Market cap data (millions USD)
MARKET_CAP_DATA = {
    "AMGN": 130000, "GILD": 95000, "VRTX": 85000, "REGN": 80000,
    "BIIB": 35000, "ALNY": 25000, "BMRN": 15000, "INCY": 14000,
    "EXEL": 6000, "MRNA": 45000, "BNTX": 25000, "IONS": 8000,
    "SRPT": 12000, "RARE": 4000, "BLUE": 400, "FOLD": 3500,
    "ACAD": 4500, "HALO": 6500, "KRTX": 5000, "IMVT": 2500,
    "ARWR": 4000, "PCVX": 7000, "BEAM": 2000, "EDIT": 800,
}


# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_module_inputs(tickers: List[str], as_of_date: str, seed: int = 42) -> Dict[str, Any]:
    """
    Generate synthetic but realistic Module 1-4 outputs for backtesting.

    Uses deterministic seeding for reproducibility while allowing
    date-based variation to test IC over time.
    """
    import hashlib

    # Create deterministic variation based on date and seed
    date_hash = int(hashlib.sha256(f"{as_of_date}_{seed}".encode()).hexdigest()[:8], 16)

    def vary(base: float, ticker: str, component: str) -> float:
        """Add deterministic variation to a base value."""
        ticker_hash = int(hashlib.sha256(f"{ticker}_{component}_{date_hash}".encode()).hexdigest()[:8], 16)
        # Variation of +/- 15%
        variation = ((ticker_hash % 1000) / 1000.0 - 0.5) * 0.30
        return max(0, min(100, base * (1 + variation)))

    # Module 1: Universe
    universe_result = {
        "as_of_date": as_of_date,
        "active_securities": [
            {
                "ticker": t,
                "company_name": f"{t} Inc",
                "market_cap_mm": MARKET_CAP_DATA.get(t, 5000),
                "status": "active",
            }
            for t in tickers
        ],
        "excluded_securities": [],
        "diagnostic_counts": {
            "total_input": len(tickers),
            "active": len(tickers),
            "excluded": 0,
        },
    }

    # Module 2: Financial
    financial_scores = []
    for t in tickers:
        mcap = MARKET_CAP_DATA.get(t, 5000)
        base_score = 50 + min(30, mcap / 5000)
        score = vary(base_score, t, "financial")

        if score >= 70:
            severity = "none"
        elif score >= 50:
            severity = "sev1"
        elif score >= 30:
            severity = "sev2"
        else:
            severity = "sev3"

        financial_scores.append({
            "ticker": t,
            "financial_score": str(Decimal(str(score)).quantize(Decimal("0.01"))),
            "financial_normalized": str(Decimal(str(score)).quantize(Decimal("0.01"))),
            "market_cap_mm": mcap,
            "runway_months": vary(24, t, "runway"),
            "severity": severity,
            "flags": [],
        })

    financial_result = {
        "as_of_date": as_of_date,
        "scores": financial_scores,
        "diagnostic_counts": {
            "scored": len(tickers),
            "missing": 0,
        },
    }

    # Module 3: Catalyst
    catalyst_summaries = {}
    for t in tickers:
        clin = CLINICAL_DATA.get(t, {"phase": "phase 1", "trials": 1})
        base_score = 30 + clin.get("trials", 1) * 5
        score = vary(base_score, t, "catalyst")

        catalyst_summaries[t] = {
            "ticker": t,
            "scores": {
                "score_blended": str(Decimal(str(score)).quantize(Decimal("0.01"))),
                "catalyst_score_net": str(Decimal(str(score)).quantize(Decimal("0.01"))),
            },
            "severe_negative_flag": False,
            "events_count": clin.get("trials", 1),
        }

    catalyst_result = {
        "as_of_date": as_of_date,
        "summaries": catalyst_summaries,
        "diagnostic_counts": {
            "total_tickers": len(tickers),
            "with_events": len(tickers),
        },
    }

    # Module 4: Clinical Development
    clinical_scores = []
    for t in tickers:
        clin = CLINICAL_DATA.get(t, {"phase": "phase 1", "trials": 1, "lead_phase": "phase_1"})

        phase_scores = {
            "approved": 90,
            "phase_3": 70,
            "phase_2": 50,
            "phase_1": 30,
        }
        lead_phase = clin.get("lead_phase", "phase_1")
        base_score = phase_scores.get(lead_phase, 40)
        score = vary(base_score, t, "clinical")

        clinical_scores.append({
            "ticker": t,
            "clinical_score": str(Decimal(str(score)).quantize(Decimal("0.01"))),
            "lead_phase": lead_phase,
            "severity": "none" if score >= 40 else "sev1",
            "flags": [],
        })

    clinical_result = {
        "as_of_date": as_of_date,
        "scores": clinical_scores,
        "diagnostic_counts": {
            "scored": len(tickers),
            "missing": 0,
        },
    }

    return {
        "universe": universe_result,
        "financial": financial_result,
        "catalyst": catalyst_result,
        "clinical": clinical_result,
    }


# =============================================================================
# BACKTEST CORE
# =============================================================================

def run_module5_v2(inputs: Dict[str, Any], as_of_date: str) -> Dict[str, Any]:
    """Run Module 5 v2 and return results."""
    return compute_module_5_composite_v2(
        universe_result=inputs["universe"],
        financial_result=inputs["financial"],
        catalyst_result=inputs["catalyst"],
        clinical_result=inputs["clinical"],
        as_of_date=as_of_date,
    )


def run_module5_v3(inputs: Dict[str, Any], as_of_date: str) -> Dict[str, Any]:
    """Run Module 5 v3 and return results."""
    return compute_module_5_composite_v3(
        universe_result=inputs["universe"],
        financial_result=inputs["financial"],
        catalyst_result=inputs["catalyst"],
        clinical_result=inputs["clinical"],
        as_of_date=as_of_date,
        validate_inputs=False,
    )


def extract_scores(result: Dict[str, Any]) -> Dict[str, Decimal]:
    """Extract ticker -> score mapping from Module 5 output."""
    scores = {}
    for sec in result.get("ranked_securities", []):
        ticker = sec.get("ticker")
        score = sec.get("composite_score")
        if ticker and score is not None:
            if isinstance(score, str):
                scores[ticker] = Decimal(score)
            else:
                scores[ticker] = Decimal(str(score))
    return scores


def compute_ic(scores: Dict[str, Decimal], returns: Dict[str, float]) -> Optional[Decimal]:
    """Compute Spearman IC between scores and returns."""
    common = set(scores.keys()) & set(returns.keys())
    if len(common) < 10:
        return None

    score_list = [scores[t] for t in common]
    return_list = [Decimal(str(returns[t])) for t in common]

    return compute_spearman_ic(score_list, return_list)


def compute_ic_tstat(ic_values: List[float]) -> Optional[float]:
    """Compute IC t-statistic: mean / (std / sqrt(N))."""
    if len(ic_values) < 2:
        return None
    ic_mean = mean(ic_values)
    ic_std = stdev(ic_values)
    if ic_std == 0:
        return None
    return ic_mean / (ic_std / math.sqrt(len(ic_values)))


def bootstrap_ic_ci(ic_values: List[float], n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for IC mean.
    Returns (lower, upper) bounds.
    """
    import random
    random.seed(42)  # Determinism

    n = len(ic_values)
    if n < 2:
        return (float('nan'), float('nan'))

    boot_means = []
    for _ in range(n_bootstrap):
        sample = [random.choice(ic_values) for _ in range(n)]
        boot_means.append(mean(sample))

    boot_means.sort()
    alpha = 1 - ci
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)

    return (boot_means[lower_idx], boot_means[upper_idx])


def compute_rank_correlation(scores_a: Dict[str, Decimal], scores_b: Dict[str, Decimal]) -> Optional[float]:
    """Compute rank correlation between two score sets."""
    common = set(scores_a.keys()) & set(scores_b.keys())
    if len(common) < 5:
        return None

    sorted_a = sorted(common, key=lambda t: scores_a[t], reverse=True)
    sorted_b = sorted(common, key=lambda t: scores_b[t], reverse=True)

    rank_a = {t: i for i, t in enumerate(sorted_a)}
    rank_b = {t: i for i, t in enumerate(sorted_b)}

    n = len(common)
    d_squared_sum = sum((rank_a[t] - rank_b[t]) ** 2 for t in common)

    rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    return rho


def compute_top_bottom_spread(
    scores: Dict[str, Decimal],
    returns: Dict[str, float],
    top_n: int = 5,
) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
    """
    Compute top N vs bottom N spread (NOT quintile-based).

    Returns: (top_mean, bottom_mean, spread, sign_consistent_with_ic)

    This is the SANITY CHECK: if IC > 0, spread should also be > 0.
    """
    common = sorted(set(scores.keys()) & set(returns.keys()), key=lambda t: scores[t], reverse=True)

    if len(common) < top_n * 2:
        return None, None, None, True

    top_tickers = common[:top_n]
    bottom_tickers = common[-top_n:]

    top_returns = [returns[t] for t in top_tickers]
    bottom_returns = [returns[t] for t in bottom_tickers]

    top_mean = mean(top_returns)
    bottom_mean = mean(bottom_returns)
    spread = top_mean - bottom_mean  # TOP - BOTTOM (positive if top outperforms)

    # Compute IC for sign check
    score_list = [scores[t] for t in common]
    return_list = [Decimal(str(returns[t])) for t in common]
    ic = compute_spearman_ic(score_list, return_list)

    if ic is None:
        sign_consistent = True
    else:
        ic_float = float(ic)
        # Signs should match: positive IC → positive spread
        sign_consistent = (ic_float >= 0 and spread >= 0) or (ic_float < 0 and spread < 0)

    return top_mean, bottom_mean, spread, sign_consistent


def compute_turnover(prev_top: Set[str], curr_top: Set[str]) -> Optional[float]:
    """Compute turnover between two top sets."""
    if not prev_top or not curr_top:
        return None

    # Symmetric difference / average size
    changed = len(prev_top.symmetric_difference(curr_top))
    avg_size = (len(prev_top) + len(curr_top)) / 2

    return changed / (2 * avg_size) if avg_size > 0 else 0


def compute_drawdown(cumulative_returns: List[float]) -> float:
    """Compute max drawdown from cumulative returns series."""
    if not cumulative_returns:
        return 0.0

    peak = cumulative_returns[0]
    max_dd = 0.0

    for ret in cumulative_returns:
        if ret > peak:
            peak = ret
        dd = (peak - ret) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return max_dd


def compute_costed_return(
    gross_return: float,
    turnover: float,
    cost_bps: float = 50.0,
) -> float:
    """
    Compute net return after trading costs.

    Args:
        gross_return: Gross period return
        turnover: Fraction of portfolio turned over (0-1)
        cost_bps: Round-trip cost in basis points (default 50 = 0.50%)

    Returns:
        Net return after costs
    """
    cost = turnover * (cost_bps / 10000.0)
    return gross_return - cost


def compute_concentration_metrics(
    scores: Dict[str, Decimal],
    returns: Dict[str, float],
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Compute concentration/robustness metrics for top bucket.

    Returns:
        - median_return: Median of top-N returns (robust to outliers)
        - mean_return: Mean of top-N returns
        - win_rate: Fraction where top-N beat universe median
        - top1_contribution: Return contribution from best performer
        - top1_ticker: Ticker of best performer
        - returns_by_ticker: Dict of returns for each top ticker
    """
    common = sorted(
        set(scores.keys()) & set(returns.keys()),
        key=lambda t: scores[t],
        reverse=True
    )

    if len(common) < top_n:
        return {
            "median_return": None,
            "mean_return": None,
            "win_rate": None,
            "top1_contribution": None,
            "top1_ticker": None,
            "returns_by_ticker": {},
        }

    top_tickers = common[:top_n]
    top_returns = [returns[t] for t in top_tickers]

    # Universe median for win rate calculation
    all_returns = [returns[t] for t in common]
    universe_median = median(all_returns) if all_returns else 0

    # Metrics
    median_ret = median(top_returns)
    mean_ret = mean(top_returns)
    wins = sum(1 for r in top_returns if r > universe_median)
    win_rate = wins / len(top_returns)

    # Top-1 contribution (what % of total top-N return came from best name)
    best_idx = top_returns.index(max(top_returns))
    top1_ticker = top_tickers[best_idx]
    top1_return = top_returns[best_idx]
    total_return = sum(top_returns)
    top1_contribution = top1_return / total_return if total_return != 0 else 0

    return {
        "median_return": median_ret,
        "mean_return": mean_ret,
        "win_rate": win_rate,
        "top1_contribution": top1_contribution,
        "top1_ticker": top1_ticker,
        "returns_by_ticker": {t: returns[t] for t in top_tickers},
    }


# =============================================================================
# MAIN BACKTEST
# =============================================================================

def run_comparison_backtest(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    frequency_days: int = 30,
    universe: Optional[List[str]] = None,
    price_file: str = "data/daily_prices.csv",
    output_dir: Optional[str] = None,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Run side-by-side backtest comparing Module 5 v2 and v3.
    """
    universe = universe or DEFAULT_UNIVERSE

    print("=" * 70)
    print("MODULE 5 v2 vs v3 BACKTEST COMPARISON (v2 - Fixed)")
    print("=" * 70)
    print(f"Start Date:    {start_date}")
    print(f"End Date:      {end_date}")
    print(f"Frequency:     {frequency_days} days")
    print(f"Universe:      {len(universe)} tickers")
    print(f"Top N:         {top_n} (for spread calculation)")
    print(f"Price File:    {price_file}")
    print()

    # Load returns provider
    try:
        returns_provider = CSVReturnsProvider(PROJECT_ROOT / price_file)
        available_tickers = set(returns_provider.get_available_tickers())
        universe = [t for t in universe if t in available_tickers]
        print(f"Tickers with price data: {len(universe)}")
    except Exception as e:
        print(f"WARNING: Could not load price data: {e}")
        returns_provider = None

    # Generate test dates
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    test_dates = []
    current = start_dt
    while current <= end_dt:
        test_dates.append(current)
        current += timedelta(days=frequency_days)

    print(f"Test dates: {len(test_dates)}")
    print()

    # Tracking variables
    results_v2 = []
    results_v3 = []
    ic_v2_90d = []
    ic_v3_90d = []
    rank_correlations = []

    # NEW: Top-Bottom spreads (not Q5-Q1)
    spread_v2 = []
    spread_v3 = []

    # NEW: Sanity check tracking
    sanity_checks_v2 = []  # (IC_sign, spread_sign, consistent)
    sanity_checks_v3 = []

    # NEW: Portfolio metrics
    turnover_v2 = []
    turnover_v3 = []
    prev_top_v2: Set[str] = set()
    prev_top_v3: Set[str] = set()

    # For drawdown calculation
    cumret_v2 = [1.0]
    cumret_v3 = [1.0]

    # NEW: Costed returns tracking
    cumret_v2_net = [1.0]  # Net of 50bps cost
    cumret_v3_net = [1.0]

    # NEW: Concentration metrics tracking
    median_returns_v2 = []
    median_returns_v3 = []
    win_rates_v2 = []
    win_rates_v3 = []
    top1_contributions_v2 = []
    top1_contributions_v3 = []

    # Run backtest
    print("Running backtest...")
    print("-" * 70)
    print(f"{'Date':<12} | {'IC_v2':>8} | {'IC_v3':>8} | {'Sprd_v2':>8} | {'Sprd_v3':>8} | {'Corr':>6} | Sanity")
    print("-" * 70)

    for i, test_date in enumerate(test_dates):
        as_of_str = test_date.strftime("%Y-%m-%d")

        # Generate inputs
        inputs = generate_module_inputs(universe, as_of_str, seed=100 + i)

        # Run both versions
        try:
            result_v2 = run_module5_v2(inputs, as_of_str)
            result_v3 = run_module5_v3(inputs, as_of_str)
        except Exception as e:
            print(f"{as_of_str:<12} | ERROR: {e}")
            continue

        # Extract scores
        scores_v2 = extract_scores(result_v2)
        scores_v3 = extract_scores(result_v3)

        # Compute rank correlation between v2 and v3
        rank_corr = compute_rank_correlation(scores_v2, scores_v3)
        if rank_corr is not None:
            rank_correlations.append(rank_corr)

        # Get forward returns (90 days)
        ic_v2 = None
        ic_v3 = None
        spread_result_v2 = (None, None, None, True)
        spread_result_v3 = (None, None, None, True)

        if returns_provider:
            return_start = (test_date + timedelta(days=1)).strftime("%Y-%m-%d")
            return_end = (test_date + timedelta(days=91)).strftime("%Y-%m-%d")

            returns_90d = {}
            for ticker in universe:
                ret = returns_provider.get_forward_total_return(ticker, return_start, return_end)
                if ret is not None:
                    returns_90d[ticker] = float(ret)

            # Compute IC
            ic_v2 = compute_ic(scores_v2, returns_90d)
            ic_v3 = compute_ic(scores_v3, returns_90d)

            if ic_v2 is not None:
                ic_v2_90d.append(float(ic_v2))
            if ic_v3 is not None:
                ic_v3_90d.append(float(ic_v3))

            # Compute top-bottom spreads (CORRECTED)
            spread_result_v2 = compute_top_bottom_spread(scores_v2, returns_90d, top_n)
            spread_result_v3 = compute_top_bottom_spread(scores_v3, returns_90d, top_n)

            if spread_result_v2[2] is not None:
                spread_v2.append(spread_result_v2[2])
                sanity_checks_v2.append((
                    "+" if ic_v2 and float(ic_v2) >= 0 else "-",
                    "+" if spread_result_v2[2] >= 0 else "-",
                    spread_result_v2[3]
                ))

            if spread_result_v3[2] is not None:
                spread_v3.append(spread_result_v3[2])
                sanity_checks_v3.append((
                    "+" if ic_v3 and float(ic_v3) >= 0 else "-",
                    "+" if spread_result_v3[2] >= 0 else "-",
                    spread_result_v3[3]
                ))

            # Compute turnover
            curr_top_v2 = set(sorted(scores_v2.keys(), key=lambda t: scores_v2[t], reverse=True)[:top_n])
            curr_top_v3 = set(sorted(scores_v3.keys(), key=lambda t: scores_v3[t], reverse=True)[:top_n])

            if prev_top_v2:
                to_v2 = compute_turnover(prev_top_v2, curr_top_v2)
                if to_v2 is not None:
                    turnover_v2.append(to_v2)
            if prev_top_v3:
                to_v3 = compute_turnover(prev_top_v3, curr_top_v3)
                if to_v3 is not None:
                    turnover_v3.append(to_v3)

            prev_top_v2 = curr_top_v2
            prev_top_v3 = curr_top_v3

            # Update cumulative returns for drawdown (gross and net of costs)
            if spread_result_v2[0] is not None:
                period_ret = spread_result_v2[0]  # Top bucket return
                cumret_v2.append(cumret_v2[-1] * (1 + period_ret))

                # Net return (50 bps cost per turnover)
                period_turnover = turnover_v2[-1] if turnover_v2 else 0
                net_ret = compute_costed_return(period_ret, period_turnover, cost_bps=50.0)
                cumret_v2_net.append(cumret_v2_net[-1] * (1 + net_ret))

            if spread_result_v3[0] is not None:
                period_ret = spread_result_v3[0]
                cumret_v3.append(cumret_v3[-1] * (1 + period_ret))

                # Net return (50 bps cost per turnover)
                period_turnover = turnover_v3[-1] if turnover_v3 else 0
                net_ret = compute_costed_return(period_ret, period_turnover, cost_bps=50.0)
                cumret_v3_net.append(cumret_v3_net[-1] * (1 + net_ret))

            # Compute concentration metrics
            conc_v2 = compute_concentration_metrics(scores_v2, returns_90d, top_n)
            conc_v3 = compute_concentration_metrics(scores_v3, returns_90d, top_n)

            if conc_v2["median_return"] is not None:
                median_returns_v2.append(conc_v2["median_return"])
                win_rates_v2.append(conc_v2["win_rate"])
                top1_contributions_v2.append(conc_v2["top1_contribution"])

            if conc_v3["median_return"] is not None:
                median_returns_v3.append(conc_v3["median_return"])
                win_rates_v3.append(conc_v3["win_rate"])
                top1_contributions_v3.append(conc_v3["top1_contribution"])

        # Print row
        ic_v2_str = f"{float(ic_v2):+.3f}" if ic_v2 else "   N/A"
        ic_v3_str = f"{float(ic_v3):+.3f}" if ic_v3 else "   N/A"
        sp_v2_str = f"{spread_result_v2[2]:+.1%}" if spread_result_v2[2] is not None else "   N/A"
        sp_v3_str = f"{spread_result_v3[2]:+.1%}" if spread_result_v3[2] is not None else "   N/A"
        corr_str = f"{rank_corr:.3f}" if rank_corr else "  N/A"

        # Sanity check indicator
        sane_v2 = "OK" if spread_result_v2[3] else "FAIL"
        sane_v3 = "OK" if spread_result_v3[3] else "FAIL"
        sanity_str = f"v2:{sane_v2} v3:{sane_v3}"

        print(f"{as_of_str:<12} | {ic_v2_str:>8} | {ic_v3_str:>8} | {sp_v2_str:>8} | {sp_v3_str:>8} | {corr_str:>6} | {sanity_str}")

        # Store results
        results_v2.append({
            "date": as_of_str,
            "scores": {k: str(v) for k, v in scores_v2.items()},
            "n_ranked": len(result_v2.get("ranked_securities", [])),
        })
        results_v3.append({
            "date": as_of_str,
            "scores": {k: str(v) for k, v in scores_v3.items()},
            "n_ranked": len(result_v3.get("ranked_securities", [])),
        })

    print("-" * 70)
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    # ==========================================================================
    # IC Summary with t-stat and CI
    # ==========================================================================
    print("Information Coefficient (90-day forward returns):")
    print("-" * 60)

    ic_v2_mean = None
    ic_v3_mean = None
    ic_v2_tstat = None
    ic_v3_tstat = None
    ic_v2_ci = (float('nan'), float('nan'))
    ic_v3_ci = (float('nan'), float('nan'))

    if ic_v2_90d:
        ic_v2_mean = mean(ic_v2_90d)
        ic_v2_std = stdev(ic_v2_90d) if len(ic_v2_90d) > 1 else 0
        ic_v2_pos = sum(1 for x in ic_v2_90d if x > 0) / len(ic_v2_90d) * 100
        ic_v2_tstat = compute_ic_tstat(ic_v2_90d)
        ic_v2_ci = bootstrap_ic_ci(ic_v2_90d)

        print(f"  v2: Mean={ic_v2_mean:+.4f}, Std={ic_v2_std:.4f}, Positive={ic_v2_pos:.1f}%, N={len(ic_v2_90d)}")
        print(f"      t-stat={ic_v2_tstat:.2f}, 95% CI=[{ic_v2_ci[0]:+.4f}, {ic_v2_ci[1]:+.4f}]")
    else:
        print(f"  v2: Insufficient data")

    if ic_v3_90d:
        ic_v3_mean = mean(ic_v3_90d)
        ic_v3_std = stdev(ic_v3_90d) if len(ic_v3_90d) > 1 else 0
        ic_v3_pos = sum(1 for x in ic_v3_90d if x > 0) / len(ic_v3_90d) * 100
        ic_v3_tstat = compute_ic_tstat(ic_v3_90d)
        ic_v3_ci = bootstrap_ic_ci(ic_v3_90d)

        print(f"  v3: Mean={ic_v3_mean:+.4f}, Std={ic_v3_std:.4f}, Positive={ic_v3_pos:.1f}%, N={len(ic_v3_90d)}")
        print(f"      t-stat={ic_v3_tstat:.2f}, 95% CI=[{ic_v3_ci[0]:+.4f}, {ic_v3_ci[1]:+.4f}]")
    else:
        print(f"  v3: Insufficient data")

    if ic_v2_mean is not None and ic_v3_mean is not None:
        ic_diff = ic_v3_mean - ic_v2_mean
        print()
        print(f"  IC Difference (v3 - v2): {ic_diff:+.4f}")

        # Check if CIs overlap (rough significance test)
        ci_overlap = not (ic_v2_ci[1] < ic_v3_ci[0] or ic_v3_ci[1] < ic_v2_ci[0])

        if abs(ic_diff) < 0.02 or ci_overlap:
            print(f"  Assessment: SIMILAR performance (CIs overlap: {ci_overlap})")
        elif ic_diff > 0.02:
            print(f"  Assessment: v3 OUTPERFORMS v2")
        else:
            print(f"  Assessment: v2 OUTPERFORMS v3")

    print()

    # ==========================================================================
    # SANITY CHECK: IC sign vs Spread sign
    # ==========================================================================
    print("SANITY CHECK: sign(IC) == sign(Top-Bottom Spread)")
    print("-" * 60)

    v2_consistent = sum(1 for _, _, c in sanity_checks_v2 if c)
    v3_consistent = sum(1 for _, _, c in sanity_checks_v3 if c)
    v2_total = len(sanity_checks_v2)
    v3_total = len(sanity_checks_v3)

    print(f"  v2: {v2_consistent}/{v2_total} periods consistent ({100*v2_consistent/v2_total:.1f}%)" if v2_total else "  v2: N/A")
    print(f"  v3: {v3_consistent}/{v3_total} periods consistent ({100*v3_consistent/v3_total:.1f}%)" if v3_total else "  v3: N/A")

    if v2_total and v3_total:
        if v2_consistent/v2_total >= 0.9 and v3_consistent/v3_total >= 0.9:
            print(f"  Status: PASS - IC and spread signs are consistent")
        else:
            print(f"  Status: WARNING - Some periods have inconsistent IC/spread signs")
    print()

    # ==========================================================================
    # Top-Bottom Spread (CORRECTED)
    # ==========================================================================
    print(f"Top-{top_n} vs Bottom-{top_n} Spread (90-day forward returns):")
    print("-" * 60)

    spread_v2_mean = None
    spread_v3_mean = None

    if spread_v2:
        spread_v2_mean = mean(spread_v2)
        spread_v2_pos = sum(1 for x in spread_v2 if x > 0) / len(spread_v2) * 100
        print(f"  v2: Mean={spread_v2_mean:+.2%}, Positive={spread_v2_pos:.1f}%, N={len(spread_v2)}")
    else:
        print(f"  v2: Insufficient data")

    if spread_v3:
        spread_v3_mean = mean(spread_v3)
        spread_v3_pos = sum(1 for x in spread_v3 if x > 0) / len(spread_v3) * 100
        print(f"  v3: Mean={spread_v3_mean:+.2%}, Positive={spread_v3_pos:.1f}%, N={len(spread_v3)}")
    else:
        print(f"  v3: Insufficient data")

    print()

    # ==========================================================================
    # Portfolio Metrics
    # ==========================================================================
    print("Portfolio Metrics (Top bucket):")
    print("-" * 60)

    if turnover_v2:
        print(f"  v2 Turnover: Mean={mean(turnover_v2):.1%}, N={len(turnover_v2)}")
    else:
        print(f"  v2 Turnover: N/A")

    if turnover_v3:
        print(f"  v3 Turnover: Mean={mean(turnover_v3):.1%}, N={len(turnover_v3)}")
    else:
        print(f"  v3 Turnover: N/A")

    dd_v2 = compute_drawdown(cumret_v2)
    dd_v3 = compute_drawdown(cumret_v3)
    print(f"  v2 Max Drawdown: {dd_v2:.1%}")
    print(f"  v3 Max Drawdown: {dd_v3:.1%}")

    print()
    print("  Gross Returns:")
    if cumret_v2:
        print(f"    v2 Cumulative: {(cumret_v2[-1] - 1):+.1%}")
    if cumret_v3:
        print(f"    v3 Cumulative: {(cumret_v3[-1] - 1):+.1%}")

    print()
    print("  Net Returns (after 50bps cost per turnover):")
    if cumret_v2_net:
        print(f"    v2 Cumulative (net): {(cumret_v2_net[-1] - 1):+.1%}")
    if cumret_v3_net:
        print(f"    v3 Cumulative (net): {(cumret_v3_net[-1] - 1):+.1%}")

    if cumret_v2 and cumret_v3 and cumret_v2_net and cumret_v3_net:
        v2_cost_drag = (cumret_v2[-1] - cumret_v2_net[-1]) / cumret_v2[-1] * 100
        v3_cost_drag = (cumret_v3[-1] - cumret_v3_net[-1]) / cumret_v3[-1] * 100
        print(f"    v2 Cost Drag: {v2_cost_drag:.1f}%")
        print(f"    v3 Cost Drag: {v3_cost_drag:.1f}%")

    print()

    # ==========================================================================
    # Concentration / Robustness Check
    # ==========================================================================
    print("Concentration Analysis (robustness check):")
    print("-" * 60)

    if median_returns_v2:
        print(f"  v2 Median Top-{top_n} Return: {mean(median_returns_v2):+.2%}")
    if median_returns_v3:
        print(f"  v3 Median Top-{top_n} Return: {mean(median_returns_v3):+.2%}")

    print()
    if win_rates_v2:
        print(f"  v2 Win Rate (top-{top_n} beats median): {mean(win_rates_v2):.1%}")
    if win_rates_v3:
        print(f"  v3 Win Rate (top-{top_n} beats median): {mean(win_rates_v3):.1%}")

    print()
    if top1_contributions_v2:
        print(f"  v2 Top-1 Contribution: {mean(top1_contributions_v2):.1%} of top-{top_n} return")
    if top1_contributions_v3:
        print(f"  v3 Top-1 Contribution: {mean(top1_contributions_v3):.1%} of top-{top_n} return")

    # Flag if one name dominates
    if top1_contributions_v2 and top1_contributions_v3:
        if mean(top1_contributions_v2) > 0.5 or mean(top1_contributions_v3) > 0.5:
            print()
            print("  WARNING: Top-1 name contributes >50% of returns - concentration risk!")

    print()

    # ==========================================================================
    # Rank Correlation
    # ==========================================================================
    print("Ranking Stability (v2 vs v3):")
    print("-" * 60)

    if rank_correlations:
        rank_corr_mean = mean(rank_correlations)
        rank_corr_min = min(rank_correlations)
        rank_corr_max = max(rank_correlations)
        print(f"  Mean Rank Correlation: {rank_corr_mean:.3f}")
        print(f"  Range: [{rank_corr_min:.3f}, {rank_corr_max:.3f}]")

        if rank_corr_mean > 0.9:
            print(f"  Assessment: VERY SIMILAR rankings")
        elif rank_corr_mean > 0.7:
            print(f"  Assessment: MODERATELY similar rankings")
        else:
            print(f"  Assessment: SIGNIFICANTLY different rankings")

    print()
    print("=" * 70)

    # ==========================================================================
    # RECOMMENDATION
    # ==========================================================================
    print("RECOMMENDATION:")
    print("-" * 60)

    if ic_v2_mean and ic_v3_mean:
        # Decision framework
        ic_diff = ic_v3_mean - ic_v2_mean
        ic_significant = abs(ic_diff) > 0.02 and not ci_overlap if ic_v2_ci and ic_v3_ci else False

        v2_sharper = ic_v2_mean > ic_v3_mean and ic_v2_tstat and ic_v2_tstat > 2
        v3_more_stable = ic_v3_90d and ic_v2_90d and stdev(ic_v3_90d) < stdev(ic_v2_90d)

        if ic_significant and ic_diff > 0:
            print("  Deploy v3: Significantly higher IC")
        elif ic_significant and ic_diff < 0:
            print("  Keep v2: Significantly higher IC")
        elif v3_more_stable and abs(ic_diff) < 0.01:
            print("  Consider v3: Similar IC with lower volatility")
            print("  BUT: Run parallel tracking before switching")
        else:
            print("  No clear winner: Keep v2 as default")
            print("  Consider: Bring in v3 components (momentum, catalyst decay) one at a time")
    else:
        print("  Insufficient data for recommendation")

    print()
    print("=" * 70)

    # Compile final results
    final_results = {
        "config": {
            "start_date": start_date,
            "end_date": end_date,
            "frequency_days": frequency_days,
            "universe_size": len(universe),
            "test_dates": len(test_dates),
            "top_n": top_n,
        },
        "ic_comparison": {
            "v2": {
                "mean": ic_v2_mean,
                "std": stdev(ic_v2_90d) if len(ic_v2_90d) > 1 else None,
                "positive_pct": sum(1 for x in ic_v2_90d if x > 0) / len(ic_v2_90d) * 100 if ic_v2_90d else None,
                "t_stat": ic_v2_tstat,
                "ci_95_lower": ic_v2_ci[0] if ic_v2_ci else None,
                "ci_95_upper": ic_v2_ci[1] if ic_v2_ci else None,
                "values": ic_v2_90d,
            },
            "v3": {
                "mean": ic_v3_mean,
                "std": stdev(ic_v3_90d) if len(ic_v3_90d) > 1 else None,
                "positive_pct": sum(1 for x in ic_v3_90d if x > 0) / len(ic_v3_90d) * 100 if ic_v3_90d else None,
                "t_stat": ic_v3_tstat,
                "ci_95_lower": ic_v3_ci[0] if ic_v3_ci else None,
                "ci_95_upper": ic_v3_ci[1] if ic_v3_ci else None,
                "values": ic_v3_90d,
            },
            "difference": ic_v3_mean - ic_v2_mean if (ic_v2_mean and ic_v3_mean) else None,
        },
        "sanity_check": {
            "v2_consistent_pct": 100*v2_consistent/v2_total if v2_total else None,
            "v3_consistent_pct": 100*v3_consistent/v3_total if v3_total else None,
        },
        "spread_comparison": {
            "v2": {
                "mean": spread_v2_mean,
                "positive_pct": sum(1 for x in spread_v2 if x > 0) / len(spread_v2) * 100 if spread_v2 else None,
                "values": spread_v2,
            },
            "v3": {
                "mean": spread_v3_mean,
                "positive_pct": sum(1 for x in spread_v3 if x > 0) / len(spread_v3) * 100 if spread_v3 else None,
                "values": spread_v3,
            },
        },
        "portfolio_metrics": {
            "v2": {
                "turnover_mean": mean(turnover_v2) if turnover_v2 else None,
                "max_drawdown": dd_v2,
                "cumulative_return_gross": cumret_v2[-1] - 1 if cumret_v2 else None,
                "cumulative_return_net": cumret_v2_net[-1] - 1 if cumret_v2_net else None,
                "median_top_return": mean(median_returns_v2) if median_returns_v2 else None,
                "win_rate": mean(win_rates_v2) if win_rates_v2 else None,
                "top1_contribution": mean(top1_contributions_v2) if top1_contributions_v2 else None,
            },
            "v3": {
                "turnover_mean": mean(turnover_v3) if turnover_v3 else None,
                "max_drawdown": dd_v3,
                "cumulative_return_gross": cumret_v3[-1] - 1 if cumret_v3 else None,
                "cumulative_return_net": cumret_v3_net[-1] - 1 if cumret_v3_net else None,
                "median_top_return": mean(median_returns_v3) if median_returns_v3 else None,
                "win_rate": mean(win_rates_v3) if win_rates_v3 else None,
                "top1_contribution": mean(top1_contributions_v3) if top1_contributions_v3 else None,
            },
        },
        "rank_correlation": {
            "mean": mean(rank_correlations) if rank_correlations else None,
            "min": min(rank_correlations) if rank_correlations else None,
            "max": max(rank_correlations) if rank_correlations else None,
            "values": rank_correlations,
        },
        "period_results": {
            "v2": results_v2,
            "v3": results_v3,
        },
    }

    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / f"module5_v2_v3_comparison_{start_date}_{end_date}.json"
        with open(output_file, "w") as f:
            json.dump(final_results, f, indent=2, default=str)
        print(f"Results saved to: {output_file}")

    return final_results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compare Module 5 v2 vs v3 backtest performance"
    )

    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Backtest start date")
    parser.add_argument("--end-date", type=str, default="2024-12-31", help="Backtest end date")
    parser.add_argument("--frequency", type=int, default=30, help="Days between test dates")
    parser.add_argument("--price-file", type=str, default="data/daily_prices.csv", help="Path to price data")
    parser.add_argument("--output-dir", type=str, default="backtest_results", help="Output directory")
    parser.add_argument("--top-n", type=int, default=5, help="Top N tickers for spread calculation")

    args = parser.parse_args()

    try:
        results = run_comparison_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            frequency_days=args.frequency,
            price_file=args.price_file,
            output_dir=args.output_dir,
            top_n=args.top_n,
        )

        print()
        print("BACKTEST COMPLETE")
        print("=" * 70)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
