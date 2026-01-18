#!/usr/bin/env python3
"""
Module 5 v2 vs v3 Backtest Comparison

Runs side-by-side backtesting of Module 5 v2 and v3 to measure:
- IC (Information Coefficient) differences
- Ranking stability
- Score distribution changes
- Feature contribution analysis

Usage:
    python backtest/compare_module5_versions.py
    python backtest/compare_module5_versions.py --start-date 2023-01-01 --end-date 2024-12-31
"""
import argparse
import json
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
from backtest.metrics import compute_spearman_ic, assign_quintiles, HORIZON_TRADING_DAYS

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
        # Larger companies tend to have better financial scores
        base_score = 50 + min(30, mcap / 5000)
        score = vary(base_score, t, "financial")

        # Determine severity based on score
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

        # Score based on phase
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
        validate_inputs=False,  # Skip validation for synthetic data
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


def compute_rank_correlation(scores_a: Dict[str, Decimal], scores_b: Dict[str, Decimal]) -> Optional[float]:
    """Compute rank correlation between two score sets."""
    common = set(scores_a.keys()) & set(scores_b.keys())
    if len(common) < 5:
        return None

    # Rank each set
    sorted_a = sorted(common, key=lambda t: scores_a[t], reverse=True)
    sorted_b = sorted(common, key=lambda t: scores_b[t], reverse=True)

    rank_a = {t: i for i, t in enumerate(sorted_a)}
    rank_b = {t: i for i, t in enumerate(sorted_b)}

    # Spearman correlation on ranks
    n = len(common)
    d_squared_sum = sum((rank_a[t] - rank_b[t]) ** 2 for t in common)

    rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    return rho


def compute_quintile_spread(
    result: Dict[str, Any],
    returns: Dict[str, float],
) -> Optional[Tuple[float, float, bool]]:
    """
    Compute Q5-Q1 spread and monotonicity.

    Returns: (q5_mean, q1_mean, is_monotonic)
    """
    ranked = result.get("ranked_securities", [])
    if len(ranked) < 10:
        return None

    # Assign quintiles
    quintiles = assign_quintiles(ranked)

    # Compute quintile returns
    q_returns = {i: [] for i in range(1, 6)}
    for sec in ranked:
        ticker = sec["ticker"]
        if ticker in returns and ticker in quintiles:
            q_returns[quintiles[ticker]].append(returns[ticker])

    # Compute means
    q_means = {}
    for q in range(1, 6):
        if len(q_returns[q]) >= 2:
            q_means[q] = mean(q_returns[q])

    if 1 not in q_means or 5 not in q_means:
        return None

    # Check monotonicity (higher quintile = higher return)
    is_monotonic = True
    for i in range(1, 5):
        if i in q_means and i+1 in q_means:
            if q_means[i] > q_means[i+1] + 0.001:
                is_monotonic = False
                break

    return q_means[5], q_means[1], is_monotonic


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
) -> Dict[str, Any]:
    """
    Run side-by-side backtest comparing Module 5 v2 and v3.
    """
    universe = universe or DEFAULT_UNIVERSE

    print("=" * 70)
    print("MODULE 5 v2 vs v3 BACKTEST COMPARISON")
    print("=" * 70)
    print(f"Start Date:    {start_date}")
    print(f"End Date:      {end_date}")
    print(f"Frequency:     {frequency_days} days")
    print(f"Universe:      {len(universe)} tickers")
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
    spread_v2 = []
    spread_v3 = []

    # Run backtest
    print("Running backtest...")
    print("-" * 70)

    for i, test_date in enumerate(test_dates):
        as_of_str = test_date.strftime("%Y-%m-%d")
        print(f"[{i+1}/{len(test_dates)}] {as_of_str}", end=" ")

        # Generate inputs
        inputs = generate_module_inputs(universe, as_of_str, seed=100 + i)

        # Run both versions
        try:
            result_v2 = run_module5_v2(inputs, as_of_str)
            result_v3 = run_module5_v3(inputs, as_of_str)
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        # Extract scores
        scores_v2 = extract_scores(result_v2)
        scores_v3 = extract_scores(result_v3)

        # Compute rank correlation between v2 and v3
        rank_corr = compute_rank_correlation(scores_v2, scores_v3)
        if rank_corr is not None:
            rank_correlations.append(rank_corr)

        # Get forward returns (90 days)
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

            # Compute spreads
            spread_result_v2 = compute_quintile_spread(result_v2, returns_90d)
            spread_result_v3 = compute_quintile_spread(result_v3, returns_90d)

            if spread_result_v2:
                spread_v2.append(spread_result_v2[0] - spread_result_v2[1])
            if spread_result_v3:
                spread_v3.append(spread_result_v3[0] - spread_result_v3[1])

            print(f"| IC_v2={float(ic_v2):.3f} IC_v3={float(ic_v3):.3f} rank_corr={rank_corr:.3f}" if ic_v2 and ic_v3 else "| (no IC)")
        else:
            print(f"| rank_corr={rank_corr:.3f}" if rank_corr else "")

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

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()

    # IC Summary
    print("Information Coefficient (90-day forward returns):")
    print("-" * 50)

    if ic_v2_90d:
        ic_v2_mean = mean(ic_v2_90d)
        ic_v2_std = stdev(ic_v2_90d) if len(ic_v2_90d) > 1 else 0
        ic_v2_pos = sum(1 for x in ic_v2_90d if x > 0) / len(ic_v2_90d) * 100
        print(f"  v2: Mean={ic_v2_mean:.4f}, Std={ic_v2_std:.4f}, Positive={ic_v2_pos:.1f}%, N={len(ic_v2_90d)}")
    else:
        ic_v2_mean = None
        print(f"  v2: Insufficient data")

    if ic_v3_90d:
        ic_v3_mean = mean(ic_v3_90d)
        ic_v3_std = stdev(ic_v3_90d) if len(ic_v3_90d) > 1 else 0
        ic_v3_pos = sum(1 for x in ic_v3_90d if x > 0) / len(ic_v3_90d) * 100
        print(f"  v3: Mean={ic_v3_mean:.4f}, Std={ic_v3_std:.4f}, Positive={ic_v3_pos:.1f}%, N={len(ic_v3_90d)}")
    else:
        ic_v3_mean = None
        print(f"  v3: Insufficient data")

    if ic_v2_mean is not None and ic_v3_mean is not None:
        ic_diff = ic_v3_mean - ic_v2_mean
        print()
        print(f"  IC Difference (v3 - v2): {ic_diff:+.4f}")
        if ic_diff > 0.02:
            print(f"  Assessment: v3 OUTPERFORMS v2")
        elif ic_diff < -0.02:
            print(f"  Assessment: v2 OUTPERFORMS v3")
        else:
            print(f"  Assessment: SIMILAR performance")

    print()

    # Spread Summary
    print("Q5-Q1 Spread (90-day forward returns):")
    print("-" * 50)

    if spread_v2:
        spread_v2_mean = mean(spread_v2)
        spread_v2_pos = sum(1 for x in spread_v2 if x > 0) / len(spread_v2) * 100
        print(f"  v2: Mean={spread_v2_mean:+.2%}, Positive={spread_v2_pos:.1f}%, N={len(spread_v2)}")
    else:
        spread_v2_mean = None
        print(f"  v2: Insufficient data")

    if spread_v3:
        spread_v3_mean = mean(spread_v3)
        spread_v3_pos = sum(1 for x in spread_v3 if x > 0) / len(spread_v3) * 100
        print(f"  v3: Mean={spread_v3_mean:+.2%}, Positive={spread_v3_pos:.1f}%, N={len(spread_v3)}")
    else:
        spread_v3_mean = None
        print(f"  v3: Insufficient data")

    print()

    # Rank Correlation
    print("Ranking Stability (v2 vs v3):")
    print("-" * 50)

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

    # Compile final results
    final_results = {
        "config": {
            "start_date": start_date,
            "end_date": end_date,
            "frequency_days": frequency_days,
            "universe_size": len(universe),
            "test_dates": len(test_dates),
        },
        "ic_comparison": {
            "v2": {
                "mean": ic_v2_mean,
                "std": stdev(ic_v2_90d) if len(ic_v2_90d) > 1 else None,
                "positive_pct": sum(1 for x in ic_v2_90d if x > 0) / len(ic_v2_90d) * 100 if ic_v2_90d else None,
                "values": ic_v2_90d,
            },
            "v3": {
                "mean": ic_v3_mean,
                "std": stdev(ic_v3_90d) if len(ic_v3_90d) > 1 else None,
                "positive_pct": sum(1 for x in ic_v3_90d if x > 0) / len(ic_v3_90d) * 100 if ic_v3_90d else None,
                "values": ic_v3_90d,
            },
            "difference": ic_v3_mean - ic_v2_mean if (ic_v2_mean and ic_v3_mean) else None,
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

    parser.add_argument(
        "--start-date",
        type=str,
        default="2023-01-01",
        help="Backtest start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="Backtest end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=30,
        help="Days between test dates"
    )
    parser.add_argument(
        "--price-file",
        type=str,
        default="data/daily_prices.csv",
        help="Path to price data CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="backtest_results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    try:
        results = run_comparison_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            frequency_days=args.frequency,
            price_file=args.price_file,
            output_dir=args.output_dir,
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
