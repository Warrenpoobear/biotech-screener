"""
Run sample backtest with generated price data.

Creates mock Module 5 snapshots and runs metrics suite.
"""
import json
import random
from decimal import Decimal
from typing import Dict, List, Any

# Add parent to path for imports
import sys
sys.path.insert(0, "/home/claude/biotech_screener")

from backtest.returns_provider import (
    CSVReturnsProvider,
    ShuffledReturnsProvider,
    LaggedReturnsProvider,
)
from backtest.metrics import (
    run_metrics_suite,
    generate_attribution_frame,
    HORIZON_DISPLAY_NAMES,
)

# Seed for reproducibility
random.seed(123)

# Tickers from our price data
TICKERS = [
    "AMGN", "GILD", "VRTX", "REGN", "BIIB",
    "ALNY", "BMRN", "SGEN", "INCY", "EXEL",
    "MRNA", "BNTX", "IONS", "SRPT", "RARE",
    "BLUE", "FOLD", "ACAD", "HALO", "KRTX",
    "IMVT", "ARWR", "PCVX", "BEAM", "EDIT",
]

# Stage buckets (mock assignment)
STAGE_MAP = {
    "AMGN": "late", "GILD": "late", "VRTX": "late", "REGN": "late", "BIIB": "late",
    "ALNY": "late", "BMRN": "mid", "SGEN": "late", "INCY": "mid", "EXEL": "mid",
    "MRNA": "late", "BNTX": "late", "IONS": "mid", "SRPT": "mid", "RARE": "early",
    "BLUE": "early", "FOLD": "mid", "ACAD": "mid", "HALO": "mid", "KRTX": "early",
    "IMVT": "early", "ARWR": "mid", "PCVX": "early", "BEAM": "early", "EDIT": "early",
}

# Market cap buckets (mock)
MCAP_MAP = {
    "AMGN": "large", "GILD": "large", "VRTX": "large", "REGN": "large", "BIIB": "large",
    "ALNY": "mid", "BMRN": "mid", "SGEN": "mid", "INCY": "mid", "EXEL": "small",
    "MRNA": "large", "BNTX": "large", "IONS": "small", "SRPT": "mid", "RARE": "small",
    "BLUE": "micro", "FOLD": "small", "ACAD": "small", "HALO": "small", "KRTX": "small",
    "IMVT": "small", "ARWR": "small", "PCVX": "small", "BEAM": "small", "EDIT": "micro",
}


def create_mock_snapshot(as_of_date: str, score_seed: int = None) -> Dict[str, Any]:
    """
    Create a mock Module 5 snapshot.
    
    Scores are semi-random but with some signal:
    - Late stage gets bonus
    - Large cap gets slight bonus
    - Some random noise
    """
    if score_seed is not None:
        random.seed(score_seed)
    
    securities = []
    for ticker in TICKERS:
        # Base score
        base = 50
        
        # Stage bonus
        stage = STAGE_MAP[ticker]
        if stage == "late":
            base += 15
        elif stage == "mid":
            base += 5
        # early gets no bonus
        
        # Market cap slight bonus (stability)
        mcap = MCAP_MAP[ticker]
        if mcap == "large":
            base += 5
        elif mcap == "mid":
            base += 2
        
        # Random noise
        noise = random.gauss(0, 10)
        score = max(10, min(95, base + noise))
        
        securities.append({
            "ticker": ticker,
            "composite_score": f"{score:.2f}",
            "stage_bucket": stage,
            "market_cap_bucket": mcap,
            "severity": "none",
            "uncertainty_penalty": "0",
            "flags": [],
        })
    
    # Sort by score descending, assign ranks
    securities.sort(key=lambda x: float(x["composite_score"]), reverse=True)
    for i, sec in enumerate(securities):
        sec["composite_rank"] = i + 1
    
    return {
        "as_of_date": as_of_date,
        "ranked_securities": securities,
        "excluded_securities": [],
        "diagnostic_counts": {"rankable": len(securities), "excluded": 0},
        "provenance": {"ruleset_version": "1.1.0"},
    }


def run_backtest():
    """Run the backtest and print results."""
    print("=" * 70)
    print("BIOTECH SCREENER BACKTEST")
    print("=" * 70)
    
    # Load price data
    print("\n1. Loading price data...")
    provider = CSVReturnsProvider("/home/claude/biotech_screener/data/daily_prices.csv")
    print(f"   Loaded {len(provider.get_available_tickers())} tickers")
    
    # Create snapshots (quarterly for 2023)
    print("\n2. Creating Module 5 snapshots...")
    dates = [
        "2023-01-02", "2023-04-03", "2023-07-03", "2023-10-02",
        "2024-01-02", "2024-04-01", "2024-07-01",
    ]
    
    snapshots = []
    for i, d in enumerate(dates):
        snap = create_mock_snapshot(d, score_seed=100 + i)
        snapshots.append(snap)
        print(f"   {d}: {len(snap['ranked_securities'])} securities")
    
    # Run main backtest
    print("\n3. Running metrics suite...")
    result = run_metrics_suite(
        snapshots,
        provider,
        run_id="sample_backtest_v1",
        horizons=["63d", "126d", "252d"],
    )
    
    # Print aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)
    
    for h_internal, h_display in HORIZON_DISPLAY_NAMES.items():
        if h_internal in result["aggregate_metrics"]:
            agg = result["aggregate_metrics"][h_internal]
            print(f"\n{h_display} ({h_internal}):")
            print(f"  Dates analyzed:     {agg['n_dates']}")
            print(f"  IC Mean:            {agg['ic_mean']}")
            print(f"  IC Median:          {agg['ic_median']}")
            print(f"  IC Positive Frac:   {agg['ic_pos_frac']}")
            print(f"  Quintile Spread:    {agg['spread_mean']}")
            print(f"  Monotonicity Rate:  {agg['monotonicity_rate']}")
            print(f"  Bucket Spread:      {agg['bucket_spread_mean']}")
            print(f"  Bucket Mono Rate:   {agg['bucket_monotonicity_rate']}")
    
    # Show one period in detail
    print("\n" + "=" * 70)
    print("SAMPLE PERIOD DETAIL: 2023-07-03 @ 63d")
    print("=" * 70)
    
    period = result["period_metrics"]["2023-07-03"]["horizons"]["63d"]
    print(f"  IC Spearman:        {period['ic_spearman']}")
    print(f"  Coverage:           {period['coverage_pct']}")
    print(f"  Bucket Type:        {period['bucket_metrics']['bucket_type']}")
    print(f"  Top-Bottom Spread:  {period['bucket_metrics']['top_minus_bottom']}")
    print(f"  Monotonic:          {period['bucket_metrics']['monotonic']}")
    
    # Attribution frame for debugging
    print("\n  Top 5 by Score (with forward returns):")
    frame = generate_attribution_frame(snapshots[2], provider, "63d")
    for row in frame[:5]:
        ret = row['forward_return'] if row['forward_return'] else "N/A"
        print(f"    {row['ticker']:6s} | Score: {row['composite_score']:>6s} | "
              f"Q{row['quintile']} | Return: {ret}")
    
    # Run null expectation test
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: NULL EXPECTATION TEST")
    print("=" * 70)
    
    shuffled = ShuffledReturnsProvider(provider, seed=42)
    
    # Pre-build shuffled mappings for each snapshot
    from backtest.metrics import compute_forward_windows
    for snap in snapshots:
        windows = compute_forward_windows(snap["as_of_date"], ["63d"])
        w = windows["63d"]
        tickers = [s["ticker"] for s in snap["ranked_securities"]]
        shuffled.prepare_for_tickers(tickers, w["start"], w["end"])
    
    null_result = run_metrics_suite(
        snapshots,
        shuffled,
        run_id="null_test",
        horizons=["63d"],
    )
    
    null_agg = null_result["aggregate_metrics"]["63d"]
    print(f"\n  Shuffled IC Mean:   {null_agg['ic_mean']} (should be ~0)")
    print(f"  Shuffled IC Median: {null_agg['ic_median']}")
    
    original_ic = float(result["aggregate_metrics"]["63d"]["ic_mean"] or 0)
    shuffled_ic = float(null_agg["ic_mean"] or 0)
    
    if abs(shuffled_ic) < 0.05:
        print("  ✓ PASS: Shuffled IC near zero - no leakage detected")
    else:
        print("  ⚠ WARNING: Shuffled IC not near zero - possible leakage")
    
    # Run lag stress test
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: LAG STRESS TEST (+30 days)")
    print("=" * 70)
    
    lagged = LaggedReturnsProvider(provider, lag_days=30)
    lag_result = run_metrics_suite(
        snapshots,
        lagged,
        run_id="lag_test",
        horizons=["63d"],
    )
    
    lag_agg = lag_result["aggregate_metrics"]["63d"]
    print(f"\n  Lagged IC Mean:     {lag_agg['ic_mean']}")
    print(f"  Original IC Mean:   {result['aggregate_metrics']['63d']['ic_mean']}")
    
    lagged_ic = float(lag_agg["ic_mean"] or 0)
    if lagged_ic < original_ic:
        print("  ✓ PASS: Lagged IC degraded - PIT enforcement working")
    else:
        print("  ⚠ NOTE: Lagged IC did not degrade (may be noise in synthetic data)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Snapshots:          {len(snapshots)}")
    print(f"  Securities/snap:    {len(TICKERS)}")
    print(f"  Horizons:           {result['horizons_display']}")
    print(f"  Metrics Version:    {result['provenance']['metrics_version']}")
    print(f"  Config Hash:        {result['provenance']['config_hash'][:20]}...")
    
    return result


if __name__ == "__main__":
    run_backtest()
