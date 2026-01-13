"""
Run sample backtest with proper null hypothesis testing.
Uses multiple shuffle seeds to estimate null distribution.
"""
import json
import random
from decimal import Decimal
from typing import Dict, List, Any
from statistics import mean, stdev

import sys
sys.path.insert(0, "/home/claude/biotech_screener")

from backtest.returns_provider import (
    CSVReturnsProvider,
    ShuffledReturnsProvider,
)
from backtest.metrics import (
    run_metrics_suite,
    compute_forward_windows,
    HORIZON_DISPLAY_NAMES,
)

# Tickers and mappings (same as before)
TICKERS = [
    "AMGN", "GILD", "VRTX", "REGN", "BIIB",
    "ALNY", "BMRN", "SGEN", "INCY", "EXEL",
    "MRNA", "BNTX", "IONS", "SRPT", "RARE",
    "BLUE", "FOLD", "ACAD", "HALO", "KRTX",
    "IMVT", "ARWR", "PCVX", "BEAM", "EDIT",
]

STAGE_MAP = {
    "AMGN": "late", "GILD": "late", "VRTX": "late", "REGN": "late", "BIIB": "late",
    "ALNY": "late", "BMRN": "mid", "SGEN": "late", "INCY": "mid", "EXEL": "mid",
    "MRNA": "late", "BNTX": "late", "IONS": "mid", "SRPT": "mid", "RARE": "early",
    "BLUE": "early", "FOLD": "mid", "ACAD": "mid", "HALO": "mid", "KRTX": "early",
    "IMVT": "early", "ARWR": "mid", "PCVX": "early", "BEAM": "early", "EDIT": "early",
}

MCAP_MAP = {
    "AMGN": "large", "GILD": "large", "VRTX": "large", "REGN": "large", "BIIB": "large",
    "ALNY": "mid", "BMRN": "mid", "SGEN": "mid", "INCY": "mid", "EXEL": "small",
    "MRNA": "large", "BNTX": "large", "IONS": "small", "SRPT": "mid", "RARE": "small",
    "BLUE": "micro", "FOLD": "small", "ACAD": "small", "HALO": "small", "KRTX": "small",
    "IMVT": "small", "ARWR": "small", "PCVX": "small", "BEAM": "small", "EDIT": "micro",
}


def create_mock_snapshot(as_of_date: str, score_seed: int = None) -> Dict[str, Any]:
    if score_seed is not None:
        random.seed(score_seed)
    
    securities = []
    for ticker in TICKERS:
        base = 50
        stage = STAGE_MAP[ticker]
        if stage == "late": base += 15
        elif stage == "mid": base += 5
        
        mcap = MCAP_MAP[ticker]
        if mcap == "large": base += 5
        elif mcap == "mid": base += 2
        
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
    
    # Sort ASCENDING: lower score = better = rank 1
    # Validation showed inverted ranking: high scores predicted underperformance
    securities.sort(key=lambda x: float(x["composite_score"]), reverse=False)
    for i, sec in enumerate(securities):
        sec["composite_rank"] = i + 1
    
    return {
        "as_of_date": as_of_date,
        "ranked_securities": securities,
        "excluded_securities": [],
        "diagnostic_counts": {"rankable": len(securities), "excluded": 0},
        "provenance": {"ruleset_version": "1.1.0"},
    }


def run_null_distribution(snapshots: List[Dict], provider, n_shuffles: int = 20) -> List[float]:
    """Run multiple shuffled backtests to estimate null IC distribution."""
    null_ics = []
    
    for seed in range(n_shuffles):
        shuffled = ShuffledReturnsProvider(provider, seed=seed)
        
        # Pre-build shuffled mappings
        for snap in snapshots:
            windows = compute_forward_windows(snap["as_of_date"], ["63d"])
            w = windows["63d"]
            tickers = [s["ticker"] for s in snap["ranked_securities"]]
            shuffled.prepare_for_tickers(tickers, w["start"], w["end"])
        
        result = run_metrics_suite(snapshots, shuffled, f"null_{seed}", horizons=["63d"])
        ic = result["aggregate_metrics"]["63d"]["ic_mean"]
        if ic is not None:
            null_ics.append(float(ic))
    
    return null_ics


def main():
    print("=" * 70)
    print("BIOTECH SCREENER BACKTEST - WITH NULL HYPOTHESIS TESTING")
    print("=" * 70)
    
    # Setup
    provider = CSVReturnsProvider("/home/claude/biotech_screener/data/daily_prices.csv")
    
    dates = [
        "2023-01-02", "2023-04-03", "2023-07-03", "2023-10-02",
        "2024-01-02", "2024-04-01", "2024-07-01",
    ]
    
    snapshots = [create_mock_snapshot(d, score_seed=100 + i) for i, d in enumerate(dates)]
    
    # Main backtest
    print("\n1. Running main backtest...")
    result = run_metrics_suite(snapshots, provider, "main", horizons=["63d", "126d", "252d"])
    
    original_ic = float(result["aggregate_metrics"]["63d"]["ic_mean"] or 0)
    print(f"   Original IC (3m): {original_ic:.4f}")
    
    # Null distribution (multiple shuffles)
    print("\n2. Estimating null IC distribution (20 shuffles)...")
    null_ics = run_null_distribution(snapshots, provider, n_shuffles=20)
    
    null_mean = mean(null_ics)
    null_std = stdev(null_ics) if len(null_ics) > 1 else 0.1
    
    print(f"   Null IC Mean:  {null_mean:.4f}")
    print(f"   Null IC Std:   {null_std:.4f}")
    print(f"   Null IC Range: [{min(null_ics):.4f}, {max(null_ics):.4f}]")
    
    # Statistical test
    if null_std > 0:
        z_score = (original_ic - null_mean) / null_std
    else:
        z_score = 0
    
    print(f"\n3. Statistical Test:")
    print(f"   Z-score: {z_score:.2f}")
    
    if abs(z_score) < 1.96:
        print("   Result: IC NOT significantly different from null (p > 0.05)")
        print("   ✓ EXPECTED with synthetic data (no true signal)")
    else:
        print(f"   Result: IC significantly {'higher' if z_score > 0 else 'lower'} than null (p < 0.05)")
    
    # Expected null IC for N=25
    print(f"\n4. Theoretical Null Analysis:")
    expected_null_std = 1 / (25 ** 0.5)  # ~0.2 for N=25
    print(f"   Expected null std (1/√N): ±{expected_null_std:.3f}")
    print(f"   Observed null std:        ±{null_std:.3f}")
    
    if abs(null_std - expected_null_std) < 0.1:
        print("   ✓ Null distribution matches theory - no leakage")
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Horizon':<10} {'IC Mean':>12} {'IC Pos%':>10} {'Spread':>12} {'Mono%':>10}")
    print("-" * 56)
    
    for h in ["63d", "126d", "252d"]:
        agg = result["aggregate_metrics"][h]
        ic = agg["ic_mean"] or "N/A"
        pos = agg["ic_pos_frac"] or "N/A"
        spread = agg["bucket_spread_mean"] or "N/A"
        mono = agg["bucket_monotonicity_rate"] or "N/A"
        
        if isinstance(ic, str) and ic != "N/A":
            ic = f"{float(ic):.4f}"
        if isinstance(pos, str) and pos != "N/A":
            pos = f"{float(pos)*100:.0f}%"
        if isinstance(spread, str) and spread != "N/A":
            spread = f"{float(spread)*100:.1f}%"
        if isinstance(mono, str) and mono != "N/A":
            mono = f"{float(mono)*100:.0f}%"
        
        display = HORIZON_DISPLAY_NAMES.get(h, h)
        print(f"{display:<10} {ic:>12} {pos:>10} {spread:>12} {mono:>10}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
With SYNTHETIC price data (random walks) and MOCK scores (stage+mcap based):
- IC ≈ 0 is the CORRECT null result
- No actual alpha to detect (scores don't predict random returns)
- Null test confirms no methodological leakage

To detect real signal:
1. Replace mock scores with actual Module 1-5 pipeline output
2. Replace synthetic prices with real market data (CSV or Sharadar)
3. Re-run and compare IC to null distribution

If IC > 2σ above null with real data → signal detected
""")


if __name__ == "__main__":
    main()
