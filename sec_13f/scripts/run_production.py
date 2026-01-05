"""
Production Pipeline Runner

Fetches real data from ClinicalTrials.gov and runs full pipeline + backtest.
Uses sample financial data (replace with real API when available).
"""
import json
import random
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import sys
sys.path.insert(0, "/home/claude/biotech_screener")

from module_1_universe import compute_module_1_universe
from module_2_financial import compute_module_2_financial
from module_3_catalyst import compute_module_3_catalyst
from module_4_clinical_dev import compute_module_4_clinical_dev
from module_5_composite import compute_module_5_composite

from backtest.returns_provider import CSVReturnsProvider
from backtest.metrics import run_metrics_suite, HORIZON_DISPLAY_NAMES

from data_sources.ctgov_client import (
    ClinicalTrialsClient,
    fetch_trials_for_tickers,
    BIOTECH_TICKER_MAP,
)

# Output directory
OUTPUT_DIR = Path("/home/claude/biotech_screener/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Sample market cap data (replace with real financial API)
MARKET_CAP_DATA = {
    "AMGN": 130000, "GILD": 95000, "VRTX": 85000, "REGN": 80000, "BIIB": 35000,
    "ALNY": 25000, "BMRN": 15000, "SGEN": 18000, "INCY": 14000, "EXEL": 6000,
    "MRNA": 45000, "BNTX": 25000, "IONS": 8000, "SRPT": 12000, "RARE": 4000,
    "BLUE": 400, "FOLD": 3500, "ACAD": 4500, "HALO": 6500, "KRTX": 5000,
    "IMVT": 2500, "ARWR": 4000, "PCVX": 7000, "BEAM": 2000, "EDIT": 800,
}


def generate_financial_records(tickers: List[str], as_of_date: str) -> List[Dict]:
    """
    Generate sample financial records.
    In production, replace with real financial data API (Sharadar, Polygon, etc.)
    """
    random.seed(42)  # Deterministic
    records = []
    
    for ticker in tickers:
        mcap = MARKET_CAP_DATA.get(ticker, 1000)
        
        # Generate realistic-ish financials
        cash = mcap * random.uniform(0.1, 0.4)
        debt = mcap * random.uniform(0.0, 0.2)
        burn = cash / random.uniform(18, 48)
        
        records.append({
            "ticker": ticker,
            "cash_mm": cash,
            "debt_mm": debt,
            "burn_rate_mm": burn,
            "market_cap_mm": mcap,
            "source_date": as_of_date,
        })
    
    return records


def run_production_pipeline(
    as_of_date: str,
    use_cached_trials: bool = True,
) -> Dict[str, Any]:
    """
    Run full pipeline with real ClinicalTrials.gov data.
    
    Args:
        as_of_date: Analysis date (YYYY-MM-DD)
        use_cached_trials: Use cached trial data if available
    
    Returns:
        Module 5 composite output
    """
    print(f"\n{'='*60}")
    print(f"PRODUCTION PIPELINE - {as_of_date}")
    print(f"{'='*60}")
    
    tickers = list(BIOTECH_TICKER_MAP.keys())
    
    # Step 1: Universe
    print("\n1. Building universe...")
    universe_records = [
        {
            "ticker": t,
            "company_name": BIOTECH_TICKER_MAP[t],
            "market_cap_mm": MARKET_CAP_DATA.get(t, 1000),
            "status": "active",
        }
        for t in tickers
    ]
    
    m1 = compute_module_1_universe(universe_records, as_of_date)
    active_tickers = [s["ticker"] for s in m1["active_securities"]]
    print(f"   Active: {len(active_tickers)} tickers")
    
    # Step 2: Financial
    print("\n2. Computing financial scores...")
    financial_records = generate_financial_records(active_tickers, as_of_date)
    m2 = compute_module_2_financial(financial_records, active_tickers, as_of_date)
    print(f"   Scored: {len(m2['scores'])} tickers")
    
    # Step 3: Fetch real trial data
    print("\n3. Fetching trial data from ClinicalTrials.gov...")
    
    cache_file = OUTPUT_DIR / f"trials_cache_{as_of_date}.json"
    
    if use_cached_trials and cache_file.exists():
        print("   Using cached trial data...")
        with open(cache_file) as f:
            trial_records = json.load(f)
    else:
        # Filter to active tickers
        ticker_map = {t: BIOTECH_TICKER_MAP[t] for t in active_tickers if t in BIOTECH_TICKER_MAP}
        trial_records = fetch_trials_for_tickers(ticker_map, as_of_date)
        
        # Cache for future runs
        with open(cache_file, "w") as f:
            json.dump(trial_records, f, indent=2)
    
    print(f"   Total trials: {len(trial_records)}")
    
    # Step 4: Catalyst scoring
    print("\n4. Computing catalyst scores...")
    m3 = compute_module_3_catalyst(trial_records, active_tickers, as_of_date)
    with_catalyst = m3["diagnostic_counts"]["with_catalyst"]
    print(f"   With catalyst: {with_catalyst} tickers")
    
    # Step 5: Clinical development scoring
    print("\n5. Computing clinical development scores...")
    m4 = compute_module_4_clinical_dev(trial_records, active_tickers, as_of_date)
    print(f"   Total trials evaluated: {m4['diagnostic_counts']['total_trials']}")
    
    # Step 6: Composite ranking
    print("\n6. Computing composite scores...")
    m5 = compute_module_5_composite(m1, m2, m3, m4, as_of_date)
    print(f"   Ranked: {m5['diagnostic_counts']['rankable']} tickers")
    
    # Show top 10
    print(f"\n   {'='*50}")
    print(f"   TOP 10 SECURITIES")
    print(f"   {'='*50}")
    print(f"   {'Rank':<6} {'Ticker':<8} {'Score':>8} {'Stage':<8} {'Phase':<12}")
    print(f"   {'-'*50}")
    
    for sec in m5["ranked_securities"][:10]:
        clin = next((s for s in m4["scores"] if s["ticker"] == sec["ticker"]), {})
        phase = clin.get("lead_phase", "unknown")
        print(f"   {sec['composite_rank']:<6} {sec['ticker']:<8} "
              f"{sec['composite_score']:>8} {sec['stage_bucket']:<8} {phase:<12}")
    
    # Save snapshot
    snapshot_file = OUTPUT_DIR / f"snapshot_{as_of_date}.json"
    with open(snapshot_file, "w") as f:
        json.dump(m5, f, indent=2)
    print(f"\n   Saved: {snapshot_file}")
    
    return m5


def run_production_backtest(
    dates: List[str],
    use_cached_trials: bool = True,
) -> Dict[str, Any]:
    """
    Run full backtest across multiple dates with real data.
    """
    print("\n" + "=" * 70)
    print("PRODUCTION BACKTEST")
    print("=" * 70)
    
    # Generate snapshots for each date
    snapshots = []
    for d in dates:
        snap = run_production_pipeline(d, use_cached_trials)
        snapshots.append(snap)
    
    # Run backtest
    print("\n" + "=" * 70)
    print("RUNNING BACKTEST METRICS")
    print("=" * 70)
    
    provider = CSVReturnsProvider("/home/claude/biotech_screener/data/daily_prices.csv")
    
    result = run_metrics_suite(
        snapshots,
        provider,
        run_id="production_v1",
        horizons=["63d", "126d", "252d"],
    )
    
    # Print results
    print(f"\n{'Horizon':<10} {'IC Mean':>12} {'IC Pos%':>10} {'Spread':>12}")
    print("-" * 46)
    
    for h in ["63d", "126d", "252d"]:
        agg = result["aggregate_metrics"][h]
        ic = float(agg["ic_mean"]) if agg["ic_mean"] else 0
        pos = float(agg["ic_pos_frac"]) if agg["ic_pos_frac"] else 0
        spread = float(agg["bucket_spread_mean"]) if agg["bucket_spread_mean"] else 0
        
        display = HORIZON_DISPLAY_NAMES.get(h, h)
        print(f"{display:<10} {ic:>12.4f} {pos*100:>9.0f}% {spread*100:>11.1f}%")
    
    # Save results
    results_file = OUTPUT_DIR / "backtest_results.json"
    with open(results_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {results_file}")
    
    return result


def main():
    """Main entry point."""
    print("\n" + "=" * 70)
    print("BIOTECH SCREENER - PRODUCTION RUN")
    print("=" * 70)
    print("\nThis will fetch REAL trial data from ClinicalTrials.gov")
    print("and run the full pipeline + backtest.")
    print("\nNote: Price data is synthetic (replace with real market data)")
    
    # Analysis dates (quarterly)
    dates = [
        "2023-07-03",
        "2023-10-02",
        "2024-01-02",
        "2024-04-01",
    ]
    
    # Run backtest
    result = run_production_backtest(dates, use_cached_trials=True)
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"\nSnapshots: {len(dates)}")
    print(f"Tickers: {len(BIOTECH_TICKER_MAP)}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
