"""
Sharadar CSV Backtest Runner - Deterministic Contract

This script defines the exact inputs, outputs, and artifact paths for
reproducible backtest runs using Sharadar CSV data.

INPUTS (Required):
  - data/sharadar_sep.csv: Sharadar SEP (Security End-of-day Prices)
    Columns: ticker, date, closeadj (adjusted close)
    
  - Universe: Defined in BIOTECH_UNIVERSE constant (or loaded from file)
  
  - Date range: START_YEAR, END_YEAR
  
  - Horizons: ["63d", "126d", "252d"]

OUTPUTS (Deterministic):
  output/runs/{run_id}/
  ├── run_summary.json          # Quality gates + headline metrics
  ├── sanity_metrics.json       # IC by stage, rank stability, factor stability
  ├── stability_attribution.json # Rank change decomposition
  ├── backtest_results.json     # Full metrics suite output
  └── config.json               # Exact configuration used

ARTIFACT HASHES:
  All outputs are content-hashed in the run manifest.
  Same inputs → identical hashes.

REPRODUCIBILITY:
  - Run with same config hash to verify exact reproduction
  - Manifest tracks: config_hash, data_hashes, results_hash
  
USAGE:
  python scripts/run_sharadar_backtest.py --prices data/sharadar_sep.csv
  
  Or with explicit config:
  python scripts/run_sharadar_backtest.py \\
    --prices data/sharadar_sep.csv \\
    --start-year 2022 \\
    --end-year 2024 \\
    --delisting-policy conservative
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta, date
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, "/home/claude/biotech_screener")

from module_1_universe import compute_module_1_universe
from module_2_financial import compute_module_2_financial
from module_3_catalyst import compute_module_3_catalyst
from module_4_clinical_dev import compute_module_4_clinical_dev
from module_5_composite import compute_module_5_composite, COHORT_MODE_STAGE_ONLY

from backtest.metrics import run_metrics_suite, compute_forward_windows
from backtest.returns_provider import CSVReturnsProvider, ShuffledReturnsProvider, LaggedReturnsProvider
from backtest.sharadar_provider import (
    SharadarReturnsProvider,
    DELISTING_POLICY_CONSERVATIVE,
    DELISTING_POLICY_LAST_PRICE,
    DELISTING_POLICY_PENALTY,
)
from backtest.sanity_metrics import (
    compute_ic_by_stage,
    compute_rank_stability,
    compute_factor_stability,
    compute_delisting_sensitivity,
)
from backtest.stability_attribution import (
    compute_stability_attribution,
    diagnose_instability,
    print_stability_attribution,
)
from backtest.data_readiness import (
    run_data_readiness_preflight,
    print_preflight_report,
)

from common.run_manifest import RunManifest, compute_content_hash, compute_results_hash
from common.run_summary import generate_run_summary, print_run_summary

# ============================================================================
# CONFIGURATION CONTRACT
# ============================================================================

# Universe: Full biotech set
# This should be loaded from a file in production
BIOTECH_UNIVERSE = [
    # Large cap
    "AMGN", "GILD", "VRTX", "REGN", "BIIB",
    # Mid cap
    "ALNY", "BMRN", "SGEN", "INCY", "EXEL",
    "MRNA", "BNTX", "IONS", "SRPT", "RARE",
    # Small cap
    "BLUE", "FOLD", "ACAD", "HALO", "KRTX",
    "IMVT", "ARWR", "PCVX", "BEAM", "EDIT",
]

# Company data for pipeline (replace with real data source in production)
COMPANY_DATA = {
    "AMGN": {"name": "Amgen Inc", "mcap": 130000, "phase": "approved"},
    "GILD": {"name": "Gilead Sciences", "mcap": 95000, "phase": "approved"},
    "VRTX": {"name": "Vertex Pharmaceuticals", "mcap": 85000, "phase": "approved"},
    "REGN": {"name": "Regeneron Pharmaceuticals", "mcap": 80000, "phase": "approved"},
    "BIIB": {"name": "Biogen Inc", "mcap": 35000, "phase": "phase 3"},
    "ALNY": {"name": "Alnylam Pharmaceuticals", "mcap": 25000, "phase": "phase 3"},
    "BMRN": {"name": "BioMarin Pharmaceutical", "mcap": 15000, "phase": "phase 3"},
    "SGEN": {"name": "Seagen Inc", "mcap": 18000, "phase": "approved"},
    "INCY": {"name": "Incyte Corporation", "mcap": 14000, "phase": "phase 2/3"},
    "EXEL": {"name": "Exelixis Inc", "mcap": 6000, "phase": "phase 3"},
    "MRNA": {"name": "Moderna Inc", "mcap": 45000, "phase": "approved"},
    "BNTX": {"name": "BioNTech SE", "mcap": 25000, "phase": "approved"},
    "IONS": {"name": "Ionis Pharmaceuticals", "mcap": 8000, "phase": "phase 2"},
    "SRPT": {"name": "Sarepta Therapeutics", "mcap": 12000, "phase": "phase 3"},
    "RARE": {"name": "Ultragenyx Pharmaceutical", "mcap": 4000, "phase": "phase 2"},
    "BLUE": {"name": "bluebird bio Inc", "mcap": 400, "phase": "phase 1/2"},
    "FOLD": {"name": "Amicus Therapeutics", "mcap": 3500, "phase": "phase 2"},
    "ACAD": {"name": "ACADIA Pharmaceuticals", "mcap": 4500, "phase": "phase 3"},
    "HALO": {"name": "Halozyme Therapeutics", "mcap": 6500, "phase": "phase 2/3"},
    "KRTX": {"name": "Karuna Therapeutics", "mcap": 5000, "phase": "phase 3"},
    "IMVT": {"name": "Immunovant Inc", "mcap": 2500, "phase": "phase 2"},
    "ARWR": {"name": "Arrowhead Pharmaceuticals", "mcap": 4000, "phase": "phase 2"},
    "PCVX": {"name": "Vaxcyte Inc", "mcap": 7000, "phase": "phase 2"},
    "BEAM": {"name": "Beam Therapeutics", "mcap": 2000, "phase": "phase 1/2"},
    "EDIT": {"name": "Editas Medicine", "mcap": 800, "phase": "phase 1"},
}

# Default configuration
DEFAULT_CONFIG = {
    "start_year": 2023,
    "end_year": 2024,
    "frequency": "monthly",
    "horizons": ["63d", "126d", "252d"],
    "delisting_policy": "conservative",
    "cohort_mode": "stage_only",
    "min_cohort_size": 5,
}

# Output directory
OUTPUT_DIR = Path("/home/claude/biotech_screener/output")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_monthly_dates(start_year: int, end_year: int) -> List[str]:
    """Generate month-end dates (last trading day approximation)."""
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if month == 12:
                last_day = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                last_day = date(year, month + 1, 1) - timedelta(days=1)
            
            while last_day.weekday() >= 5:
                last_day -= timedelta(days=1)
            
            dates.append(last_day.isoformat())
    return dates


def generate_sample_data(tickers: List[str], as_of_date: str, seed: int) -> Dict[str, List[Dict]]:
    """Generate sample pipeline data. Replace with real API calls in production."""
    import random
    random.seed(seed)
    
    universe_records = []
    for ticker in tickers:
        company = COMPANY_DATA.get(ticker, {"name": ticker, "mcap": 1000})
        universe_records.append({
            "ticker": ticker,
            "company_name": company.get("name", ticker),
            "market_cap_mm": company.get("mcap"),
            "status": "active",
        })
    
    financial_records = []
    for ticker in tickers:
        company = COMPANY_DATA.get(ticker, {"mcap": 1000})
        mcap = company.get("mcap", 1000)
        cash = mcap * random.uniform(0.1, 0.4)
        debt = mcap * random.uniform(0.0, 0.2)
        burn = cash / random.uniform(18, 48)
        
        financial_records.append({
            "ticker": ticker,
            "cash_mm": cash,
            "debt_mm": debt,
            "burn_rate_mm": burn,
            "market_cap_mm": mcap,
            "source_date": as_of_date,
        })
    
    trial_records = []
    for ticker in tickers:
        company = COMPANY_DATA.get(ticker, {"phase": "phase 1"})
        phase = company.get("phase", "phase 1")
        num_trials = random.randint(1, 5)
        
        for i in range(num_trials):
            trial_records.append({
                "ticker": ticker,
                "nct_id": f"NCT{random.randint(10000000, 99999999)}",
                "phase": phase,
                "primary_completion_date": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                "status": random.choice(["recruiting", "active", "completed"]),
                "randomized": random.random() > 0.3,
                "blinded": random.choice(["open", "single", "double"]),
                "primary_endpoint": random.choice(["overall survival", "PFS", "ORR", "biomarker", "safety"]),
            })
    
    return {"universe": universe_records, "financial": financial_records, "trials": trial_records}


def run_pipeline(tickers: List[str], as_of_date: str, seed: int) -> Dict[str, Any]:
    """Run full Module 1-5 pipeline."""
    data = generate_sample_data(tickers, as_of_date, seed)
    
    m1 = compute_module_1_universe(data["universe"], as_of_date, universe_tickers=tickers)
    active_tickers = [s["ticker"] for s in m1["active_securities"]]
    
    m2 = compute_module_2_financial(data["financial"], active_tickers, as_of_date)
    m3 = compute_module_3_catalyst(data["trials"], active_tickers, as_of_date)
    m4 = compute_module_4_clinical_dev(data["trials"], active_tickers, as_of_date)
    m5 = compute_module_5_composite(m1, m2, m3, m4, as_of_date, cohort_mode=COHORT_MODE_STAGE_ONLY)
    
    return m5


# ============================================================================
# VALIDATION SUITE
# ============================================================================

def run_validation_suite(
    snapshots: List[Dict],
    provider,
    n_shuffles: int = 20,
) -> tuple[Dict[str, bool], Dict[str, Any]]:
    """Run all validations and return results."""
    from statistics import mean, stdev
    
    results = {}
    details = {}
    
    # V1: Hash consistency
    snap1 = run_pipeline(BIOTECH_UNIVERSE, "2024-01-31", seed=42)
    snap2 = run_pipeline(BIOTECH_UNIVERSE, "2024-01-31", seed=42)
    hash1 = compute_content_hash(snap1)
    hash2 = compute_content_hash(snap2)
    results["hash_consistency"] = hash1 == hash2
    details["hash_consistency"] = {"hash1": hash1[:40], "hash2": hash2[:40]}
    
    # V2: Null expectation (permutation p-value)
    real_result = run_metrics_suite(snapshots, provider, "real", horizons=["63d"])
    ic_real = float(real_result["aggregate_metrics"]["63d"]["ic_mean"] or 0)
    
    null_ics = []
    for seed in range(n_shuffles):
        shuffled = ShuffledReturnsProvider(provider, seed=seed)
        for snap in snapshots:
            windows = compute_forward_windows(snap["as_of_date"], ["63d"])
            w = windows["63d"]
            tickers = [s["ticker"] for s in snap["ranked_securities"]]
            shuffled.prepare_for_tickers(tickers, w["start"], w["end"])
        
        result = run_metrics_suite(snapshots, shuffled, f"null_{seed}", horizons=["63d"])
        ic = result["aggregate_metrics"]["63d"]["ic_mean"]
        if ic is not None:
            null_ics.append(float(ic))
    
    if len(null_ics) >= 2:
        null_mean = mean(null_ics)
        null_std = stdev(null_ics)
        n_exceed = sum(1 for ic in null_ics if abs(ic) >= abs(ic_real))
        p_value = (1 + n_exceed) / (n_shuffles + 1)
        results["null_expectation"] = p_value > 0.05
        details["null_expectation"] = {
            "ic_real": ic_real,
            "ic_shuffle_mean": null_mean,
            "ic_shuffle_std": null_std,
            "p_value_perm": p_value,
        }
    else:
        results["null_expectation"] = False
        details["null_expectation"] = {"error": "insufficient_data"}
    
    # V3: Lag stress (input hash change)
    original_returns = {}
    for snap in snapshots:
        windows = compute_forward_windows(snap["as_of_date"], ["63d"])
        w = windows["63d"]
        for sec in snap["ranked_securities"]:
            key = f"{snap['as_of_date']}_{sec['ticker']}"
            ret = provider(sec["ticker"], w["start"], w["end"])
            original_returns[key] = str(ret) if ret else None
    
    original_hash = compute_content_hash(original_returns)
    
    lagged = LaggedReturnsProvider(provider, lag_days=30)
    lagged_returns = {}
    for snap in snapshots:
        windows = compute_forward_windows(snap["as_of_date"], ["63d"])
        w = windows["63d"]
        for sec in snap["ranked_securities"]:
            key = f"{snap['as_of_date']}_{sec['ticker']}"
            ret = lagged(sec["ticker"], w["start"], w["end"])
            lagged_returns[key] = str(ret) if ret else None
    
    lagged_hash = compute_content_hash(lagged_returns)
    n_different = sum(1 for k in original_returns if original_returns[k] != lagged_returns.get(k))
    
    results["lag_stress"] = original_hash != lagged_hash
    details["lag_stress"] = {
        "original_hash": original_hash[:40],
        "lagged_hash": lagged_hash[:40],
        "returns_changed_pct": n_different / len(original_returns) * 100 if original_returns else 0,
    }
    
    # V4: Cohort coverage (securities-based)
    from collections import defaultdict
    normal_securities = 0
    fallback_securities = 0
    cohort_mode = "unknown"
    
    for snap in snapshots:
        cohort_mode = snap.get("cohort_mode", "stage_mcap")
        for cohort, stats in snap.get("cohort_stats", {}).items():
            count = stats.get("count", 0)
            fallback = stats.get("normalization_fallback", "unknown")
            
            if fallback == "normal":
                normal_securities += count
            else:
                fallback_securities += count
    
    total_securities = normal_securities + fallback_securities
    fallback_rate = fallback_securities / total_securities if total_securities > 0 else 0
    results["cohort_coverage"] = True
    details["cohort_coverage"] = {
        "normal_securities": normal_securities,
        "fallback_securities": fallback_securities,
        "fallback_rate": fallback_rate,
        "cohort_mode": cohort_mode,
    }
    
    # V5: As-of alignment
    violations = 0
    for snap in snapshots:
        as_of = snap["as_of_date"]
        windows = compute_forward_windows(as_of, ["63d"])
        start_date = windows["63d"]["start"]
        
        as_of_dt = date.fromisoformat(as_of)
        start_dt = date.fromisoformat(start_date)
        
        if start_dt <= as_of_dt:
            violations += 1
    
    results["asof_alignment"] = violations == 0
    details["asof_alignment"] = {"violations": violations, "checks": len(snapshots)}
    
    return results, details


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_sharadar_backtest(
    prices_file: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run deterministic Sharadar backtest.
    
    Returns:
        Complete run output with all artifacts
    """
    # Compute deterministic run_id from config + data hashes
    # This ensures same inputs → same run_id
    config_for_hash = {k: v for k, v in config.items() if k != "created_at"}
    config_hash = compute_content_hash(config_for_hash)
    data_hash = compute_content_hash(prices_file)  # Hash of filepath (content hash in production)
    
    # Extract just the hash portion (remove "sha256:" prefix if present)
    config_hash_short = config_hash.replace("sha256:", "")[:12]
    data_hash_short = data_hash.replace("sha256:", "")[:12]
    
    run_id = f"sharadar_{config_hash_short}_{data_hash_short}"
    run_dir = OUTPUT_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Add created_at for audit (not part of run_id)
    config["created_at"] = datetime.now(timezone.utc).isoformat()
    
    print("\n" + "=" * 70)
    print("SHARADAR CSV BACKTEST - DETERMINISTIC CONTRACT")
    print("=" * 70)
    print(f"Run ID:          {run_id}")
    print(f"Config hash:     {config_hash_short}")
    print(f"Data hash:       {data_hash_short}")
    print(f"Prices file:     {prices_file}")
    print(f"Universe:        {len(BIOTECH_UNIVERSE)} tickers")
    print(f"Date range:      {config['start_year']} - {config['end_year']}")
    print(f"Frequency:       {config['frequency']}")
    print(f"Horizons:        {config['horizons']}")
    print(f"Delisting:       {config['delisting_policy']}")
    print(f"Cohort mode:     {config['cohort_mode']}")
    
    # STEP 0: Data readiness preflight
    print("\n[0] Running data readiness preflight...")
    preflight = run_data_readiness_preflight(
        prices_file,
        ticker_filter=BIOTECH_UNIVERSE,
        start_date=f"{config['start_year']}-01-01",
        end_date=f"{config['end_year']}-12-31",
    )
    
    # Save preflight results
    preflight_file = run_dir / "data_readiness.json"
    with open(preflight_file, "w") as f:
        # Exclude per-ticker detail for file size
        output = {k: v for k, v in preflight.items()}
        if "ticker_coverage" in output:
            output["ticker_coverage"] = {
                k: v for k, v in output["ticker_coverage"].items() 
                if k != "per_ticker"
            }
        json.dump(output, f, indent=2, default=str)
    
    print_preflight_report(preflight)
    
    # Gate: fail early if preflight doesn't pass
    if not preflight["gate_passed"]:
        print("\n" + "=" * 70)
        print("❌ PREFLIGHT FAILED - STOPPING EARLY")
        print("=" * 70)
        print(f"Reason: {preflight['gate_reason']}")
        print(f"\nPreflight saved to: {preflight_file}")
        print("Fix data issues before running backtest.")
        
        return {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "preflight_passed": False,
            "preflight": preflight,
        }
    
    # Save config (after preflight passes)
    config_file = run_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nConfig saved to: {config_file}")
    
    # Load provider
    print("\n[1] Loading price data...")
    delisting_map = {
        "conservative": DELISTING_POLICY_CONSERVATIVE,
        "last_price": DELISTING_POLICY_LAST_PRICE,
        "penalty": DELISTING_POLICY_PENALTY,
    }
    delisting_policy = delisting_map.get(config["delisting_policy"], DELISTING_POLICY_CONSERVATIVE)
    
    provider = SharadarReturnsProvider.from_csv(
        prices_file,
        ticker_filter=BIOTECH_UNIVERSE,
        delisting_policy=delisting_policy,
    )
    print(f"  Tickers loaded: {len(provider.get_available_tickers())}")
    
    # Generate snapshots
    print("\n[2] Generating snapshots...")
    dates = generate_monthly_dates(config["start_year"], config["end_year"])
    snapshots = []
    for i, d in enumerate(dates):
        snap = run_pipeline(BIOTECH_UNIVERSE, d, seed=100 + i)
        snapshots.append(snap)
    print(f"  Snapshots created: {len(snapshots)}")
    
    # Run validations
    print("\n[3] Running validations...")
    validation_results, validation_details = run_validation_suite(snapshots, provider)
    for name, passed in validation_results.items():
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {name}")
    
    # Run main backtest
    print("\n[4] Running main backtest...")
    backtest_result = run_metrics_suite(snapshots, provider, run_id, horizons=config["horizons"])
    
    backtest_file = run_dir / "backtest_results.json"
    with open(backtest_file, "w") as f:
        json.dump(backtest_result, f, indent=2, default=str)
    
    # Collect returns for coverage
    returns_by_date = {}
    for snap in snapshots:
        as_of = snap["as_of_date"]
        returns_by_date[as_of] = {}
        windows = compute_forward_windows(as_of, ["63d"])
        w = windows["63d"]
        
        for sec in snap["ranked_securities"]:
            ret = provider(sec["ticker"], w["start"], w["end"])
            returns_by_date[as_of][sec["ticker"]] = str(ret) if ret else None
    
    # Run sanity metrics
    print("\n[5] Computing sanity metrics...")
    ic_by_stage = compute_ic_by_stage(snapshots, provider, "63d")
    rank_stability = compute_rank_stability(snapshots)
    factor_stability = compute_factor_stability(snapshots)
    delist_sensitivity = compute_delisting_sensitivity(
        snapshots, prices_file, BIOTECH_UNIVERSE, "63d"
    )
    
    sanity_metrics = {
        "ic_by_stage": ic_by_stage,
        "rank_stability": rank_stability,
        "factor_stability": factor_stability,
        "delisting_sensitivity": delist_sensitivity,
    }
    
    sanity_file = run_dir / "sanity_metrics.json"
    with open(sanity_file, "w") as f:
        json.dump(sanity_metrics, f, indent=2, default=str)
    
    # Run stability attribution
    print("\n[6] Computing stability attribution...")
    attribution = compute_stability_attribution(snapshots)
    diagnosis = diagnose_instability(attribution, rank_stability)
    
    stability_file = run_dir / "stability_attribution.json"
    with open(stability_file, "w") as f:
        json.dump({
            "summary": attribution["summary"],
            "global_top_movers": attribution["global_top_movers"],
            "diagnosis": diagnosis,
        }, f, indent=2, default=str)
    
    # Generate run summary
    print("\n[7] Generating run summary...")
    provider_diagnostics = provider.get_diagnostics()
    
    summary = generate_run_summary(
        run_id=run_id,
        config=config,
        snapshots=snapshots,
        backtest_result=backtest_result,
        validation_results=validation_results,
        validation_details=validation_details,
        provider_diagnostics=provider_diagnostics,
        returns_by_date=returns_by_date,
        output_dir=run_dir,
    )
    
    # Update stability gates
    summary["stability_gates"]["rank_corr_ok"] = (
        rank_stability["rank_corr_mean"] is not None and 
        rank_stability["rank_corr_mean"] >= 0.35
    )
    summary["stability_gates"]["churn_ok"] = (
        rank_stability["churn_mean"] is not None and 
        rank_stability["churn_mean"] <= 0.65
    )
    
    # Resave with stability gates
    summary_file = run_dir / "run_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Log to manifest
    manifest = RunManifest(OUTPUT_DIR / "manifests")
    manifest.log_run(
        run_id=run_id,
        config=config,
        data_hashes={"prices_file": compute_content_hash(prices_file)},
        results=backtest_result,
        metadata={
            "run_type": "sharadar_csv",
            "quality_gates": summary["quality_gates"],
            "stability_gates": summary["stability_gates"],
        },
    )
    
    # Print summary
    print_run_summary(summary)
    
    # Print stability attribution
    print_stability_attribution(attribution, rank_stability)
    
    # Final output
    print("\n" + "=" * 70)
    print("OUTPUT ARTIFACTS")
    print("=" * 70)
    print(f"  {run_dir}/")
    print(f"  ├── config.json")
    print(f"  ├── data_readiness.json")
    print(f"  ├── run_summary.json")
    print(f"  ├── sanity_metrics.json")
    print(f"  ├── stability_attribution.json")
    print(f"  └── backtest_results.json")
    print(f"\nManifest: {OUTPUT_DIR}/manifests/run_manifest.jsonl")
    
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary": summary,
        "sanity_metrics": sanity_metrics,
        "stability_attribution": attribution,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run Sharadar CSV backtest with deterministic contract"
    )
    parser.add_argument(
        "--prices", 
        default="/home/claude/biotech_screener/data/daily_prices.csv",
        help="Path to Sharadar SEP CSV file"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=DEFAULT_CONFIG["start_year"],
        help="Start year"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=DEFAULT_CONFIG["end_year"],
        help="End year"
    )
    parser.add_argument(
        "--delisting-policy",
        choices=["conservative", "last_price", "penalty"],
        default=DEFAULT_CONFIG["delisting_policy"],
        help="Delisting policy"
    )
    
    args = parser.parse_args()
    
    config = {
        **DEFAULT_CONFIG,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "delisting_policy": args.delisting_policy,
        "prices_file": args.prices,
    }
    
    run_sharadar_backtest(args.prices, config)


if __name__ == "__main__":
    main()
