"""
First Real-Data Backtest Runner

Measurement mode - characterize data + behavior before optimization.

Setup:
- Universe: Configurable biotech tickers (200-300)
- As-of frequency: Monthly (end of month)
- Horizon: 63d primary (126d/252d secondary)
- Delisting policy: Conservative (drop obs, measure coverage)

Outputs:
- run_summary.json (one-glance quality gate)
- Full validation suite
- Coverage/fallback/delisting diagnostics

Decision gate:
- If return_coverage < 80% or fallback > 20% → fix before interpreting IC
"""
import json
import random
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, "/home/claude/biotech_screener")

from module_1_universe import compute_module_1_universe
from module_2_financial import compute_module_2_financial
from module_3_catalyst import compute_module_3_catalyst
from module_4_clinical_dev import compute_module_4_clinical_dev
from module_5_composite import compute_module_5_composite

from backtest.returns_provider import CSVReturnsProvider, ShuffledReturnsProvider, LaggedReturnsProvider
from backtest.sharadar_provider import (
    SharadarReturnsProvider,
    DELISTING_POLICY_CONSERVATIVE,
)
from backtest.metrics import run_metrics_suite, compute_forward_windows

from common.run_manifest import RunManifest, compute_content_hash, create_data_hashes
from common.run_summary import generate_run_summary, print_run_summary

# Output directory
OUTPUT_DIR = Path("/home/claude/biotech_screener/output")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Universe: Full biotech set (expand as needed)
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

# Company data for sample financial/clinical generation
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

# As-of dates: Monthly (last trading day approximation)
def generate_monthly_dates(start_year: int, end_year: int) -> List[str]:
    """Generate month-end dates."""
    dates = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Last day of month
            if month == 12:
                last_day = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                last_day = date(year, month + 1, 1) - timedelta(days=1)
            
            # Adjust for weekends
            while last_day.weekday() >= 5:
                last_day -= timedelta(days=1)
            
            dates.append(last_day.isoformat())
    return dates


# ============================================================================
# DATA GENERATION (Replace with real data sources)
# ============================================================================

def generate_sample_data(tickers: List[str], as_of_date: str, seed: int) -> Dict[str, List[Dict]]:
    """Generate sample data. Replace with real API calls."""
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
    m5 = compute_module_5_composite(m1, m2, m3, m4, as_of_date)
    
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
    
    results = {}
    details = {}
    
    # V1: Hash consistency
    print("\n[V1] Hash Consistency...")
    snap1 = run_pipeline(BIOTECH_UNIVERSE, "2024-01-31", seed=42)
    snap2 = run_pipeline(BIOTECH_UNIVERSE, "2024-01-31", seed=42)
    hash1 = compute_content_hash(snap1)
    hash2 = compute_content_hash(snap2)
    results["hash_consistency"] = hash1 == hash2
    details["hash_consistency"] = {"hash1": hash1[:40], "hash2": hash2[:40]}
    print(f"  {'✓' if results['hash_consistency'] else '✗'} Hash consistency")
    
    # V2: Null expectation (permutation p-value)
    print("[V2] Null Expectation...")
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
    print(f"  {'✓' if results['null_expectation'] else '✗'} Null expectation (p={details['null_expectation'].get('p_value_perm', 0):.3f})")
    
    # V3: Lag stress (input hash change)
    print("[V3] Lag Stress...")
    original_returns = {}
    for snap in snapshots:
        windows = compute_forward_windows(snap["as_of_date"], ["63d"])
        w = windows["63d"]
        for sec in snap["ranked_securities"]:
            key = f"{snap['as_of_date']}_{sec['ticker']}"
            ret = provider.get_forward_total_return(sec["ticker"], w["start"], w["end"])
            original_returns[key] = str(ret) if ret else None
    
    original_hash = compute_content_hash(original_returns)
    
    lagged = LaggedReturnsProvider(provider, lag_days=30)
    lagged_returns = {}
    for snap in snapshots:
        windows = compute_forward_windows(snap["as_of_date"], ["63d"])
        w = windows["63d"]
        for sec in snap["ranked_securities"]:
            key = f"{snap['as_of_date']}_{sec['ticker']}"
            ret = lagged.get_forward_total_return(sec["ticker"], w["start"], w["end"])
            lagged_returns[key] = str(ret) if ret else None
    
    lagged_hash = compute_content_hash(lagged_returns)
    n_different = sum(1 for k in original_returns if original_returns[k] != lagged_returns.get(k))
    
    results["lag_stress"] = original_hash != lagged_hash
    details["lag_stress"] = {
        "original_hash": original_hash[:40],
        "lagged_hash": lagged_hash[:40],
        "returns_changed_pct": n_different / len(original_returns) * 100 if original_returns else 0,
    }
    print(f"  {'✓' if results['lag_stress'] else '✗'} Lag stress (hash differs)")
    
    # V4: Cohort coverage (securities-based)
    print("[V4] Cohort Coverage...")
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
    results["cohort_coverage"] = True  # Always passes (informational)
    details["cohort_coverage"] = {
        "normal_securities": normal_securities,
        "fallback_securities": fallback_securities,
        "fallback_rate": fallback_rate,
        "cohort_mode": cohort_mode,
    }
    print(f"  {'✓' if results['cohort_coverage'] else '✗'} Cohort coverage (mode={cohort_mode}, fallback={fallback_rate*100:.1f}% of securities)")
    
    # V5: As-of alignment
    print("[V5] As-Of Alignment...")
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
    print(f"  {'✓' if results['asof_alignment'] else '✗'} As-of alignment ({violations} violations)")
    
    return results, details


# ============================================================================
# MAIN RUNNER
# ============================================================================

def run_first_real_backtest(
    prices_file: str,
    start_year: int = 2023,
    end_year: int = 2024,
    horizons: List[str] = None,
) -> Dict[str, Any]:
    """
    Run first real-data backtest in measurement mode.
    
    Args:
        prices_file: Path to price CSV (Sharadar format or sample)
        start_year: Start year for monthly dates
        end_year: End year for monthly dates
        horizons: Horizons to test (default: ["63d"])
    
    Returns:
        Run summary dict
    """
    if horizons is None:
        horizons = ["63d"]
    
    run_id = f"real_monthly_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    
    print("\n" + "=" * 70)
    print("FIRST REAL-DATA BACKTEST - MEASUREMENT MODE")
    print("=" * 70)
    print(f"Run ID:          {run_id}")
    print(f"Universe:        {len(BIOTECH_UNIVERSE)} tickers")
    print(f"Date range:      {start_year} - {end_year}")
    print(f"Frequency:       Monthly")
    print(f"Horizons:        {horizons}")
    print(f"Delisting:       Conservative (drop obs)")
    print(f"Prices:          {prices_file}")
    
    # Load returns provider
    print("\n[1] Loading price data...")
    try:
        # Try Sharadar format first
        provider = SharadarReturnsProvider.from_csv(
            prices_file,
            ticker_filter=BIOTECH_UNIVERSE,
            delisting_policy=DELISTING_POLICY_CONSERVATIVE,
        )
        provider_type = "sharadar"
    except:
        # Fall back to simple CSV
        provider = CSVReturnsProvider(prices_file)
        provider_type = "csv"
    
    print(f"  Provider: {provider_type}")
    print(f"  Tickers loaded: {len(provider.get_available_tickers())}")
    
    # Generate monthly as-of dates
    print("\n[2] Generating snapshots...")
    dates = generate_monthly_dates(start_year, end_year)
    print(f"  Monthly dates: {len(dates)}")
    
    snapshots = []
    for i, d in enumerate(dates):
        snap = run_pipeline(BIOTECH_UNIVERSE, d, seed=100 + i)
        snapshots.append(snap)
    print(f"  Snapshots created: {len(snapshots)}")
    
    # Run validations
    print("\n[3] Running validations...")
    validation_results, validation_details = run_validation_suite(
        snapshots, provider, n_shuffles=20
    )
    
    # Run main backtest
    print("\n[4] Running main backtest...")
    backtest_result = run_metrics_suite(
        snapshots, provider, run_id, horizons=horizons
    )
    
    # Collect returns for coverage calculation
    returns_by_date = {}
    for snap in snapshots:
        as_of = snap["as_of_date"]
        returns_by_date[as_of] = {}
        windows = compute_forward_windows(as_of, ["63d"])
        w = windows["63d"]
        
        for sec in snap["ranked_securities"]:
            ret = provider.get_forward_total_return(sec["ticker"], w["start"], w["end"])
            returns_by_date[as_of][sec["ticker"]] = str(ret) if ret else None
    
    # Get provider diagnostics
    provider_diagnostics = {}
    if hasattr(provider, "get_diagnostics"):
        provider_diagnostics = provider.get_diagnostics()
    
    # Generate summary
    print("\n[5] Generating run summary...")
    config = {
        "universe_size": len(BIOTECH_UNIVERSE),
        "date_range": f"{start_year}-{end_year}",
        "frequency": "monthly",
        "horizons": horizons,
        "delisting_policy": "conservative",
        "provider_type": provider_type,
    }
    
    summary = generate_run_summary(
        run_id=run_id,
        config=config,
        snapshots=snapshots,
        backtest_result=backtest_result,
        validation_results=validation_results,
        validation_details=validation_details,
        provider_diagnostics=provider_diagnostics,
        returns_by_date=returns_by_date,
        output_dir=OUTPUT_DIR / "runs" / run_id,
    )
    
    # Log to manifest
    manifest = RunManifest(OUTPUT_DIR / "manifests")
    manifest.log_run(
        run_id=run_id,
        config=config,
        data_hashes=create_data_hashes(prices_file=prices_file, snapshots=snapshots),
        results=backtest_result,
        metadata={
            "run_type": "first_real_backtest",
            "measurement_mode": True,
            "quality_gates": summary["quality_gates"],
        },
    )
    
    # Print summary
    print_run_summary(summary)
    
    return summary


def main():
    """Entry point."""
    # Use sample prices for now (replace with real Sharadar file)
    prices_file = "/home/claude/biotech_screener/data/daily_prices.csv"
    
    summary = run_first_real_backtest(
        prices_file=prices_file,
        start_year=2023,
        end_year=2024,
        horizons=["63d", "126d", "252d"],
    )
    
    print(f"\nSummary saved to: output/runs/{summary['run_id']}/run_summary.json")


if __name__ == "__main__":
    main()
