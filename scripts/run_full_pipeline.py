"""
Full Pipeline Integration

Runs Modules 1-5 with sample data, then executes backtest.
Demonstrates end-to-end flow from raw data to IC metrics.
"""
import json
import random
from decimal import Decimal
from typing import Dict, List, Any

import sys
sys.path.insert(0, "/home/claude/biotech_screener")

from module_1_universe import compute_module_1_universe
from module_2_financial import compute_module_2_financial
from module_3_catalyst import compute_module_3_catalyst
from module_4_clinical_dev import compute_module_4_clinical_dev
from module_5_composite import compute_module_5_composite

from backtest.returns_provider import CSVReturnsProvider
from backtest.metrics import run_metrics_suite, HORIZON_DISPLAY_NAMES

# Seed for reproducibility
random.seed(42)

# Sample universe (25 biotech tickers)
TICKERS = [
    "AMGN", "GILD", "VRTX", "REGN", "BIIB",
    "ALNY", "BMRN", "SGEN", "INCY", "EXEL",
    "MRNA", "BNTX", "IONS", "SRPT", "RARE",
    "BLUE", "FOLD", "ACAD", "HALO", "KRTX",
    "IMVT", "ARWR", "PCVX", "BEAM", "EDIT",
]

# Sample company metadata
COMPANY_DATA = {
    "AMGN": {"name": "Amgen Inc", "mcap": 130000},
    "GILD": {"name": "Gilead Sciences", "mcap": 95000},
    "VRTX": {"name": "Vertex Pharmaceuticals", "mcap": 85000},
    "REGN": {"name": "Regeneron Pharmaceuticals", "mcap": 80000},
    "BIIB": {"name": "Biogen Inc", "mcap": 35000},
    "ALNY": {"name": "Alnylam Pharmaceuticals", "mcap": 25000},
    "BMRN": {"name": "BioMarin Pharmaceutical", "mcap": 15000},
    "SGEN": {"name": "Seagen Inc", "mcap": 18000},
    "INCY": {"name": "Incyte Corporation", "mcap": 14000},
    "EXEL": {"name": "Exelixis Inc", "mcap": 6000},
    "MRNA": {"name": "Moderna Inc", "mcap": 45000},
    "BNTX": {"name": "BioNTech SE", "mcap": 25000},
    "IONS": {"name": "Ionis Pharmaceuticals", "mcap": 8000},
    "SRPT": {"name": "Sarepta Therapeutics", "mcap": 12000},
    "RARE": {"name": "Ultragenyx Pharmaceutical", "mcap": 4000},
    "BLUE": {"name": "bluebird bio Inc", "mcap": 400},
    "FOLD": {"name": "Amicus Therapeutics", "mcap": 3500},
    "ACAD": {"name": "ACADIA Pharmaceuticals", "mcap": 4500},
    "HALO": {"name": "Halozyme Therapeutics", "mcap": 6500},
    "KRTX": {"name": "Karuna Therapeutics", "mcap": 5000},
    "IMVT": {"name": "Immunovant Inc", "mcap": 2500},
    "ARWR": {"name": "Arrowhead Pharmaceuticals", "mcap": 4000},
    "PCVX": {"name": "Vaxcyte Inc", "mcap": 7000},
    "BEAM": {"name": "Beam Therapeutics", "mcap": 2000},
    "EDIT": {"name": "Editas Medicine", "mcap": 800},
}

# Sample clinical data
CLINICAL_DATA = {
    "AMGN": {"phase": "approved", "trials": 15, "endpoint": "overall survival"},
    "GILD": {"phase": "approved", "trials": 12, "endpoint": "overall survival"},
    "VRTX": {"phase": "approved", "trials": 10, "endpoint": "complete response"},
    "REGN": {"phase": "approved", "trials": 14, "endpoint": "overall survival"},
    "BIIB": {"phase": "phase 3", "trials": 8, "endpoint": "progression-free survival"},
    "ALNY": {"phase": "phase 3", "trials": 6, "endpoint": "biomarker reduction"},
    "BMRN": {"phase": "phase 3", "trials": 5, "endpoint": "functional improvement"},
    "SGEN": {"phase": "approved", "trials": 7, "endpoint": "objective response rate"},
    "INCY": {"phase": "phase 2/3", "trials": 4, "endpoint": "complete response"},
    "EXEL": {"phase": "phase 3", "trials": 3, "endpoint": "progression-free survival"},
    "MRNA": {"phase": "approved", "trials": 6, "endpoint": "efficacy"},
    "BNTX": {"phase": "approved", "trials": 5, "endpoint": "efficacy"},
    "IONS": {"phase": "phase 2", "trials": 8, "endpoint": "biomarker"},
    "SRPT": {"phase": "phase 3", "trials": 4, "endpoint": "functional improvement"},
    "RARE": {"phase": "phase 2", "trials": 3, "endpoint": "biomarker"},
    "BLUE": {"phase": "phase 1/2", "trials": 2, "endpoint": "safety"},
    "FOLD": {"phase": "phase 2", "trials": 3, "endpoint": "biomarker"},
    "ACAD": {"phase": "phase 3", "trials": 2, "endpoint": "symptom improvement"},
    "HALO": {"phase": "phase 2/3", "trials": 4, "endpoint": "pharmacokinetic"},
    "KRTX": {"phase": "phase 3", "trials": 3, "endpoint": "symptom improvement"},
    "IMVT": {"phase": "phase 2", "trials": 2, "endpoint": "biomarker"},
    "ARWR": {"phase": "phase 2", "trials": 4, "endpoint": "biomarker reduction"},
    "PCVX": {"phase": "phase 2", "trials": 2, "endpoint": "immunogenicity"},
    "BEAM": {"phase": "phase 1/2", "trials": 2, "endpoint": "safety"},
    "EDIT": {"phase": "phase 1", "trials": 1, "endpoint": "safety"},
}


def generate_sample_data(as_of_date: str) -> Dict[str, List[Dict]]:
    """Generate sample data for all modules."""
    
    # Universe records
    universe_records = []
    for ticker in TICKERS:
        company = COMPANY_DATA.get(ticker, {})
        universe_records.append({
            "ticker": ticker,
            "company_name": company.get("name", ticker),
            "market_cap_mm": company.get("mcap"),
            "status": "active",
        })
    
    # Financial records
    financial_records = []
    for ticker in TICKERS:
        company = COMPANY_DATA.get(ticker, {})
        mcap = company.get("mcap", 1000)
        
        # Generate realistic financials based on market cap
        cash = mcap * random.uniform(0.1, 0.4)
        debt = mcap * random.uniform(0.0, 0.2)
        burn = cash / random.uniform(18, 48)  # 18-48 month runway
        
        financial_records.append({
            "ticker": ticker,
            "cash_mm": cash,
            "debt_mm": debt,
            "burn_rate_mm": burn,
            "market_cap_mm": mcap,
            "source_date": as_of_date,
        })
    
    # Trial records
    trial_records = []
    for ticker in TICKERS:
        clinical = CLINICAL_DATA.get(ticker, {})
        num_trials = clinical.get("trials", 1)
        
        for i in range(num_trials):
            # Generate trial with some variation
            days_offset = random.randint(30, 365)
            
            trial_records.append({
                "ticker": ticker,
                "nct_id": f"NCT{random.randint(10000000, 99999999)}",
                "phase": clinical.get("phase", "phase 1"),
                "primary_completion_date": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
                "status": random.choice(["recruiting", "active", "completed"]),
                "randomized": random.random() > 0.3,
                "blinded": random.choice(["open", "single", "double"]),
                "primary_endpoint": clinical.get("endpoint", "safety"),
            })
    
    return {
        "universe": universe_records,
        "financial": financial_records,
        "trials": trial_records,
    }


def run_pipeline(as_of_date: str) -> Dict[str, Any]:
    """Run full Module 1-5 pipeline for a single date."""
    
    # Generate sample data
    data = generate_sample_data(as_of_date)
    
    # Module 1: Universe
    m1 = compute_module_1_universe(
        data["universe"],
        as_of_date,
        universe_tickers=TICKERS,
    )
    active_tickers = [s["ticker"] for s in m1["active_securities"]]
    
    # Module 2: Financial
    m2 = compute_module_2_financial(
        data["financial"],
        active_tickers,
        as_of_date,
    )
    
    # Module 3: Catalyst
    m3 = compute_module_3_catalyst(
        data["trials"],
        active_tickers,
        as_of_date,
    )
    
    # Module 4: Clinical Development
    m4 = compute_module_4_clinical_dev(
        data["trials"],
        active_tickers,
        as_of_date,
    )
    
    # Module 5: Composite
    m5 = compute_module_5_composite(
        m1, m2, m3, m4,
        as_of_date,
    )
    
    return m5


def main():
    print("=" * 70)
    print("BIOTECH SCREENER - FULL PIPELINE INTEGRATION")
    print("=" * 70)
    
    # Generate snapshots for multiple dates
    dates = [
        "2023-01-02", "2023-04-03", "2023-07-03", "2023-10-02",
        "2024-01-02", "2024-04-01", "2024-07-01",
    ]
    
    print("\n1. Running Module 1-5 pipeline for each date...")
    snapshots = []
    for i, d in enumerate(dates):
        random.seed(42 + i)  # Different seed per date for variety
        snap = run_pipeline(d)
        snapshots.append(snap)
        
        top3 = snap["ranked_securities"][:3]
        print(f"   {d}: {snap['diagnostic_counts']['rankable']} ranked | "
              f"Top 3: {', '.join(s['ticker'] for s in top3)}")
    
    # Show sample snapshot detail
    print("\n2. Sample Snapshot Detail (2023-07-03):")
    sample = snapshots[2]
    print(f"   Cohorts: {list(sample['cohort_stats'].keys())}")
    print(f"   Weights: {sample['weights_used']}")
    print(f"\n   Top 5 by Composite Score:")
    print(f"   {'Ticker':<8} {'Score':>8} {'Rank':>6} {'Stage':<8} {'MCap':<8} {'Severity':<8}")
    print("   " + "-" * 50)
    for s in sample["ranked_securities"][:5]:
        print(f"   {s['ticker']:<8} {s['composite_score']:>8} {s['composite_rank']:>6} "
              f"{s['stage_bucket']:<8} {s['market_cap_bucket']:<8} {s['severity']:<8}")
    
    # Run backtest
    print("\n3. Running Backtest Metrics...")
    provider = CSVReturnsProvider("/home/claude/biotech_screener/data/daily_prices.csv")
    
    result = run_metrics_suite(
        snapshots,
        provider,
        run_id="full_pipeline_v1",
        horizons=["63d", "126d", "252d"],
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    
    print(f"\n{'Horizon':<10} {'IC Mean':>12} {'IC Pos%':>10} {'Spread':>12} {'Mono%':>10}")
    print("-" * 56)
    
    for h in ["63d", "126d", "252d"]:
        agg = result["aggregate_metrics"][h]
        ic = float(agg["ic_mean"]) if agg["ic_mean"] else 0
        pos = float(agg["ic_pos_frac"]) if agg["ic_pos_frac"] else 0
        spread = float(agg["bucket_spread_mean"]) if agg["bucket_spread_mean"] else 0
        mono = float(agg["bucket_monotonicity_rate"]) if agg["bucket_monotonicity_rate"] else 0
        
        display = HORIZON_DISPLAY_NAMES.get(h, h)
        print(f"{display:<10} {ic:>12.4f} {pos*100:>9.0f}% {spread*100:>11.1f}% {mono*100:>9.0f}%")
    
    # Sample period detail
    print("\n" + "-" * 56)
    print("Sample Period: 2023-07-03 @ 3m")
    period = result["period_metrics"]["2023-07-03"]["horizons"]["63d"]
    print(f"  IC:         {period['ic_spearman']}")
    print(f"  Coverage:   {period['coverage_pct']}")
    print(f"  Q5-Q1:      {period['q5_minus_q1']}")
    print(f"  Bucket:     {period['bucket_metrics']['bucket_type']}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
This backtest uses:
- REAL pipeline scores (Modules 1-5 with sample biotech data)
- SYNTHETIC price returns (random walks)

Expected Results:
- IC ≈ 0 (no true relationship between scores and random returns)
- This validates the pipeline is working correctly

To Detect Real Signal:
1. Replace synthetic prices with real market data
2. If IC > 2σ above null distribution → signal detected!

Pipeline Flow:
  Module 1 (Universe) → 25 active tickers
  Module 2 (Financial) → Cash runway, debt, size scores
  Module 3 (Catalyst) → Trial proximity scores  
  Module 4 (Clinical) → Phase, design, execution scores
  Module 5 (Composite) → Weighted combination with cohort normalization
  Backtest → IC, quintile spreads, monotonicity
""")
    
    print(f"\nProvenance:")
    print(f"  Metrics Version: {result['provenance']['metrics_version']}")
    print(f"  Ruleset Versions: {result['provenance']['module_ruleset_versions']}")


if __name__ == "__main__":
    main()
