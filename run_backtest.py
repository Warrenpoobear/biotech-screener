#!/usr/bin/env python3
"""
Backtest Runner with Production Scorer Support

Runs point-in-time backtests using either a sample scorer or the production
composite scorer (Module 5).

Usage:
    python run_backtest.py                       # Run with sample scorer
    python run_backtest.py --use-production-scorer  # Run with production scorer
    python run_backtest.py --start-date 2023-01-01 --end-date 2024-12-31
"""
import argparse
import json
import sys
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest_engine import PointInTimeBacktester, create_sample_scoring_function
from backtest.returns_provider import CSVReturnsProvider
from backtest.metrics import run_metrics_suite, HORIZON_DISPLAY_NAMES


# =============================================================================
# DIAGNOSTIC HELPERS (Deterministic)
# =============================================================================

def _bucket_mcap(mcap_mm: Optional[float]) -> str:
    """Classify market cap (in millions) into bucket."""
    if mcap_mm is None:
        return "UNKNOWN"
    if mcap_mm < 500:      # < $0.5B
        return "SMALL"
    if mcap_mm < 2000:     # $0.5B–$2B
        return "MID"
    return "LARGE"         # >= $2B


def _bucket_adv(adv_usd: Optional[float]) -> str:
    """Classify ADV$ into liquidity bucket."""
    if adv_usd is None:
        return "UNKNOWN"
    if adv_usd < 250_000:
        return "ILLIQ"
    if adv_usd < 2_000_000:
        return "MID"
    return "LIQ"


def _top_bottom_sets(sorted_items: List[Tuple[str, float]], frac: float = 0.10) -> Tuple[Set[str], Set[str]]:
    """
    Get top and bottom decile tickers from sorted list.

    sorted_items: list of (ticker, score) sorted high->low deterministically.
    """
    n = len(sorted_items)
    k = max(1, int(n * frac))
    top = {t for t, _ in sorted_items[:k]}
    bot = {t for t, _ in sorted_items[-k:]}
    return top, bot


def _turnover(prev: Optional[Set[str]], cur: Set[str]) -> Optional[float]:
    """
    Compute symmetric turnover between two sets.

    Returns 1 - overlap / avg_size (0 = no change, 1 = complete change)
    """
    if prev is None:
        return None
    if not prev and not cur:
        return 0.0
    inter = len(prev & cur)
    avg_size = (len(prev) + len(cur)) / 2.0
    if avg_size == 0:
        return 0.0
    return 1.0 - (inter / avg_size)


def _compute_bucket_ic(
    scores_by_ticker: Dict[str, float],
    returns_by_ticker: Dict[str, float],
    bucket_by_ticker: Dict[str, str],
    target_bucket: str,
) -> Tuple[Optional[float], int]:
    """Compute IC for a specific bucket. Returns (IC, n_tickers)."""
    common = set(scores_by_ticker.keys()) & set(returns_by_ticker.keys()) & set(bucket_by_ticker.keys())
    filtered = [t for t in common if bucket_by_ticker.get(t) == target_bucket]

    if len(filtered) < 5:
        return None, len(filtered)

    scores = [scores_by_ticker[t] for t in filtered]
    returns = [returns_by_ticker[t] for t in filtered]
    ic = _calculate_spearman_ic(scores, returns)
    return ic, len(filtered)


# Default universe for backtesting (tickers with complete real data + price history)
DEFAULT_UNIVERSE = [
    "ACAD", "BEAM", "BIIB", "BMRN", "EDIT",
    "EXEL", "FOLD", "GILD", "HALO", "IMVT",
    "IONS", "MRNA", "RARE", "REGN", "SRPT",
]

# Sample data for production scorer
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


def load_real_data():
    """Load real production data from files."""
    data_dir = Path(__file__).parent / "production_data"

    # Load financial data
    financial_data = {}
    financial_file = data_dir / "financial_data.json"
    if financial_file.exists():
        with open(financial_file) as f:
            for record in json.load(f):
                ticker = record.get("ticker")
                if ticker:
                    financial_data[ticker] = record

    # Load historical financial snapshots for PIT backtesting
    historical_financials = {}  # ticker -> list of snapshots sorted by date
    hist_file = data_dir / "historical_financial_snapshots.json"
    if hist_file.exists():
        with open(hist_file) as f:
            snapshots = json.load(f)
            for snap in snapshots:
                ticker = snap.get('ticker')
                if ticker:
                    if ticker not in historical_financials:
                        historical_financials[ticker] = []
                    historical_financials[ticker].append(snap)
            # Sort each ticker's snapshots by date
            for ticker in historical_financials:
                historical_financials[ticker].sort(key=lambda x: x.get('date', ''))
        print(f"  Loaded historical financials for {len(historical_financials)} tickers")

    # Load trial records
    trial_data = {}
    trial_file = data_dir / "trial_records.json"
    if trial_file.exists():
        with open(trial_file) as f:
            for record in json.load(f):
                ticker = record.get("ticker")
                if ticker:
                    if ticker not in trial_data:
                        trial_data[ticker] = []
                    trial_data[ticker].append(record)

    # Load universe data for market caps
    universe_data = {}
    universe_file = data_dir / "universe.json"
    if universe_file.exists():
        with open(universe_file) as f:
            for record in json.load(f):
                ticker = record.get("ticker")
                if ticker:
                    universe_data[ticker] = record

    return financial_data, trial_data, universe_data, historical_financials


def get_pit_financial_data(ticker: str, as_of_date: str, historical_financials: dict, fallback_data: dict) -> dict:
    """
    Get point-in-time financial data for a ticker as of a specific date.

    Looks up the most recent financial snapshot that was available before as_of_date.
    """
    snapshots = historical_financials.get(ticker, [])

    if snapshots:
        # Find the most recent snapshot before or on as_of_date
        valid_snapshots = [s for s in snapshots if s.get('date', '') <= as_of_date]
        if valid_snapshots:
            # Return the most recent valid snapshot
            latest = valid_snapshots[-1]
            return {
                'ticker': ticker,
                'Cash': latest.get('cash'),
                'Debt': latest.get('debt'),
                'Assets': latest.get('assets'),
                'Liabilities': latest.get('liabilities'),
                'R&D': latest.get('rd_expense'),
                'source_date': latest.get('date'),
                'pit_lookup': True
            }

    # Fall back to current data if no historical data
    fallback = fallback_data.get(ticker, {})
    return {
        'ticker': ticker,
        'Cash': fallback.get('Cash'),
        'Debt': fallback.get('Debt'),
        'Assets': fallback.get('Assets'),
        'Liabilities': fallback.get('Liabilities'),
        'R&D': fallback.get('R&D'),
        'source_date': as_of_date,
        'pit_lookup': False
    }


def create_production_scorer():
    """
    Create production scorer using Module 5 composite with REAL data.

    This integrates modules 1-5 for full production scoring using actual
    financial and clinical trial data from production_data/.
    """
    from module_1_universe import compute_module_1_universe
    from module_2_financial import compute_module_2_financial
    from module_3_catalyst import compute_module_3_catalyst
    from module_4_clinical_dev import compute_module_4_clinical_dev
    from module_5_composite import compute_module_5_composite

    # Load real data once at scorer creation time
    print("Loading real production data...")
    financial_data, trial_data, universe_data, historical_financials = load_real_data()
    print(f"  Loaded financial data for {len(financial_data)} tickers")
    print(f"  Loaded trial data for {len(trial_data)} tickers")
    print(f"  Loaded universe data for {len(universe_data)} tickers")
    print(f"  Loaded historical financials for {len(historical_financials)} tickers (PIT)")

    # Debug: Show sample historical financials info
    if historical_financials:
        sample_ticker = next(iter(historical_financials.keys()))
        sample_snaps = historical_financials[sample_ticker]
        dates = [s.get('date', 'N/A') for s in sample_snaps]
        print(f"    Sample ({sample_ticker}): {len(sample_snaps)} snapshots, dates: {dates[0]} to {dates[-1]}")
    else:
        print(f"    ⚠️  WARNING: No historical financials - PIT will use static current data!")

    # Print first 20 keys of each data source for debug ticker selection
    print(f"  Data source keys (first 20):")
    print(f"    trial_data: {sorted(trial_data.keys())[:20]}")
    print(f"    historical_financials: {sorted(historical_financials.keys())[:20]}")
    covered_intersection = set(trial_data.keys()) & set(historical_financials.keys())
    print(f"    intersection (trial ∩ hist_fin): {len(covered_intersection)} tickers → {sorted(covered_intersection)[:10]}...")

    def production_scorer(ticker: str, data: Dict, as_of_date: datetime) -> Dict:
        """
        Production scorer using full Module 1-5 pipeline with REAL data.

        Args:
            ticker: Stock ticker
            data: Historical data for the ticker
            as_of_date: Point-in-time date for scoring

        Returns:
            Dict with final_score and components
        """
        as_of_str = as_of_date.strftime("%Y-%m-%d")

        # Get real company/universe data
        uni_record = universe_data.get(ticker, {})
        market_data = uni_record.get("market_data", {})

        # Get market cap from universe data (convert from raw to millions)
        raw_mcap = market_data.get("market_cap")
        if raw_mcap:
            mcap_mm = raw_mcap / 1_000_000  # Convert to millions
        else:
            mcap_mm = COMPANY_DATA.get(ticker, {}).get("mcap", 1000)

        company_name = market_data.get("company_name", ticker)

        # Build universe record
        universe_records = [{
            "ticker": ticker,
            "company_name": company_name,
            "market_cap_mm": mcap_mm,
            "status": "active",
        }]

        # Get point-in-time financial data (uses historical snapshots if available)
        fin_record = get_pit_financial_data(ticker, as_of_str, historical_financials, financial_data)

        # Convert financial data to module format (values in millions)
        cash_raw = fin_record.get("Cash", 0) or 0
        cash_mm = cash_raw / 1_000_000 if cash_raw else mcap_mm * 0.2

        # Estimate debt from liabilities - current liabilities
        liabilities = fin_record.get("Liabilities", 0) or 0
        current_liab = fin_record.get("CurrentLiabilities", 0) or 0
        debt_raw = fin_record.get("Debt", 0) or (liabilities - current_liab if liabilities else 0)
        debt_mm = max(0, debt_raw / 1_000_000) if debt_raw else 0

        # Estimate burn rate from R&D expense
        rd_raw = fin_record.get("R&D", 0) or 0
        burn_mm = rd_raw / 1_000_000 / 4 if rd_raw else cash_mm / 24  # Quarterly burn

        # Use the source_date from PIT lookup for proper filtering
        source_date = fin_record.get("source_date", as_of_str)

        financial_records = [{
            "ticker": ticker,
            "cash_mm": cash_mm,
            "debt_mm": debt_mm,
            "burn_rate_mm": burn_mm,
            "market_cap_mm": mcap_mm,
            "source_date": source_date,
        }]

        # Get real trial data
        ticker_trials = trial_data.get(ticker, [])

        # Convert trial records to module format
        trial_records = []
        for trial in ticker_trials:
            # Map status to expected format
            status_raw = trial.get("status", "UNKNOWN")
            status_map = {
                "RECRUITING": "recruiting",
                "ACTIVE_NOT_RECRUITING": "active",
                "COMPLETED": "completed",
                "TERMINATED": "terminated",
                "WITHDRAWN": "withdrawn",
                "SUSPENDED": "suspended",
                "NOT_YET_RECRUITING": "not_yet_recruiting",
            }
            status = status_map.get(status_raw, status_raw.lower() if status_raw else "unknown")

            # Map phase to expected format
            phase_raw = trial.get("phase", "PHASE1")
            phase_map = {
                "PHASE1": "phase 1",
                "PHASE2": "phase 2",
                "PHASE3": "phase 3",
                "PHASE4": "phase 4",
                "EARLY_PHASE1": "phase 1",
                "NA": "preclinical",
            }
            phase = phase_map.get(phase_raw, phase_raw.lower().replace("phase", "phase ") if phase_raw else "phase 1")

            trial_records.append({
                "ticker": ticker,
                "nct_id": trial.get("nct_id", ""),
                "title": trial.get("title", ""),
                "phase": phase,
                "primary_completion_date": trial.get("primary_completion_date"),
                "completion_date": trial.get("completion_date"),
                "status": status,
                "conditions": trial.get("conditions", []),
                "interventions": trial.get("interventions", []),
                "sponsor": trial.get("sponsor", ""),
                # PIT-relevant date fields (CRITICAL for PIT filtering)
                "last_update_posted": trial.get("last_update_posted"),
                "first_posted": trial.get("first_posted"),
                "source_date": trial.get("source_date"),
            })

        # If no trials found, use fallback
        if not trial_records:
            clinical = CLINICAL_DATA.get(ticker, {"phase": "phase 1", "trials": 1, "endpoint": "safety"})
            trial_records = [{
                "ticker": ticker,
                "nct_id": f"NCT00000000",
                "phase": clinical.get("phase", "phase 1"),
                "primary_completion_date": "2025-12-31",
                "status": "active",
            }]

        # Run Module 1-5 pipeline
        try:
            m1 = compute_module_1_universe(universe_records, as_of_str, universe_tickers=[ticker])
            m2 = compute_module_2_financial(financial_records, [ticker], as_of_str)

            # Module 3 requires file-based API, create fallback for backtest
            # Calculate simple catalyst score based on trial activity
            upcoming_trials = sum(1 for t in trial_records
                                  if (t.get("primary_completion_date") or "9999") > as_of_str
                                  and t.get("status") in ("recruiting", "active"))
            catalyst_score = min(100, 50 + upcoming_trials * 5)  # Simple heuristic
            m3 = {
                "scores": [{
                    "ticker": ticker,
                    "catalyst_normalized": catalyst_score,
                    "catalyst_raw": catalyst_score,
                    "upcoming_catalysts": upcoming_trials,
                }],
                "catalyst_events": [],
            }

            m4 = compute_module_4_clinical_dev(trial_records, [ticker], as_of_str)
            m5 = compute_module_5_composite(m1, m2, m3, m4, as_of_str)

            # Extract final score
            ranked = m5.get("ranked_securities", [])
            if ranked:
                score = float(ranked[0].get("composite_score", 50))
            else:
                score = 50.0

            # Extract component scores with full detail
            fin_score = 0
            fin_severity = None
            if m2.get("scores"):
                m2_score = m2["scores"][0]
                fin_score = float(m2_score.get("financial_normalized", 0) or 0)
                fin_severity = m2_score.get("severity")

            clin_score = 0
            clin_raw = None
            m4_trial_count = 0
            m4_pit_filtered = 0
            m4_lead_phase = None
            if m4.get("scores"):
                m4_score = m4["scores"][0]
                clin_score = float(m4_score.get("clinical_score", 0) or 0)
                clin_raw = clin_score  # clinical_score IS the raw score
                m4_trial_count = m4_score.get("n_trials_unique", 0)
                m4_pit_filtered = m4_score.get("pit_filtered_count_ticker", 0)
                m4_lead_phase = m4_score.get("lead_phase")

            catalyst_score = 0
            catalyst_raw = None
            if m3.get("scores"):
                m3_score = m3["scores"][0]
                catalyst_score = float(m3_score.get("catalyst_normalized", 0) or 0)
                catalyst_raw = m3_score.get("catalyst_raw")

            # Get data state from composite
            data_state = None
            if ranked:
                data_state = ranked[0].get("composite_data_state")

            return {
                "ticker": ticker,
                "final_score": score,
                "components": {
                    "financial": fin_score,
                    "clinical": clin_score,
                    "catalyst": catalyst_score,
                    "trials_count": len(ticker_trials),
                },
                "raw_components": {
                    "clinical_raw": clin_raw,
                    "catalyst_raw": catalyst_raw,
                    "fin_severity": fin_severity,
                    "data_state": data_state,
                },
                # M4 (clinical) debug info
                "m4_debug": {
                    "trial_count": m4_trial_count,
                    "pit_filtered": m4_pit_filtered,
                    "lead_phase": m4_lead_phase,
                },
                "production_pipeline": True,
                "data_source": "real",
                # PIT debug info
                "pit_source_date": source_date,
                "pit_lookup": fin_record.get("pit_lookup", False),
            }
        except Exception as e:
            # Fallback to basic scoring
            return {
                "ticker": ticker,
                "final_score": 50.0,
                "components": {"error": str(e)},
                "production_pipeline": False,
                "data_source": "fallback",
                "pit_lookup": None,
            }

    return production_scorer


def run_direct_backtest(
    start_date: str,
    end_date: str,
    use_production_scorer: bool = False,
    universe: Optional[List[str]] = None,
    price_file: Optional[str] = None,
    output_file: Optional[str] = None,
    frequency_days: int = 30,
    mcap_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run point-in-time backtest with direct data generation.

    This bypasses the need for historical snapshot files by generating
    data on-the-fly for each test date.

    Args:
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        use_production_scorer: Use full Module 1-5 pipeline instead of sample scorer
        universe: List of tickers to backtest
        price_file: Path to historical price CSV
        output_file: Path to save results
        frequency_days: Days between scoring dates
        mcap_filter: Filter by market cap bucket (small/mid/large/mid+large)

    Returns:
        Backtest results dictionary
    """
    from datetime import timedelta
    from statistics import mean, stdev

    price_file = price_file or "data/daily_prices.csv"

    # Check if price file has volume column (needed for proper ADV calculation)
    price_file_has_volume = False
    try:
        import csv
        with open(price_file, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            headers = [h.lower().strip() for h in (reader.fieldnames or [])]
            price_file_has_volume = 'volume' in headers or 'adj_volume' in headers
    except Exception:
        pass  # Will be caught below when loading returns provider

    # Load price data first to get available tickers
    try:
        returns_provider = CSVReturnsProvider(price_file)
        available_tickers = set(returns_provider.get_available_tickers())
        print(f"Loaded price data for {len(available_tickers)} tickers")
        if not price_file_has_volume:
            print(f"  NOTE: Price file missing volume column - ADV$ diagnostics will be disabled")

        # Use all available tickers if no universe specified
        if universe is None:
            universe = sorted(list(available_tickers))
            print(f"Using all {len(universe)} tickers from price data")
        else:
            # Filter provided universe to available tickers
            universe = [t for t in universe if t.upper() in available_tickers]
            print(f"Filtered universe to {len(universe)} tickers with price data")
    except Exception as e:
        print(f"Warning: Could not load price data: {e}")
        returns_provider = None
        available_tickers = set()
        universe = universe or DEFAULT_UNIVERSE

    print()
    print("=" * 70)
    print("BIOTECH SCREENER BACKTEST")
    print("=" * 70)
    print(f"Start Date:    {start_date}")
    print(f"End Date:      {end_date}")
    print(f"Universe:      {len(universe)} tickers")
    print(f"Scorer:        {'Production (Module 1-5)' if use_production_scorer else 'Sample'}")
    print(f"Price File:    {price_file}")
    print(f"Frequency:     Every {frequency_days} days")
    print()

    # Generate test dates
    start_dt = datetime.fromisoformat(start_date)
    end_dt = datetime.fromisoformat(end_date)
    test_dates = []
    current = start_dt
    while current <= end_dt:
        test_dates.append(current)
        current += timedelta(days=frequency_days)

    print(f"Generated {len(test_dates)} test dates")
    if test_dates:
        print(f"  First: {test_dates[0].strftime('%Y-%m-%d')}")
        print(f"  Last:  {test_dates[-1].strftime('%Y-%m-%d')}")
    print()

    # Select scorer
    if use_production_scorer:
        print("Initializing production scorer (Module 1-5 pipeline)...")
        scorer = create_production_scorer()
    else:
        print("Using sample scorer...")
        scorer = create_sample_scoring_function()

    # Run backtest
    print("\nRunning backtest...")
    print("-" * 70)

    all_period_results = []
    ic_30d_values = []
    ic_60d_values = []
    ic_90d_values = []

    # New diagnostic tracking
    spread_90d_values = []  # Top - Bottom decile returns
    turnover_values = []    # Top decile turnover
    prev_top_decile = None  # Previous period's top decile set

    # Bucket IC tracking (use 90d as primary horizon)
    ic_mcap_small, ic_mcap_mid, ic_mcap_large = [], [], []
    ic_mcap_unknown = []  # Track UNKNOWN mcap bucket
    ic_adv_illiq, ic_adv_mid, ic_adv_liq = [], [], []
    ic_adv_unknown = []  # Track UNKNOWN ADV bucket

    # Intersection bucket IC tracking (MID-cap ∩ ADV buckets)
    ic_mid_liq, ic_mid_mid_adv, ic_mid_illiq, ic_mid_unknown_adv = [], [], [], []

    # Load market cap and ADV$ data for bucket classification
    mcap_by_ticker = {}
    adv_by_ticker = {}
    for ticker, data in COMPANY_DATA.items():
        mcap_by_ticker[ticker] = data.get("mcap", 0)
    # Also try to load from universe_data if production scorer
    has_adv_data = False  # Initialize - will be set if sufficient ADV data available
    if use_production_scorer:
        try:
            _, _, universe_data, _ = load_real_data()
            for ticker, record in universe_data.items():
                mkt = record.get("market_data", {})
                mcap = mkt.get("market_cap")
                if mcap:
                    mcap_by_ticker[ticker] = mcap / 1_000_000  # Convert to millions
                # Compute ADV$ from volume_avg_30d and price (correct field names)
                volume = mkt.get("volume_avg_30d")
                price = mkt.get("price")
                if volume and price:
                    adv_by_ticker[ticker] = volume * price  # ADV$ in dollars
            print(f"  Loaded mcap for {len(mcap_by_ticker)} tickers, ADV$ for {len(adv_by_ticker)} tickers")
            # Flag for gating ADV diagnostics - REQUIRE volume column in price file for proper PIT ADV
            # Without volume in price file, ADV from universe.json is static/non-PIT and misleading
            if not price_file_has_volume:
                has_adv_data = False
                print(f"  ADV$ diagnostics DISABLED: price file missing volume column (ADV from universe.json is non-PIT)")
            elif len(adv_by_ticker) >= len(universe) * 0.1:
                has_adv_data = True
            else:
                has_adv_data = False
                print(f"  NOTE: ADV data available for <10% of universe - ADV diagnostics will be skipped")
        except Exception as e:
            print(f"  Warning: Failed to load market data: {e}")

    # Apply market cap filter if specified
    if mcap_filter:
        def passes_mcap_filter(ticker):
            mcap = mcap_by_ticker.get(ticker)
            if mcap is None:
                return False
            bucket = _bucket_mcap(mcap)
            if mcap_filter == "small":
                return bucket == "SMALL"
            elif mcap_filter == "mid":
                return bucket == "MID"
            elif mcap_filter == "large":
                return bucket == "LARGE"
            elif mcap_filter == "mid+large":
                return bucket in ("MID", "LARGE")
            return True

        original_size = len(universe)
        universe = [t for t in universe if passes_mcap_filter(t)]
        print(f"  Applied mcap filter '{mcap_filter}': {original_size} -> {len(universe)} tickers")

    # Pick debug ticker from KNOWN-COVERED intersection (not REGN which may have no data)
    # Load data sources to find covered tickers
    if use_production_scorer:
        try:
            _, trial_data_check, universe_data_check, hist_fin_check = load_real_data()

            # Print first 20 keys of each data source
            print()
            print("DATA SOURCE COVERAGE:")
            print("-" * 50)
            print(f"  trial_data keys (first 20): {sorted(trial_data_check.keys())[:20]}")
            print(f"  historical_financials keys (first 20): {sorted(hist_fin_check.keys())[:20]}")

            # Identify tickers with ADV data
            adv_available_tickers = set()
            for ticker, record in universe_data_check.items():
                mkt = record.get("market_data", {})
                volume = mkt.get("volume_avg_30d")
                price = mkt.get("price")
                if volume and price:
                    adv_available_tickers.add(ticker)
            print(f"  ADV$ available (volume_avg_30d + price): {len(adv_available_tickers)} tickers")

            covered_tickers = set(trial_data_check.keys()) & set(hist_fin_check.keys()) & set(universe)
            covered_with_adv = covered_tickers & adv_available_tickers

            print(f"  trial ∩ hist_fin ∩ universe: {len(covered_tickers)} tickers")
            print(f"  trial ∩ hist_fin ∩ universe ∩ ADV$: {len(covered_with_adv)} tickers")

            # Prefer ticker with ADV data for more complete debugging
            if covered_with_adv:
                debug_ticker = sorted(covered_with_adv)[0]
                print(f"  Debug ticker: {debug_ticker} (from {len(covered_with_adv)} fully-covered + ADV tickers)")
            elif covered_tickers:
                debug_ticker = sorted(covered_tickers)[0]
                print(f"  Debug ticker: {debug_ticker} (from {len(covered_tickers)} covered tickers, no ADV)")
            else:
                debug_ticker = universe[0] if universe else None
                print(f"  Debug ticker: {debug_ticker} (WARNING: no fully-covered tickers found)")
            print()
        except Exception as e:
            debug_ticker = universe[0] if universe else None
            print(f"  Debug ticker: {debug_ticker} (error loading coverage: {e})")
    else:
        debug_ticker = universe[0] if universe else None
    debug_scores = []  # Track score changes for debug_ticker
    all_scores_hashes = []  # Track scores_hash for every period

    # Track ADV diagnostics per period
    all_adv_diagnostics = []

    for i, test_date in enumerate(test_dates):
        print(f"[{i+1}/{len(test_dates)}] Backtesting: {test_date.strftime('%Y-%m-%d')}")

        period_scores = {}
        period_returns_30d = {}
        period_returns_60d = {}
        period_returns_90d = {}
        scored_count = 0

        for ticker in universe:
            # Generate synthetic data for scorer (production scorer generates its own)
            data = {
                "ticker": ticker,
                "company": COMPANY_DATA.get(ticker, {}),
                "clinical": CLINICAL_DATA.get(ticker, {}),
            }

            try:
                score_result = scorer(ticker=ticker, data=data, as_of_date=test_date)
                if score_result and "final_score" in score_result:
                    period_scores[ticker] = float(score_result["final_score"])
                    scored_count += 1

                    # Store PIT debug info for debug ticker
                    if ticker == debug_ticker:
                        if not hasattr(run_direct_backtest, '_debug_pit_info'):
                            run_direct_backtest._debug_pit_info = {}
                        run_direct_backtest._debug_pit_info[ticker] = {
                            'pit_source_date': score_result.get('pit_source_date'),
                            'pit_lookup': score_result.get('pit_lookup'),
                            'raw_components': score_result.get('raw_components', {}),
                            'm4_debug': score_result.get('m4_debug', {}),
                        }

                    # Calculate forward returns if price data available
                    if returns_provider:
                        start_str = (test_date + timedelta(days=1)).strftime("%Y-%m-%d")
                        for horizon, ret_dict in [(30, period_returns_30d),
                                                   (60, period_returns_60d),
                                                   (90, period_returns_90d)]:
                            end_str = (test_date + timedelta(days=horizon+1)).strftime("%Y-%m-%d")
                            ret = returns_provider.get_forward_total_return(ticker, start_str, end_str)
                            if ret is not None:
                                ret_dict[ticker] = float(ret)
            except Exception as e:
                pass  # Skip tickers with errors

        print(f"    Scored: {scored_count}/{len(universe)} ({100*scored_count/len(universe):.1f}%)")

        # Debug: track sample ticker's score to detect static scoring
        if debug_ticker and debug_ticker in period_scores:
            debug_scores.append((test_date.strftime("%Y-%m-%d"), period_scores[debug_ticker]))

        # Probe A: Log PIT details + raw components for debug ticker
        if hasattr(run_direct_backtest, '_debug_pit_info') and debug_ticker:
            pit_info = run_direct_backtest._debug_pit_info.get(debug_ticker, {})
            raw = pit_info.get('raw_components', {})
            m4 = pit_info.get('m4_debug', {})
            if i == 0 or i == len(test_dates) - 1:  # First and last
                print(f"    DEBUG {debug_ticker}: score={period_scores.get(debug_ticker, 0):.2f}")
                print(f"      pit_date={pit_info.get('pit_source_date', 'N/A')} pit_lookup={pit_info.get('pit_lookup', 'N/A')}")
                clin_val = raw.get('clinical_raw')
                clin_str = f"{clin_val:.2f}" if clin_val else 'N/A'
                print(f"      clin_raw={clin_str} cat_raw={raw.get('catalyst_raw')} sev={raw.get('fin_severity')} state={raw.get('data_state')}")
                print(f"      M4: trials={m4.get('trial_count')} pit_filt={m4.get('pit_filtered')} lead={m4.get('lead_phase')}")

        # Calculate IC for this period
        if len(period_scores) >= 5 and len(period_returns_30d) >= 5:
            common_tickers = set(period_scores.keys()) & set(period_returns_30d.keys())
            if len(common_tickers) >= 5:
                scores = [period_scores[t] for t in common_tickers]
                returns = [period_returns_30d[t] for t in common_tickers]
                ic = _calculate_spearman_ic(scores, returns)
                if ic is not None:
                    ic_30d_values.append(ic)

        if len(period_scores) >= 5 and len(period_returns_60d) >= 5:
            common_tickers = set(period_scores.keys()) & set(period_returns_60d.keys())
            if len(common_tickers) >= 5:
                scores = [period_scores[t] for t in common_tickers]
                returns = [period_returns_60d[t] for t in common_tickers]
                ic = _calculate_spearman_ic(scores, returns)
                if ic is not None:
                    ic_60d_values.append(ic)

        if len(period_scores) >= 5 and len(period_returns_90d) >= 5:
            common_tickers = set(period_scores.keys()) & set(period_returns_90d.keys())
            if len(common_tickers) >= 5:
                scores = [period_scores[t] for t in common_tickers]
                returns = [period_returns_90d[t] for t in common_tickers]
                ic = _calculate_spearman_ic(scores, returns)
                if ic is not None:
                    ic_90d_values.append(ic)

                # ============== NEW DIAGNOSTICS (90d horizon) ==============
                # Build sorted list for decile computation (deterministic: by score desc, then ticker asc)
                sorted_items = sorted(
                    [(t, period_scores[t]) for t in common_tickers],
                    key=lambda x: (-x[1], x[0])  # High score first, then alphabetical
                )

                # Top/bottom decile spread
                top_set, bot_set = _top_bottom_sets(sorted_items, frac=0.10)
                top_ret = [period_returns_90d[t] for t in top_set if t in period_returns_90d]
                bot_ret = [period_returns_90d[t] for t in bot_set if t in period_returns_90d]
                if top_ret and bot_ret:
                    top_mean = sum(top_ret) / len(top_ret)
                    bot_mean = sum(bot_ret) / len(bot_ret)
                    spread_90d_values.append(top_mean - bot_mean)

                # Turnover of top decile vs previous period + hash for debugging
                turnover = _turnover(prev_top_decile, top_set)
                if turnover is not None:
                    turnover_values.append(turnover)

                # Probe B: Hash the entire score vector to detect any variation
                import hashlib
                scores_sorted = sorted((t, round(s, 4)) for t, s in period_scores.items())
                blob = "|".join(f"{t}:{s}" for t, s in scores_sorted)
                all_hash = hashlib.sha256(blob.encode()).hexdigest()[:8]
                top_hash = hashlib.sha256(",".join(sorted(top_set)).encode()).hexdigest()[:8]
                all_scores_hashes.append((test_date.strftime("%Y-%m-%d"), all_hash))
                prev_top_decile = top_set

                # Bucket ICs: market cap
                mcap_bucket = {t: _bucket_mcap(mcap_by_ticker.get(t)) for t in common_tickers}
                for bucket, ic_list in [("SMALL", ic_mcap_small), ("MID", ic_mcap_mid), ("LARGE", ic_mcap_large), ("UNKNOWN", ic_mcap_unknown)]:
                    bucket_ic, n = _compute_bucket_ic(period_scores, period_returns_90d, mcap_bucket, bucket)
                    if bucket_ic is not None:
                        ic_list.append(bucket_ic)

                # Bucket ICs: ADV$ - only compute if we have proper volume data
                adv_bucket = {t: _bucket_adv(adv_by_ticker.get(t)) for t in common_tickers}
                if has_adv_data:
                    for bucket, ic_list in [("ILLIQ", ic_adv_illiq), ("MID", ic_adv_mid), ("LIQ", ic_adv_liq), ("UNKNOWN", ic_adv_unknown)]:
                        bucket_ic, n = _compute_bucket_ic(period_scores, period_returns_90d, adv_bucket, bucket)
                        if bucket_ic is not None:
                            ic_list.append(bucket_ic)

                # Print bucket counts for EVERY period (not just first)
                mcap_counts = {}
                for t in common_tickers:
                    mb = mcap_bucket.get(t, "UNKNOWN")
                    mcap_counts[mb] = mcap_counts.get(mb, 0) + 1

                # ADV diagnostics - only track if we have proper volume data
                if has_adv_data:
                    adv_counts = {}
                    adv_non_null_count = 0
                    for t in common_tickers:
                        ab = adv_bucket.get(t, "UNKNOWN")
                        adv_counts[ab] = adv_counts.get(ab, 0) + 1
                        if adv_by_ticker.get(t) is not None:
                            adv_non_null_count += 1

                    # Store ADV diagnostics for this period
                    all_adv_diagnostics.append({
                        'date': test_date.strftime("%Y-%m-%d"),
                        'adv_non_null': adv_non_null_count,
                        'total': len(common_tickers),
                        'ILLIQ': adv_counts.get('ILLIQ', 0),
                        'MID': adv_counts.get('MID', 0),
                        'LIQ': adv_counts.get('LIQ', 0),
                        'UNKNOWN': adv_counts.get('UNKNOWN', 0),
                    })

                    # Compact format: show hash + ADV counts on one line
                    adv_str = f"ILLIQ={adv_counts.get('ILLIQ',0)} MID={adv_counts.get('MID',0)} LIQ={adv_counts.get('LIQ',0)} UNK={adv_counts.get('UNKNOWN',0)}"
                    print(f"    scores_hash={all_hash} | ADV$: non_null={adv_non_null_count}/{len(common_tickers)} {adv_str}")
                else:
                    # Just print scores hash without ADV info
                    print(f"    scores_hash={all_hash}")

                # Intersection ICs: MID-cap ∩ ADV buckets - only compute if we have proper volume data
                if has_adv_data:
                    def compute_intersection_ic(mcap_target, adv_target):
                        filtered = [t for t in common_tickers
                                    if mcap_bucket.get(t) == mcap_target and adv_bucket.get(t) == adv_target]
                        if len(filtered) < 3:  # Lower threshold for intersection
                            return None, len(filtered)
                        scores = [period_scores[t] for t in filtered]
                        returns = [period_returns_90d[t] for t in filtered]
                        ic = _calculate_spearman_ic(scores, returns)
                        return ic, len(filtered)

                    for adv_target, ic_list in [("LIQ", ic_mid_liq), ("MID", ic_mid_mid_adv),
                                                 ("ILLIQ", ic_mid_illiq), ("UNKNOWN", ic_mid_unknown_adv)]:
                        ic_val, n = compute_intersection_ic("MID", adv_target)
                        if ic_val is not None:
                            ic_list.append(ic_val)

        all_period_results.append({
            "date": test_date.isoformat(),
            "scores": period_scores,
            "returns_30d": period_returns_30d,
            "returns_60d": period_returns_60d,
            "returns_90d": period_returns_90d,
        })

    # Compile results
    results = {
        "periods": len(test_dates),
        "start_date": start_date,
        "end_date": end_date,
        "universe_size": len(universe),
        "scorer": "production" if use_production_scorer else "sample",
        "period_results": all_period_results,
    }

    # IC statistics
    if ic_30d_values:
        results["ic_30d"] = {
            "mean": mean(ic_30d_values),
            "std": stdev(ic_30d_values) if len(ic_30d_values) > 1 else 0,
            "positive_pct": 100 * sum(1 for x in ic_30d_values if x > 0) / len(ic_30d_values),
            "count": len(ic_30d_values),
        }
    if ic_60d_values:
        results["ic_60d"] = {
            "mean": mean(ic_60d_values),
            "std": stdev(ic_60d_values) if len(ic_60d_values) > 1 else 0,
            "positive_pct": 100 * sum(1 for x in ic_60d_values if x > 0) / len(ic_60d_values),
            "count": len(ic_60d_values),
        }
    if ic_90d_values:
        results["ic_90d"] = {
            "mean": mean(ic_90d_values),
            "std": stdev(ic_90d_values) if len(ic_90d_values) > 1 else 0,
            "positive_pct": 100 * sum(1 for x in ic_90d_values if x > 0) / len(ic_90d_values),
            "count": len(ic_90d_values),
        }

    # Generate report
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print(f"Periods Analyzed: {results['periods']}")
    print(f"Date Range:       {results['start_date']} to {results['end_date']}")
    print(f"Universe Size:    {results['universe_size']}")
    print()

    print("Information Coefficient (IC) Summary:")
    print("-" * 50)
    for horizon in ["30d", "60d", "90d"]:
        ic_key = f"ic_{horizon}"
        if ic_key in results:
            ic_data = results[ic_key]
            mean_ic = ic_data["mean"]
            pos_pct = ic_data["positive_pct"]
            count = ic_data["count"]
            print(f"  IC {horizon}:  Mean={mean_ic:.4f}, Positive={pos_pct:.1f}%, N={count}")

            # Assessment
            if mean_ic > 0.08:
                assessment = "EXCELLENT"
            elif mean_ic > 0.05:
                assessment = "GOOD"
            elif mean_ic > 0.02:
                assessment = "MODERATE"
            else:
                assessment = "WEAK"
            print(f"          Assessment: {assessment}")
        else:
            print(f"  IC {horizon}:  Insufficient data")
    print()

    # ============== NEW DIAGNOSTIC SUMMARY ==============

    # Debug: Check if sample ticker score changed over time
    if debug_scores:
        scores_only = [s for _, s in debug_scores]
        score_range = max(scores_only) - min(scores_only)
        unique_scores = len(set(round(s, 2) for s in scores_only))
        print(f"DEBUG Score Variation ({debug_ticker}):")
        print("-" * 50)
        print(f"  Score range: {min(scores_only):.2f} - {max(scores_only):.2f} (delta={score_range:.2f})")
        print(f"  Unique scores: {unique_scores}/{len(scores_only)}")
        if score_range < 0.01:
            print(f"  ⚠️  WARNING: Scores are STATIC - inputs not time-varying!")
        print()

    # Summary of ALL scores_hash values (critical for diagnosing static scoring)
    print("Scores Hash Analysis (entire score vector):")
    print("-" * 50)
    if all_scores_hashes:
        unique_hashes = set(h for _, h in all_scores_hashes)
        print(f"  Unique hashes: {len(unique_hashes)}/{len(all_scores_hashes)} periods")
        if len(unique_hashes) == 1:
            print(f"  ⚠️  CRITICAL: Score vector is COMPLETELY STATIC across all periods!")
            print(f"     Hash: {all_scores_hashes[0][1]}")
        elif len(unique_hashes) < len(all_scores_hashes) * 0.5:
            print(f"  ⚠️  WARNING: Score vector has limited variation (< 50% unique)")
        else:
            print(f"  ✓ Score vector varies across periods")

        # Print ALL scores_hash values (user requested all 17)
        print(f"  All {len(all_scores_hashes)} period hashes:")
        for date, hash_val in all_scores_hashes:
            print(f"    {date}: {hash_val}")
    print()

    # ADV Diagnostics Summary (per-period counts) - only show if we have ADV data
    if has_adv_data:
        print("ADV$ Bucket Diagnostics (per period):")
        print("-" * 50)
        if all_adv_diagnostics:
            print(f"  {'Date':<12} {'non_null':>10} {'ILLIQ':>6} {'MID':>6} {'LIQ':>6} {'UNK':>6}")
            print(f"  {'-'*12} {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
            for diag in all_adv_diagnostics:
                print(f"  {diag['date']:<12} {diag['adv_non_null']:>10}/{diag['total']:<4} {diag['ILLIQ']:>6} {diag['MID']:>6} {diag['LIQ']:>6} {diag['UNKNOWN']:>6}")

            # Summary statistics
            avg_non_null = sum(d['adv_non_null'] for d in all_adv_diagnostics) / len(all_adv_diagnostics)
            avg_unknown = sum(d['UNKNOWN'] for d in all_adv_diagnostics) / len(all_adv_diagnostics)
            avg_total = sum(d['total'] for d in all_adv_diagnostics) / len(all_adv_diagnostics)
            pct_unknown = 100 * avg_unknown / avg_total if avg_total > 0 else 0
            print()
            print(f"  Averages: non_null={avg_non_null:.1f}, UNKNOWN={avg_unknown:.1f} ({pct_unknown:.1f}% of tickers)")
            if pct_unknown > 50:
                print(f"  WARNING: UNKNOWN dominates - ADV calculation likely failing")
                print(f"     Common causes: volume as strings, lookback too strict, date misalignment")
        else:
            print("  No ADV diagnostics available")
        print()
    else:
        print("ADV$ Bucket Diagnostics: SKIPPED (insufficient ADV data)")
        print()

    print("Top/Bottom Decile Spread (90d horizon):")
    print("-" * 50)
    if spread_90d_values:
        spread_mean = sum(spread_90d_values) / len(spread_90d_values)
        spread_pos_pct = 100 * sum(1 for x in spread_90d_values if x > 0) / len(spread_90d_values)
        print(f"  Mean Spread:      {spread_mean:+.2%}")
        print(f"  Positive Periods: {spread_pos_pct:.1f}% ({sum(1 for x in spread_90d_values if x > 0)}/{len(spread_90d_values)})")
        results["spread_90d"] = {
            "mean": spread_mean,
            "positive_pct": spread_pos_pct,
            "count": len(spread_90d_values),
            "values": spread_90d_values,
        }
    else:
        print("  Insufficient data")
    print()

    print("Top Decile Turnover:")
    print("-" * 50)
    if turnover_values:
        turnover_mean = sum(turnover_values) / len(turnover_values)
        print(f"  Mean Turnover:    {turnover_mean:.1%}")
        print(f"  Observations:     {len(turnover_values)}")
        results["turnover"] = {
            "mean": turnover_mean,
            "count": len(turnover_values),
            "values": turnover_values,
        }
    else:
        print("  Insufficient data (need 2+ periods)")
    print()

    print("IC by Market Cap Bucket (90d horizon):")
    print("-" * 50)
    for label, ic_list in [("SMALL (<$0.5B)", ic_mcap_small),
                           ("MID ($0.5-2B)", ic_mcap_mid),
                           ("LARGE (>$2B)", ic_mcap_large),
                           ("UNKNOWN", ic_mcap_unknown)]:
        if ic_list:
            bucket_mean = sum(ic_list) / len(ic_list)
            bucket_pos = 100 * sum(1 for x in ic_list if x > 0) / len(ic_list)
            print(f"  {label:16s}: IC={bucket_mean:+.4f}, Pos={bucket_pos:.0f}%, N={len(ic_list)}")
        else:
            print(f"  {label:16s}: Insufficient data")
    results["ic_by_mcap"] = {
        "small": {"mean": sum(ic_mcap_small)/len(ic_mcap_small), "n": len(ic_mcap_small)} if ic_mcap_small else None,
        "mid": {"mean": sum(ic_mcap_mid)/len(ic_mcap_mid), "n": len(ic_mcap_mid)} if ic_mcap_mid else None,
        "large": {"mean": sum(ic_mcap_large)/len(ic_mcap_large), "n": len(ic_mcap_large)} if ic_mcap_large else None,
        "unknown": {"mean": sum(ic_mcap_unknown)/len(ic_mcap_unknown), "n": len(ic_mcap_unknown)} if ic_mcap_unknown else None,
    }
    print()

    # ADV bucket ICs - only show if we have ADV data
    if has_adv_data:
        print("IC by ADV$ Bucket (90d horizon):")
        print("-" * 50)
        for label, ic_list in [("ILLIQ (<$250K)", ic_adv_illiq),
                               ("MID ($250K-2M)", ic_adv_mid),
                               ("LIQ (>$2M)", ic_adv_liq),
                               ("UNKNOWN", ic_adv_unknown)]:
            if ic_list:
                bucket_mean = sum(ic_list) / len(ic_list)
                bucket_pos = 100 * sum(1 for x in ic_list if x > 0) / len(ic_list)
                print(f"  {label:16s}: IC={bucket_mean:+.4f}, Pos={bucket_pos:.0f}%, N={len(ic_list)}")
            else:
                print(f"  {label:16s}: Insufficient data")
        results["ic_by_adv"] = {
            "illiq": {"mean": sum(ic_adv_illiq)/len(ic_adv_illiq), "n": len(ic_adv_illiq)} if ic_adv_illiq else None,
            "mid": {"mean": sum(ic_adv_mid)/len(ic_adv_mid), "n": len(ic_adv_mid)} if ic_adv_mid else None,
            "liq": {"mean": sum(ic_adv_liq)/len(ic_adv_liq), "n": len(ic_adv_liq)} if ic_adv_liq else None,
            "unknown": {"mean": sum(ic_adv_unknown)/len(ic_adv_unknown), "n": len(ic_adv_unknown)} if ic_adv_unknown else None,
        }
        print()

        # Intersection ICs: MID-cap by ADV bucket
        print("IC for MID-cap by ADV$ Bucket (90d horizon):")
        print("-" * 50)
        for label, ic_list in [("MID ∩ LIQ", ic_mid_liq),
                               ("MID ∩ MID-ADV", ic_mid_mid_adv),
                               ("MID ∩ ILLIQ", ic_mid_illiq),
                               ("MID ∩ UNKNOWN", ic_mid_unknown_adv)]:
            if ic_list:
                bucket_mean = sum(ic_list) / len(ic_list)
                bucket_pos = 100 * sum(1 for x in ic_list if x > 0) / len(ic_list)
                print(f"  {label:16s}: IC={bucket_mean:+.4f}, Pos={bucket_pos:.0f}%, N={len(ic_list)}")
            else:
                print(f"  {label:16s}: Insufficient data")
        results["ic_mid_by_adv"] = {
            "mid_liq": {"mean": sum(ic_mid_liq)/len(ic_mid_liq), "n": len(ic_mid_liq)} if ic_mid_liq else None,
            "mid_mid_adv": {"mean": sum(ic_mid_mid_adv)/len(ic_mid_mid_adv), "n": len(ic_mid_mid_adv)} if ic_mid_mid_adv else None,
            "mid_illiq": {"mean": sum(ic_mid_illiq)/len(ic_mid_illiq), "n": len(ic_mid_illiq)} if ic_mid_illiq else None,
            "mid_unknown": {"mean": sum(ic_mid_unknown_adv)/len(ic_mid_unknown_adv), "n": len(ic_mid_unknown_adv)} if ic_mid_unknown_adv else None,
        }
        print()
    else:
        print("IC by ADV$ Bucket: SKIPPED (insufficient ADV data)")
        print("IC for MID-cap by ADV$ Bucket: SKIPPED (insufficient ADV data)")
        print()

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to: {output_file}")

    print("=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    return results


def _calculate_spearman_ic(scores: List[float], returns: List[float]) -> Optional[float]:
    """Calculate Spearman rank correlation (Information Coefficient)."""
    if len(scores) < 5 or len(scores) != len(returns):
        return None

    n = len(scores)

    # Rank the scores and returns
    def rank_data(data):
        sorted_indices = sorted(range(len(data)), key=lambda i: data[i])
        ranks = [0.0] * len(data)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank + 1
        return ranks

    score_ranks = rank_data(scores)
    return_ranks = rank_data(returns)

    # Calculate Spearman correlation
    mean_s = sum(score_ranks) / n
    mean_r = sum(return_ranks) / n

    numerator = sum((score_ranks[i] - mean_s) * (return_ranks[i] - mean_r) for i in range(n))
    denom_s = sum((score_ranks[i] - mean_s) ** 2 for i in range(n)) ** 0.5
    denom_r = sum((return_ranks[i] - mean_r) ** 2 for i in range(n)) ** 0.5

    if denom_s == 0 or denom_r == 0:
        return None

    return numerator / (denom_s * denom_r)


def run_backtest(
    start_date: str,
    end_date: str,
    use_production_scorer: bool = False,
    universe: Optional[List[str]] = None,
    price_file: Optional[str] = None,
    output_file: Optional[str] = None,
    frequency_days: int = 30,
    mcap_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run point-in-time backtest.

    This is a wrapper that uses the direct backtest method for production scorers.
    """
    return run_direct_backtest(
        start_date=start_date,
        end_date=end_date,
        use_production_scorer=use_production_scorer,
        universe=universe,
        price_file=price_file,
        output_file=output_file,
        frequency_days=frequency_days,
        mcap_filter=mcap_filter,
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run point-in-time backtest for biotech screener",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_backtest.py --use-production-scorer
    python run_backtest.py --start-date 2022-01-01 --end-date 2024-12-31
    python run_backtest.py --use-production-scorer --output results/backtest_prod.json
        """
    )

    parser.add_argument(
        "--use-production-scorer",
        action="store_true",
        help="Use production Module 1-5 composite scorer instead of sample scorer"
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
        "--price-file",
        type=str,
        default="data/daily_prices.csv",
        help="Path to historical price CSV"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--frequency",
        type=int,
        default=30,
        help="Days between scoring dates"
    )
    parser.add_argument(
        "--universe",
        type=str,
        nargs="+",
        default=None,
        help="Custom universe of tickers"
    )
    parser.add_argument(
        "--mcap-filter",
        type=str,
        choices=["small", "mid", "large", "mid+large"],
        default=None,
        help="Filter universe by market cap bucket (small=<$0.5B, mid=$0.5-2B, large=>$2B)"
    )

    args = parser.parse_args()

    try:
        results = run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            use_production_scorer=args.use_production_scorer,
            universe=args.universe,
            price_file=args.price_file,
            output_file=args.output,
            frequency_days=args.frequency,
            mcap_filter=args.mcap_filter,
        )

        # Exit with success if we got results
        if results:
            sys.exit(0)
        else:
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure the price file exists at the specified path.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError running backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
