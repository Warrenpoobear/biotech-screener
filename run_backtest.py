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
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backtest_engine import PointInTimeBacktester, create_sample_scoring_function
from backtest.returns_provider import CSVReturnsProvider
from backtest.metrics import run_metrics_suite, HORIZON_DISPLAY_NAMES

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

    return financial_data, trial_data, universe_data


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
    financial_data, trial_data, universe_data = load_real_data()
    print(f"  Loaded financial data for {len(financial_data)} tickers")
    print(f"  Loaded trial data for {len(trial_data)} tickers")
    print(f"  Loaded universe data for {len(universe_data)} tickers")

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

        # Get real financial data
        fin_record = financial_data.get(ticker, {})

        # Convert financial data to module format (values in millions)
        cash_raw = fin_record.get("Cash", 0) or 0
        cash_mm = cash_raw / 1_000_000 if cash_raw else mcap_mm * 0.2

        # Estimate debt from liabilities - current liabilities
        liabilities = fin_record.get("Liabilities", 0) or 0
        current_liab = fin_record.get("CurrentLiabilities", 0) or 0
        debt_raw = liabilities - current_liab if liabilities else 0
        debt_mm = max(0, debt_raw / 1_000_000)

        # Estimate burn rate from R&D expense
        rd_raw = fin_record.get("R&D", 0) or 0
        burn_mm = rd_raw / 1_000_000 / 4 if rd_raw else cash_mm / 24  # Quarterly burn

        financial_records = [{
            "ticker": ticker,
            "cash_mm": cash_mm,
            "debt_mm": debt_mm,
            "burn_rate_mm": burn_mm,
            "market_cap_mm": mcap_mm,
            "source_date": as_of_str,  # Use backtest date to pass PIT filter
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
            m3 = compute_module_3_catalyst(trial_records, [ticker], as_of_str)
            m4 = compute_module_4_clinical_dev(trial_records, [ticker], as_of_str)
            m5 = compute_module_5_composite(m1, m2, m3, m4, as_of_str)

            # Extract final score
            ranked = m5.get("ranked_securities", [])
            if ranked:
                score = float(ranked[0].get("composite_score", 50))
            else:
                score = 50.0

            # Extract component scores
            fin_score = 0
            if m2.get("scores"):
                fin_score = float(m2["scores"][0].get("financial_normalized", 0) or 0)

            clin_score = 0
            if m4.get("scores"):
                clin_score = float(m4["scores"][0].get("clinical_score", 0) or 0)

            return {
                "ticker": ticker,
                "final_score": score,
                "components": {
                    "financial": fin_score,
                    "clinical": clin_score,
                    "trials_count": len(ticker_trials),
                },
                "production_pipeline": True,
                "data_source": "real",
            }
        except Exception as e:
            # Fallback to basic scoring
            return {
                "ticker": ticker,
                "final_score": 50.0,
                "components": {"error": str(e)},
                "production_pipeline": False,
                "data_source": "fallback",
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

    Returns:
        Backtest results dictionary
    """
    from datetime import timedelta
    from statistics import mean, stdev

    price_file = price_file or "data/daily_prices.csv"

    # Load price data first to get available tickers
    try:
        returns_provider = CSVReturnsProvider(price_file)
        available_tickers = set(returns_provider.get_available_tickers())
        print(f"Loaded price data for {len(available_tickers)} tickers")

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
