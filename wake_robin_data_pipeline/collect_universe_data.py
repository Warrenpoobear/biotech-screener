#!/usr/bin/env python3
"""
collect_universe_data.py - Master orchestrator for Wake Robin data pipeline

Orchestrates collection from:
1. Yahoo Finance (market data)
2. SEC EDGAR (financial data)
3. ClinicalTrials.gov (trial data)
4. Time-Series Data (historical prices/returns for defensive overlays)
5. Defensive Overlays (volatility, correlation, gates, position sizing)

Outputs unified snapshot with data quality metrics.

Usage:
    python collect_universe_data.py                          # Use full universe (346 tickers)
    python collect_universe_data.py --universe path/to/file  # Use custom universe file
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List

# Add collectors to path
sys.path.insert(0, str(Path(__file__).parent))

from collectors import yahoo_collector, sec_collector, trials_collector, time_series_collector
from defensive_overlays import enrich_universe_with_defensive_overlays, print_defensive_summary

# Ticker validation for fail-loud data quality
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from validators.ticker_validator import validate_ticker_list
    HAS_TICKER_VALIDATION = True
except ImportError:
    HAS_TICKER_VALIDATION = False

def load_universe(universe_file: str = "universe/full_universe.json") -> Dict:
    """Load universe configuration."""
    universe_path = Path(__file__).parent / universe_file

    if not universe_path.exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}")

    with open(universe_path) as f:
        return json.load(f)

def merge_data_sources(ticker: str, yahoo_data: dict, sec_data: dict, trials_data: dict, time_series_data: dict = None) -> dict:
    """
    Merge data from all sources into unified record.

    Returns:
        Unified company record with provenance tracking
    """
    record = {
        "ticker": ticker,
        "as_of_date": datetime.now().isoformat(),
        "data_quality": {
            "yahoo_success": yahoo_data.get('success', False),
            "sec_success": sec_data.get('success', False),
            "trials_success": trials_data.get('success', False),
            "overall_coverage": 0.0
        }
    }

    # Market data from Yahoo
    if yahoo_data.get('success'):
        record['market_data'] = {
            "price": yahoo_data['price']['current'],
            "market_cap": yahoo_data['market_cap']['value'],
            "shares_outstanding": yahoo_data.get('shares_outstanding', 0),
            "volume_avg_30d": yahoo_data['volume']['average_30d'],
            "52_week_high": yahoo_data['price']['52_week_high'],
            "52_week_low": yahoo_data['price']['52_week_low'],
            "pe_ratio": yahoo_data['valuation'].get('pe_ratio'),
            "company_name": yahoo_data['company_info']['name'],
            "sector": yahoo_data['company_info']['sector'],
            "industry": yahoo_data['company_info']['industry']
        }
        record['data_quality']['has_price'] = True
    else:
        record['market_data'] = {"error": yahoo_data.get('error')}
        record['data_quality']['has_price'] = False

    # Financial data from SEC
    if sec_data.get('success'):
        record['financials'] = sec_data['financials']
        record['financials']['cik'] = sec_data['cik']
        record['data_quality']['financial_coverage'] = sec_data['coverage']['pct_complete']
        record['data_quality']['has_cash'] = sec_data['coverage']['has_cash']
        record['data_quality']['has_balance_sheet'] = sec_data['coverage']['has_balance_sheet']
    else:
        record['financials'] = {"error": sec_data.get('error')}
        record['data_quality']['financial_coverage'] = 0.0
        record['data_quality']['has_cash'] = False
        record['data_quality']['has_balance_sheet'] = False

    # Clinical trial data
    if trials_data.get('success'):
        record['clinical'] = {
            "total_trials": trials_data['summary']['total_trials'],
            "active_trials": trials_data['summary']['active_trials'],
            "completed_trials": trials_data['summary']['completed_trials'],
            "lead_stage": trials_data['summary']['lead_stage'],
            "by_phase": trials_data['summary']['by_phase'],
            "conditions": trials_data['summary'].get('conditions', []),
            "top_trials": trials_data['trials'][:5]  # Top 5 for reference
        }
        record['data_quality']['has_clinical'] = True
    else:
        record['clinical'] = {"error": trials_data.get('error')}
        record['data_quality']['has_clinical'] = False

    # Time-series data (historical prices/returns for defensive overlays)
    if time_series_data and time_series_data.get('success'):
        record['time_series'] = {
            "prices": time_series_data['time_series']['prices'],
            "returns": time_series_data['time_series']['returns'],
            "volumes": time_series_data['time_series']['volumes'],
            "num_days": time_series_data['time_series']['num_days'],
            "lookback_days": time_series_data['time_series']['lookback_days'],
            "adv_20d": time_series_data['liquidity']['adv_20d']
        }
        record['data_quality']['has_time_series'] = True
        record['data_quality']['time_series_days'] = time_series_data['time_series']['num_days']
    else:
        record['time_series'] = None
        record['data_quality']['has_time_series'] = False
        record['data_quality']['time_series_days'] = 0

    # Calculate overall coverage score
    scores = []
    if record['data_quality']['has_price']:
        scores.append(100.0)

    if record['data_quality']['financial_coverage'] > 0:
        scores.append(record['data_quality']['financial_coverage'])

    if record['data_quality']['has_clinical']:
        scores.append(80.0)  # Weight clinical data at 80% if available

    if record['data_quality']['has_time_series']:
        scores.append(90.0)  # Weight time-series data at 90% if available

    record['data_quality']['overall_coverage'] = sum(scores) / len(scores) if scores else 0.0

    # Provenance tracking
    record['provenance'] = {
        "collection_timestamp": datetime.now().isoformat(),
        "sources": {
            "yahoo_finance": yahoo_data.get('provenance', {}),
            "sec_edgar": sec_data.get('provenance', {}),
            "clinicaltrials_gov": trials_data.get('provenance', {}),
            "time_series": time_series_data.get('provenance', {}) if time_series_data else {}
        }
    }

    return record

def generate_quality_report(records: List[dict]) -> dict:
    """Generate data quality summary across all records."""
    total = len([r for r in records if r['ticker'] != '_XBI_BENCHMARK_'])

    quality = {
        "universe_size": total,
        "collection_date": datetime.now().isoformat(),
        "coverage": {
            "price_data": sum(1 for r in records if r['ticker'] != '_XBI_BENCHMARK_' and r['data_quality']['has_price']),
            "financial_data": sum(1 for r in records if r['ticker'] != '_XBI_BENCHMARK_' and r['data_quality']['financial_coverage'] >= 50),
            "clinical_data": sum(1 for r in records if r['ticker'] != '_XBI_BENCHMARK_' and r['data_quality']['has_clinical']),
            "time_series_data": sum(1 for r in records if r['ticker'] != '_XBI_BENCHMARK_' and r['data_quality'].get('has_time_series', False)),
            "defensive_features": sum(1 for r in records if r['ticker'] != '_XBI_BENCHMARK_' and r.get('defensive_features')),
        },
        "coverage_pct": {},
        "avg_financial_coverage": 0.0,
        "avg_overall_coverage": 0.0,
        "tickers_by_quality": {
            "excellent": [],  # >80% coverage
            "good": [],       # 60-80% coverage
            "fair": [],       # 40-60% coverage
            "poor": []        # <40% coverage
        }
    }

    # Calculate percentages
    quality['coverage_pct'] = {
        "price_data": quality['coverage']['price_data'] / total * 100,
        "financial_data": quality['coverage']['financial_data'] / total * 100,
        "clinical_data": quality['coverage']['clinical_data'] / total * 100,
        "time_series_data": quality['coverage']['time_series_data'] / total * 100,
        "defensive_features": quality['coverage']['defensive_features'] / total * 100,
    }

    # Calculate averages (excluding XBI)
    non_xbi_records = [r for r in records if r['ticker'] != '_XBI_BENCHMARK_']
    financial_coverages = [r['data_quality']['financial_coverage'] for r in non_xbi_records]
    overall_coverages = [r['data_quality']['overall_coverage'] for r in non_xbi_records]

    quality['avg_financial_coverage'] = sum(financial_coverages) / total if total > 0 else 0
    quality['avg_overall_coverage'] = sum(overall_coverages) / total if total > 0 else 0

    # Categorize by quality
    for record in non_xbi_records:
        ticker = record['ticker']
        coverage = record['data_quality']['overall_coverage']

        if coverage >= 80:
            quality['tickers_by_quality']['excellent'].append(ticker)
        elif coverage >= 60:
            quality['tickers_by_quality']['good'].append(ticker)
        elif coverage >= 40:
            quality['tickers_by_quality']['fair'].append(ticker)
        else:
            quality['tickers_by_quality']['poor'].append(ticker)

    return quality

def save_snapshot(records: List[dict], quality_report: dict):
    """Save universe snapshot and quality report to outputs directory."""
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full snapshot
    snapshot_file = output_dir / f"universe_snapshot_{timestamp}.json"
    with open(snapshot_file, 'w') as f:
        json.dump(records, f, indent=2)

    # Save quality report
    report_file = output_dir / f"quality_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(quality_report, f, indent=2)

    # Save latest symlink (for easy access)
    latest_snapshot = output_dir / "universe_snapshot_latest.json"
    latest_report = output_dir / "quality_report_latest.json"

    if latest_snapshot.exists():
        latest_snapshot.unlink()
    if latest_report.exists():
        latest_report.unlink()

    import shutil
    shutil.copy(snapshot_file, latest_snapshot)
    shutil.copy(report_file, latest_report)

    return snapshot_file, report_file

def print_summary(quality_report: dict):
    """Print human-readable summary to console."""
    print("\n" + "="*60)
    print("DATA QUALITY SUMMARY")
    print("="*60)

    print(f"\nUniverse Size: {quality_report['universe_size']} companies")
    print(f"Collection Date: {quality_report['collection_date']}")

    print("\nData Source Coverage:")
    cov = quality_report['coverage_pct']
    print(f"  • Price Data (Yahoo):     {cov['price_data']:.1f}%")
    print(f"  • Financial Data (SEC):   {cov['financial_data']:.1f}%")
    print(f"  • Clinical Data (CT.gov): {cov['clinical_data']:.1f}%")
    print(f"  • Time-Series Data:       {cov['time_series_data']:.1f}%")
    print(f"  • Defensive Features:     {cov['defensive_features']:.1f}%")

    print("\nAverage Coverage:")
    print(f"  • Financial Metrics: {quality_report['avg_financial_coverage']:.1f}%")
    print(f"  • Overall Data:      {quality_report['avg_overall_coverage']:.1f}%")

    print("\nQuality Distribution:")
    qual = quality_report['tickers_by_quality']
    print(f"  • Excellent (>80%): {len(qual['excellent'])} tickers")
    print(f"  • Good (60-80%):    {len(qual['good'])} tickers")
    print(f"  • Fair (40-60%):    {len(qual['fair'])} tickers")
    print(f"  • Poor (<40%):      {len(qual['poor'])} tickers")

    if qual['poor']:
        print(f"\nTickers needing attention: {', '.join(qual['poor'])}")

    print("\n" + "="*60)

def main(universe_file: str = "universe/full_universe.json"):
    """Main execution flow."""
    print("\n🚀 Wake Robin Data Pipeline - Universe Collection")
    print("="*60)

    # Load universe
    print("\n1. Loading universe configuration...")
    universe = load_universe(universe_file)
    tickers = [t['ticker'] for t in universe['tickers']]
    universe_name = "full" if "full" in universe_file else "pilot"
    print(f"   ✓ Loaded {len(tickers)} tickers from {universe_name} universe")

    # Validate tickers (fail-loud on contaminated data)
    if HAS_TICKER_VALIDATION:
        print("\n1b. Validating tickers...")
        validation_result = validate_ticker_list(tickers)
        if validation_result['invalid']:
            invalid_sample = list(validation_result['invalid'].items())[:5]
            raise ValueError(
                f"Universe contains {len(validation_result['invalid'])} invalid tickers. "
                f"Examples: {invalid_sample}. "
                f"Run: python src/scripts/clean_universe.py to fix."
            )
        print(f"   ✓ All {len(tickers)} tickers validated")

    # Collect Yahoo Finance data
    print("\n2. Collecting market data from Yahoo Finance...")
    yahoo_results = yahoo_collector.collect_batch(tickers, delay_seconds=1.0)

    # Collect SEC EDGAR data
    print("\n3. Collecting financial data from SEC EDGAR...")
    sec_results = sec_collector.collect_batch(tickers, delay_seconds=1.0)

    # Collect ClinicalTrials.gov data
    print("\n4. Collecting clinical trial data from ClinicalTrials.gov...")
    ticker_company_map = {
        t['ticker']: t['company']
        for t in universe['tickers']
    }
    trials_results = trials_collector.collect_batch(ticker_company_map, delay_seconds=1.0)

    # Collect time-series data (historical prices/returns)
    print("\n4b. Collecting time-series data (historical prices/returns)...")
    as_of_date = date.today()  # Or pass as parameter for PIT backtesting
    time_series_results = time_series_collector.collect_batch(
        tickers, 
        as_of=as_of_date,
        lookback_days=365,
        delay_seconds=0.1  # Fast due to caching
    )

    # Merge all data sources
    print("\n5. Merging data sources...")
    records = []
    for ticker in tickers:
        yahoo_data = yahoo_results.get(ticker, {})
        sec_data = sec_results.get(ticker, {})
        trials_data = trials_results.get(ticker, {})
        time_series_data = time_series_results.get(ticker, {})

        record = merge_data_sources(ticker, yahoo_data, sec_data, trials_data, time_series_data)
        records.append(record)

    # Add XBI benchmark to records
    if "_XBI_BENCHMARK_" in time_series_results:
        xbi_record = {
            "ticker": "_XBI_BENCHMARK_",
            "time_series": time_series_results["_XBI_BENCHMARK_"]["time_series"],
            "data_quality": {"has_time_series": True}
        }
        records.append(xbi_record)

    print(f"   ✓ Merged data for {len(records)} companies")

    # Compute defensive overlays (volatility, correlation, gates)
    print("\n5b. Computing defensive overlays (volatility, correlation, gates)...")
    xbi_benchmark = next((r for r in records if r['ticker'] == '_XBI_BENCHMARK_'), None)
    enrich_universe_with_defensive_overlays(records, xbi_benchmark)
    print_defensive_summary(records)

    # Generate quality report
    print("\n6. Generating data quality report...")
    quality_report = generate_quality_report(records)

    # Save outputs
    print("\n7. Saving outputs...")
    snapshot_file, report_file = save_snapshot(records, quality_report)
    print(f"   ✓ Snapshot: {snapshot_file.name}")
    print(f"   ✓ Report:   {report_file.name}")

    # Print summary
    print_summary(quality_report)

    print("\n✅ Data collection complete!\n")

    return records, quality_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wake Robin Data Pipeline - Universe Collection")
    parser.add_argument(
        "--universe", "-u",
        default="full",
        help="Universe to use: 'full' (346 tickers) or path to custom JSON file"
    )
    args = parser.parse_args()

    # Resolve universe file path
    if args.universe == "full":
        universe_file = "universe/full_universe.json"
    else:
        universe_file = args.universe

    try:
        records, quality_report = main(universe_file=universe_file)
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
