#!/usr/bin/env python3
"""
collect_etf_universe_data.py

Orchestrates data collection for the full ETF universe (~200 stocks).
Collects market data, financial data, and clinical data in parallel batches.

Usage:
    python collect_etf_universe_data.py --as-of-date 2026-01-06 --output-dir etf_universe_data
"""

import argparse
import json
import sys
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import subprocess

# Try to import yfinance for financial data
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

# Try to import yahoo_collector for market data
try:
    from wake_robin_data_pipeline.collectors.yahoo_collector import (
        collect_yahoo_data,
        collect_batch as yahoo_collect_batch,
    )
    HAS_YAHOO_COLLECTOR = True
except ImportError:
    HAS_YAHOO_COLLECTOR = False


def load_ticker_list(universe_file: str) -> List[str]:
    """Load tickers from universe template."""
    with open(universe_file, 'r') as f:
        data = json.load(f)
    
    tickers = [sec['ticker'] for sec in data]
    print(f"Loaded {len(tickers)} tickers from {universe_file}")
    return tickers


def collect_market_data_batch(tickers: List[str], as_of_date: str, output_dir: Path) -> Dict:
    """
    Collect market data (defensive features) for all tickers using yahoo_collector.

    Collects:
    - Current price, market cap
    - 52-week high/low
    - Volume data
    - Balance sheet basics (for supplementary data)
    """
    print("\n" + "="*80)
    print("COLLECTING MARKET DATA (defensive_features)")
    print("="*80)
    print(f"Tickers: {len(tickers)}")
    print(f"As of date: {as_of_date}")

    market_data_file = output_dir / "market_data_raw.json"

    if not HAS_YAHOO_COLLECTOR:
        print("\nWARNING: yahoo_collector not available")
        print("Install with: pip install yfinance")
        print("Or ensure wake_robin_data_pipeline is in PYTHONPATH")
        return {
            "status": "error",
            "message": "yahoo_collector not available",
            "tickers": len(tickers),
            "output": str(market_data_file)
        }

    try:
        # Use yahoo_collector's batch collection
        print(f"\nCollecting data for {len(tickers)} tickers...")
        results = yahoo_collect_batch(tickers, delay_seconds=0.5, force_refresh=False)

        # Convert to list format for output
        market_data = []
        successful = 0
        for ticker, data in results.items():
            if data.get('success'):
                successful += 1
                market_data.append({
                    "ticker": ticker,
                    "price": data.get('price', {}).get('current'),
                    "market_cap": data.get('market_cap', {}).get('value'),
                    "52_week_high": data.get('price', {}).get('52_week_high'),
                    "52_week_low": data.get('price', {}).get('52_week_low'),
                    "volume_avg_30d": data.get('volume', {}).get('average_30d'),
                    "shares_outstanding": data.get('shares_outstanding'),
                    "balance_sheet": data.get('balance_sheet', {}),
                    "cash_flow": data.get('cash_flow', {}),
                    "collected_at": datetime.now().isoformat(),
                    "source": "yahoo_collector",
                })
            else:
                market_data.append({
                    "ticker": ticker,
                    "error": data.get('error', 'Unknown error'),
                    "collected_at": datetime.now().isoformat(),
                })

        # Save to file
        with open(market_data_file, 'w') as f:
            json.dump(market_data, f, indent=2)

        print(f"\nMarket data collected: {successful}/{len(tickers)} successful")
        print(f"Saved to: {market_data_file}")

        return {
            "status": "success",
            "message": f"Collected {successful}/{len(tickers)} tickers",
            "tickers": len(tickers),
            "successful": successful,
            "output": str(market_data_file)
        }

    except Exception as e:
        print(f"\nERROR collecting market data: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "tickers": len(tickers),
            "output": str(market_data_file)
        }


def collect_financial_data_batch(tickers: List[str], as_of_date: str, output_dir: Path) -> Dict:
    """
    Collect financial data for all tickers using yfinance.

    Collects:
    - Cash & equivalents
    - Total debt (aggregated from components)
    - Operating cash flow (burn rate proxy)
    - R&D expense
    - Total assets/liabilities
    """
    print("\n" + "="*80)
    print("COLLECTING FINANCIAL DATA")
    print("="*80)
    print(f"Tickers: {len(tickers)}")
    print(f"As of date: {as_of_date}")

    financial_data_file = output_dir / "financial_data_raw.json"

    if not HAS_YFINANCE:
        print("\nWARNING: yfinance not available")
        print("Install with: pip install yfinance")
        return {
            "status": "error",
            "message": "yfinance not available",
            "tickers": len(tickers),
            "output": str(financial_data_file)
        }

    try:
        financial_data = []
        successful = 0
        total = len(tickers)

        print(f"\nCollecting financial data for {total} tickers...")

        for i, ticker in enumerate(tickers, 1):
            print(f"[{i:3d}/{total}] {ticker:6s}", end=" ", flush=True)

            try:
                stock = yf.Ticker(ticker)

                # Get quarterly balance sheet
                bs = stock.quarterly_balance_sheet
                cf = stock.quarterly_cashflow

                record = {"ticker": ticker}

                if bs is not None and not bs.empty:
                    latest_col = bs.columns[0]
                    period_date = latest_col.strftime('%Y-%m-%d') if hasattr(latest_col, 'strftime') else str(latest_col)
                    record["period_date"] = period_date

                    # Helper to safely get values
                    def get_bs_value(keys):
                        if isinstance(keys, str):
                            keys = [keys]
                        for key in keys:
                            if key in bs.index:
                                val = bs.loc[key, latest_col]
                                if val is not None and str(val) != 'nan':
                                    return float(val)
                        return None

                    record["cash"] = get_bs_value(['Cash And Cash Equivalents'])
                    record["total_liquidity"] = get_bs_value(['Cash Cash Equivalents And Short Term Investments'])
                    record["total_assets"] = get_bs_value(['Total Assets'])
                    record["total_liabilities"] = get_bs_value(['Total Liabilities Net Minority Interest', 'Current Liabilities'])
                    record["stockholders_equity"] = get_bs_value(['Stockholders Equity', 'Common Stock Equity'])
                    record["long_term_debt"] = get_bs_value(['Long Term Debt', 'Long Term Debt And Capital Lease Obligation'])
                    record["current_debt"] = get_bs_value(['Current Debt', 'Current Debt And Capital Lease Obligation'])
                    record["total_debt"] = get_bs_value(['Total Debt'])

                    # Aggregate total debt if not directly available
                    if record["total_debt"] is None:
                        debt_sum = 0
                        if record["long_term_debt"]:
                            debt_sum += record["long_term_debt"]
                        if record["current_debt"]:
                            debt_sum += record["current_debt"]
                        if debt_sum > 0:
                            record["total_debt"] = debt_sum
                            record["total_debt_aggregated"] = True

                if cf is not None and not cf.empty:
                    latest_col = cf.columns[0]

                    def get_cf_value(keys):
                        if isinstance(keys, str):
                            keys = [keys]
                        for key in keys:
                            if key in cf.index:
                                val = cf.loc[key, latest_col]
                                if val is not None and str(val) != 'nan':
                                    return float(val)
                        return None

                    record["operating_cash_flow"] = get_cf_value(['Operating Cash Flow', 'Cash Flow From Continuing Operating Activities'])
                    record["free_cash_flow"] = get_cf_value(['Free Cash Flow'])

                # Calculate runway if we have cash and burn
                if record.get("total_liquidity") and record.get("operating_cash_flow"):
                    ocf = record["operating_cash_flow"]
                    if ocf < 0:  # Negative CFO = burning cash
                        quarterly_burn = abs(ocf)
                        record["quarterly_burn"] = quarterly_burn
                        record["runway_quarters"] = record["total_liquidity"] / quarterly_burn if quarterly_burn > 0 else None

                record["collected_at"] = datetime.now().isoformat()
                record["source"] = "yfinance"

                # Check if we got meaningful data
                if record.get("cash") or record.get("total_assets"):
                    successful += 1
                    cash_str = f"${record.get('cash', 0)/1e9:.2f}B" if record.get('cash') else "N/A"
                    print(f"OK Cash: {cash_str}")
                else:
                    print(f"No financial data")

                financial_data.append(record)

            except Exception as e:
                print(f"ERROR: {e}")
                financial_data.append({
                    "ticker": ticker,
                    "error": str(e),
                    "collected_at": datetime.now().isoformat(),
                })

            # Rate limit
            if i < total:
                time.sleep(0.3)

        # Save to file
        with open(financial_data_file, 'w') as f:
            json.dump(financial_data, f, indent=2)

        print(f"\nFinancial data collected: {successful}/{total} successful")
        print(f"Saved to: {financial_data_file}")

        return {
            "status": "success",
            "message": f"Collected {successful}/{total} tickers",
            "tickers": total,
            "successful": successful,
            "output": str(financial_data_file)
        }

    except Exception as e:
        print(f"\nERROR collecting financial data: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "tickers": len(tickers),
            "output": str(financial_data_file)
        }


def fetch_ctgov_trials_for_sponsor(sponsor_name: str) -> List[Dict]:
    """
    Fetch clinical trials from CT.gov API v2 for a given sponsor.
    """
    base_url = "https://clinicaltrials.gov/api/v2/studies"

    params = {
        "query.spons": sponsor_name,
        "pageSize": 100,
        "fields": "NCTId,BriefTitle,Phase,OverallStatus,Condition,StartDate,PrimaryCompletionDate,LastUpdatePostDate,ResultsFirstPostDate"
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        studies = data.get("studies", [])
        trials = []

        for study in studies:
            protocol = study.get("protocolSection", {})
            id_module = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            conditions_module = protocol.get("conditionsModule", {})
            design_module = protocol.get("designModule", {})

            trial = {
                "nct_id": id_module.get("nctId"),
                "title": id_module.get("briefTitle"),
                "phase": design_module.get("phases", ["Unknown"])[0] if design_module.get("phases") else "Unknown",
                "status": status_module.get("overallStatus"),
                "conditions": conditions_module.get("conditions", []),
                "start_date": status_module.get("startDateStruct", {}).get("date"),
                "primary_completion_date": status_module.get("primaryCompletionDateStruct", {}).get("date"),
                "last_update_posted": status_module.get("lastUpdatePostDateStruct", {}).get("date"),
            }
            trials.append(trial)

        return trials

    except Exception as e:
        return []


def collect_clinical_data_batch(tickers: List[str], as_of_date: str, output_dir: Path) -> Dict:
    """
    Collect clinical trial data for all tickers using CT.gov API v2.

    Collects:
    - All trials where ticker/company is sponsor
    - Phase, status, conditions
    - Key date fields for PIT filtering
    """
    print("\n" + "="*80)
    print("COLLECTING CLINICAL DATA (ClinicalTrials.gov API v2)")
    print("="*80)
    print(f"Tickers: {len(tickers)}")
    print(f"As of date: {as_of_date}")

    clinical_data_file = output_dir / "clinical_data_raw.json"

    try:
        all_trials = []
        successful = 0
        total_trials = 0
        total = len(tickers)

        print(f"\nSearching CT.gov for {total} tickers...")
        print("Note: Using ticker as sponsor search term (may need company name mapping)")

        for i, ticker in enumerate(tickers, 1):
            print(f"[{i:3d}/{total}] {ticker:6s}", end=" ", flush=True)

            try:
                # Search using ticker as sponsor name
                # In production, you may want a ticker-to-company-name mapping
                trials = fetch_ctgov_trials_for_sponsor(ticker)

                if trials:
                    successful += 1
                    total_trials += len(trials)

                    # Add ticker to each trial record
                    for trial in trials:
                        trial["ticker"] = ticker
                        trial["collected_at"] = datetime.now().isoformat()
                        all_trials.append(trial)

                    print(f"OK {len(trials):3d} trials")
                else:
                    print(f"No trials found")

            except Exception as e:
                print(f"ERROR: {e}")

            # Rate limit - CT.gov requests 1 request per second
            if i < total:
                time.sleep(1.0)

        # Save to file
        with open(clinical_data_file, 'w') as f:
            json.dump(all_trials, f, indent=2)

        print(f"\nClinical data collected:")
        print(f"  Tickers with trials: {successful}/{total}")
        print(f"  Total trials: {total_trials}")
        print(f"Saved to: {clinical_data_file}")

        return {
            "status": "success",
            "message": f"Found {total_trials} trials for {successful}/{total} tickers",
            "tickers": total,
            "tickers_with_trials": successful,
            "total_trials": total_trials,
            "output": str(clinical_data_file)
        }

    except Exception as e:
        print(f"\nERROR collecting clinical data: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
            "tickers": len(tickers),
            "output": str(clinical_data_file)
        }


def build_universe_snapshot(output_dir: Path, as_of_date: str) -> str:
    """
    Combine all collected data into final universe snapshot.
    """
    print("\n" + "="*80)
    print("BUILDING UNIVERSE SNAPSHOT")
    print("="*80)
    
    output_file = output_dir / f"universe_snapshot_{as_of_date}.json"
    
    print(f"Combining data sources:")
    print(f"  • Market data (defensive_features)")
    print(f"  • Financial data (cash, burn, runway)")
    print(f"  • Clinical data (lead programs)")
    
    print(f"\nOutput: {output_file}")
    print("TO IMPLEMENT: Merge all data sources into final universe structure")
    
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description="Collect ETF universe data")
    parser.add_argument("--universe-file", default="etf_universe_template.json",
                       help="Universe template file")
    parser.add_argument("--as-of-date", required=True, help="Date YYYY-MM-DD")
    parser.add_argument("--output-dir", default="etf_universe_data",
                       help="Output directory")
    parser.add_argument("--skip-market", action="store_true", help="Skip market data collection")
    parser.add_argument("--skip-financial", action="store_true", help="Skip financial data collection")
    parser.add_argument("--skip-clinical", action="store_true", help="Skip clinical data collection")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("ETF UNIVERSE DATA COLLECTION")
    print("="*80)
    print(f"As of date: {args.as_of_date}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load tickers
    tickers = load_ticker_list(args.universe_file)
    
    # Collect data
    results = {}
    
    if not args.skip_market:
        results['market'] = collect_market_data_batch(tickers, args.as_of_date, output_dir)
    
    if not args.skip_financial:
        results['financial'] = collect_financial_data_batch(tickers, args.as_of_date, output_dir)
    
    if not args.skip_clinical:
        results['clinical'] = collect_clinical_data_batch(tickers, args.as_of_date, output_dir)
    
    # Build final universe
    universe_file = build_universe_snapshot(output_dir, args.as_of_date)
    
    # Summary
    print("\n" + "="*80)
    print("COLLECTION SUMMARY")
    print("="*80)
    print(f"Tickers processed: {len(tickers)}")
    print(f"Data collected:")
    for data_type, result in results.items():
        print(f"  • {data_type}: {result['status']}")
    
    print(f"\nFinal universe: {universe_file}")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. IMPLEMENT DATA COLLECTION:")
    print("   This script is a FRAMEWORK - you need to add:")
    print("   - Market data fetching (Yahoo Finance, Alpha Vantage, etc.)")
    print("   - Financial data parsing (SEC EDGAR, FMP API, etc.)")
    print("   - Clinical data API calls (ClinicalTrials.gov)")
    print()
    print("2. OR USE YOUR EXISTING PIPELINE:")
    print("   If you have wake_robin_data_pipeline working:")
    print()
    print("   cd wake_robin_data_pipeline")
    print(f"   python collect_universe_data.py --tickers {','.join(tickers[:5])}... --as-of-date {args.as_of_date}")
    print()
    print("3. ONCE DATA COLLECTED:")
    print("   python run_screen.py \\")
    print(f"       --as-of-date {args.as_of_date} \\")
    print("       --data-dir etf_universe_data \\")
    print("       --output etf_screening_results.json")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
