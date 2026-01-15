#!/usr/bin/env python3
"""
Wake Robin Case Data Updater

Refreshes case files with live market data, financials, and trial status.

This script automatically updates case files in data/inputs/ with the latest
data from:
- Yahoo Finance: Price, volume, market cap
- SEC EDGAR: Cash position, debt, R&D expenses
- ClinicalTrials.gov: Trial status, phase, enrollment

Features:
- Staleness detection (24-hour cache by default)
- Force refresh option
- Audit trail of data sources
- Canonical JSON output for determinism

Usage:
    python update_case.py TICKER [--force]
    python update_case.py XYZ              # Update if stale
    python update_case.py XYZ --force      # Force refresh
    python update_case.py --all            # Update all case files

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_sources.yahoo_finance import fetch_quote, fetch_key_statistics
from data_sources.sec_edgar import fetch_company_facts, lookup_cik
from data_sources.clinicaltrials_gov import fetch_trial_status

# Try to import canonical JSON helper
try:
    from utils.json_canonical import to_canonical_json_pretty
except ImportError:
    # Fallback to standard JSON
    def to_canonical_json_pretty(data):
        return json.dumps(data, indent=2, sort_keys=True)


# Module metadata
__version__ = "1.0.0"

# Configuration
CACHE_HOURS = 24  # Refresh if older than 24 hours
DATA_DIR = Path("data/inputs")


def load_case(ticker: str) -> Optional[Dict]:
    """
    Load case file from data/inputs/.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Case data dict, or None if not found
    """
    case_path = DATA_DIR / f"{ticker.upper()}.json"
    if not case_path.exists():
        print(f"  Case file not found: {case_path}")
        return None

    try:
        with open(case_path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"  Invalid JSON in case file: {e}")
        return None


def save_case(ticker: str, case_data: Dict) -> bool:
    """
    Save updated case file with canonical JSON.

    Args:
        ticker: Stock ticker symbol
        case_data: Case data dict to save

    Returns:
        True if successful, False otherwise
    """
    case_path = DATA_DIR / f"{ticker.upper()}.json"
    case_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(case_path, 'w') as f:
            f.write(to_canonical_json_pretty(case_data))
        print(f"  Updated case file: {case_path}")
        return True
    except Exception as e:
        print(f"  Failed to save case file: {e}")
        return False


def needs_refresh(case_data: Dict) -> bool:
    """
    Check if case data is stale (>24 hours old).

    Args:
        case_data: Case data dict

    Returns:
        True if refresh needed, False otherwise
    """
    last_updated = case_data.get("last_updated")
    if not last_updated:
        return True

    try:
        last_update_time = datetime.fromisoformat(last_updated)
        age_hours = (datetime.now() - last_update_time).total_seconds() / 3600
        return age_hours > CACHE_HOURS
    except ValueError:
        return True


def update_market_data(case_data: Dict) -> Dict:
    """
    Refresh price, volume, market cap from Yahoo Finance.

    Args:
        case_data: Current case data

    Returns:
        Updated case data
    """
    ticker = case_data.get("ticker", "UNKNOWN")
    print(f"  Fetching market data for {ticker}...")

    quote = fetch_quote(ticker)
    if quote:
        case_data.update({
            "price": float(quote["price"]),
            "volume": quote["volume"],
            "market_cap": float(quote["market_cap"]),
            "currency": quote["currency"],
            "exchange": quote["exchange"],
            "market_data_timestamp": datetime.fromtimestamp(quote["timestamp"]).isoformat() if quote["timestamp"] else datetime.now().isoformat()
        })
        print(f"    Price: ${quote['price']}")
        print(f"    Market Cap: ${quote['market_cap']:,.0f}")
    else:
        print(f"    Market data fetch failed, using cached values")

    # Also fetch key statistics
    stats = fetch_key_statistics(ticker)
    if stats:
        if stats.get("beta"):
            case_data["beta"] = stats["beta"]
        if stats.get("short_percent"):
            case_data["short_percent"] = stats["short_percent"]
        if stats.get("shares_outstanding"):
            case_data["shares_outstanding"] = stats["shares_outstanding"]

    return case_data


def update_financials(case_data: Dict) -> Dict:
    """
    Refresh cash position from SEC EDGAR.

    Args:
        case_data: Current case data

    Returns:
        Updated case data
    """
    ticker = case_data.get("ticker", "UNKNOWN")
    cik = case_data.get("cik")

    # Look up CIK if not provided
    if not cik:
        print(f"  Looking up CIK for {ticker}...")
        cik = lookup_cik(ticker)
        if cik:
            case_data["cik"] = cik
            print(f"    Found CIK: {cik}")
        else:
            print(f"    Could not find CIK, skipping financial update")
            return case_data

    print(f"  Fetching financials from SEC EDGAR...")

    facts = fetch_company_facts(cik)
    if facts:
        if facts.get("cash"):
            case_data["cash"] = facts["cash"]
            print(f"    Cash: ${facts['cash']:,.0f}")
        if facts.get("total_debt"):
            case_data["total_debt"] = facts["total_debt"]
        if facts.get("revenue"):
            case_data["revenue"] = facts["revenue"]
        if facts.get("rd_expense"):
            case_data["rd_expense"] = facts["rd_expense"]

        case_data["financials_filing_date"] = facts.get("filing_date")
        case_data["financials_form_type"] = facts.get("form_type")
        case_data["financials_fiscal_period"] = facts.get("fiscal_period")

        if facts.get("filing_date"):
            print(f"    Filing: {facts['form_type']} on {facts['filing_date']}")
    else:
        print(f"    Financial data fetch failed, using cached values")

    return case_data


def update_trials(case_data: Dict) -> Dict:
    """
    Refresh clinical trial status from ClinicalTrials.gov.

    Args:
        case_data: Current case data

    Returns:
        Updated case data
    """
    nct_ids = case_data.get("nct_ids", [])
    if not nct_ids:
        print(f"  No NCT IDs provided, skipping trial updates")
        return case_data

    print(f"  Fetching trial status for {len(nct_ids)} trial(s)...")

    updated_trials = []
    for nct_id in nct_ids:
        trial = fetch_trial_status(nct_id)
        if trial:
            updated_trials.append({
                "nct_id": trial["nct_id"],
                "status": trial["status"],
                "phase": trial["phase"],
                "enrollment": trial["enrollment"],
                "primary_completion_date": trial["primary_completion_date"],
                "last_update": trial["last_update"],
                "title": trial.get("title", ""),
                "sponsor": trial.get("sponsor", ""),
            })
            print(f"    {nct_id}: {trial['status']} ({trial['phase']})")
        else:
            print(f"    {nct_id}: Fetch failed")

    if updated_trials:
        case_data["trials"] = updated_trials

    return case_data


def update_case_data(ticker: str, force: bool = False) -> bool:
    """
    Main update function - refresh all data sources.

    Args:
        ticker: Stock ticker symbol
        force: Force update even if cache is fresh

    Returns:
        True if update successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"UPDATING CASE DATA: {ticker.upper()}")
    print(f"{'='*60}")

    # Load existing case or create new
    case_data = load_case(ticker)
    if not case_data:
        # Create new case file
        print(f"  Creating new case file for {ticker}")
        case_data = {"ticker": ticker.upper()}

    # Check staleness
    if not force and not needs_refresh(case_data):
        last_updated = case_data.get("last_updated", "unknown")
        try:
            age_hours = (datetime.now() - datetime.fromisoformat(last_updated)).total_seconds() / 3600
            print(f"  Cache is fresh ({age_hours:.1f} hours old), skipping update")
        except ValueError:
            print(f"  Cache is fresh, skipping update")
        print(f"  Use --force to refresh anyway")
        return True

    # Update each data source
    case_data = update_market_data(case_data)
    case_data = update_financials(case_data)
    case_data = update_trials(case_data)

    # Update metadata
    case_data["last_updated"] = datetime.now().isoformat()
    case_data["data_sources"] = ["yahoo_finance", "sec_edgar", "clinicaltrials_gov"]
    case_data["update_version"] = __version__

    # Save updated case
    success = save_case(ticker, case_data)

    print(f"\n{'='*60}")
    if success:
        print(f"  UPDATE COMPLETE: {ticker.upper()}")
    else:
        print(f"  UPDATE FAILED: {ticker.upper()}")
    print(f"{'='*60}")

    return success


def update_all_cases(force: bool = False) -> Dict[str, bool]:
    """
    Update all case files in data/inputs/.

    Args:
        force: Force update even if cache is fresh

    Returns:
        Dict mapping ticker to success status
    """
    results = {}

    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return results

    case_files = list(DATA_DIR.glob("*.json"))
    if not case_files:
        print(f"No case files found in {DATA_DIR}")
        return results

    print(f"\n{'='*60}")
    print(f"UPDATING ALL CASE FILES ({len(case_files)} total)")
    print(f"{'='*60}")

    for case_file in sorted(case_files):
        ticker = case_file.stem.upper()
        try:
            success = update_case_data(ticker, force=force)
            results[ticker] = success
        except Exception as e:
            print(f"  Error updating {ticker}: {e}")
            results[ticker] = False

    # Summary
    print(f"\n{'='*60}")
    print(f"UPDATE SUMMARY")
    print(f"{'='*60}")
    succeeded = sum(1 for s in results.values() if s)
    failed = sum(1 for s in results.values() if not s)
    print(f"  Succeeded: {succeeded}")
    print(f"  Failed: {failed}")

    return results


def create_sample_case(ticker: str) -> Dict:
    """
    Create a sample case file structure.

    Args:
        ticker: Stock ticker symbol

    Returns:
        Sample case data dict
    """
    return {
        "ticker": ticker.upper(),
        "cik": None,  # Will be looked up automatically
        "nct_ids": [],  # Add NCT IDs for trials to track
        "price": None,
        "volume": None,
        "market_cap": None,
        "currency": "USD",
        "exchange": None,
        "cash": None,
        "total_debt": None,
        "revenue": None,
        "rd_expense": None,
        "trials": [],
        "last_updated": None,
        "data_sources": [],
    }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Update case files with live data from Yahoo Finance, SEC EDGAR, and ClinicalTrials.gov"
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        help="Stock ticker symbol to update"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force update even if cache is fresh"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Update all case files"
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create new case file for ticker"
    )

    args = parser.parse_args()

    if args.all:
        results = update_all_cases(force=args.force)
        return 0 if all(results.values()) else 1

    if not args.ticker:
        parser.print_help()
        return 1

    if args.create:
        # Create new case file
        ticker = args.ticker.upper()
        case_path = DATA_DIR / f"{ticker}.json"
        if case_path.exists():
            print(f"Case file already exists: {case_path}")
            return 1

        case_data = create_sample_case(ticker)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(case_path, 'w') as f:
            f.write(to_canonical_json_pretty(case_data))
        print(f"Created case file: {case_path}")
        print(f"Edit the file to add CIK and NCT IDs, then run update_case.py {ticker}")
        return 0

    success = update_case_data(args.ticker, force=args.force)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
