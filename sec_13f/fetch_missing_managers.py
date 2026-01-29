#!/usr/bin/env python3
"""
Fetch missing 13F filings and update holdings_snapshots.json

Usage:
    python3 fetch_missing_managers.py
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sec_13f.edgar_13f import SEC13FFetcher
from elite_managers import get_all_managers

HOLDINGS_FILE = Path(__file__).parent.parent / "production_data" / "holdings_snapshots.json"
CACHE_DIR = Path(__file__).parent.parent / "data" / "13f_cache"


def load_existing_holdings():
    """Load existing holdings_snapshots.json"""
    if HOLDINGS_FILE.exists():
        with open(HOLDINGS_FILE, 'r') as f:
            return json.load(f)
    return {}


def get_ciks_in_holdings(holdings: dict) -> set:
    """Extract all CIKs present in current holdings"""
    ciks = set()
    for ticker, data in holdings.items():
        current = data.get('holdings', {}).get('current', {})
        ciks.update(current.keys())
    return ciks


def fetch_manager_filings(cik: str, fetcher: SEC13FFetcher, quarters: int = 2):
    """Fetch current and prior quarter filings for a manager"""
    filings = fetcher.get_recent_filings(cik, count=quarters)
    if not filings:
        return None, None, None

    current_filing = filings[0]
    prior_filing = filings[1] if len(filings) > 1 else None

    # Parse holdings
    current_holdings = fetcher.parse_holdings(current_filing)
    prior_holdings = fetcher.parse_holdings(prior_filing) if prior_filing else []

    return current_filing, current_holdings, prior_holdings


def update_holdings_snapshots(holdings: dict, cik: str, filing, current_holdings, prior_holdings):
    """Update holdings_snapshots with new manager data"""
    cik_padded = cik.zfill(10)
    quarter_end = filing.report_date.isoformat()

    # Build prior lookup if available
    prior_by_ticker = {}
    for h in prior_holdings:
        if h.ticker:
            prior_by_ticker[h.ticker] = h

    updated_tickers = 0

    for h in current_holdings:
        if not h.ticker:
            continue  # Skip positions without tickers

        ticker = h.ticker.upper()

        # Initialize ticker entry if needed
        if ticker not in holdings:
            holdings[ticker] = {
                "market_cap_usd": 0,
                "holdings": {"current": {}, "prior": {}}
            }

        # Add current holding
        holdings[ticker]["holdings"]["current"][cik_padded] = {
            "quarter_end": quarter_end,
            "state": "KNOWN",
            "shares": h.shares,
            "value_kusd": h.value,  # Already in thousands from 13F
            "put_call": h.put_call or ""
        }

        # Add prior holding if available
        prior_h = prior_by_ticker.get(ticker)
        if prior_h:
            holdings[ticker]["holdings"]["prior"][cik_padded] = {
                "quarter_end": prior_h.report_date.isoformat() if hasattr(prior_h, 'report_date') else quarter_end,
                "state": "KNOWN",
                "shares": prior_h.shares,
                "value_kusd": prior_h.value,
                "put_call": prior_h.put_call or ""
            }

        updated_tickers += 1

    return updated_tickers


def main():
    print("=" * 60)
    print("13F Filings Fetcher - Updating Missing Managers")
    print("=" * 60)
    print()

    # Load manager registry
    managers = get_all_managers()
    print(f"Total managers in registry: {len(managers)}")

    # Load existing holdings
    holdings = load_existing_holdings()
    print(f"Existing tickers in holdings: {len(holdings)}")

    # Find missing managers
    existing_ciks = get_ciks_in_holdings(holdings)
    registry_ciks = {m['cik'].zfill(10) for m in managers}
    missing_ciks = registry_ciks - existing_ciks

    print(f"Managers with filings: {len(registry_ciks - missing_ciks)}")
    print(f"Missing managers: {len(missing_ciks)}")
    print()

    if not missing_ciks:
        print("All managers have filings. Nothing to fetch.")
        return

    # Initialize fetcher
    fetcher = SEC13FFetcher(cache_dir=str(CACHE_DIR))

    # Fetch missing managers
    success_count = 0
    error_count = 0

    for i, m in enumerate(managers):
        cik_padded = m['cik'].zfill(10)
        if cik_padded not in missing_ciks:
            continue

        name = m['name']
        tier = "Elite" if m.get('tier') == 1 else "Cond"

        print(f"[{success_count + error_count + 1}/{len(missing_ciks)}] Fetching {name} ({cik_padded}) [{tier}]...")

        try:
            filing, current_holdings, prior_holdings = fetch_manager_filings(m['cik'], fetcher)

            if filing is None:
                print(f"  ❌ No filings found")
                error_count += 1
                continue

            ticker_count = update_holdings_snapshots(holdings, m['cik'], filing, current_holdings, prior_holdings)
            print(f"  ✓ Q{filing.report_date.month // 4 + 1} {filing.report_date.year}: {len(current_holdings)} holdings, {ticker_count} tickers updated")
            success_count += 1

            # Rate limit: SEC requires max 10 requests/second
            time.sleep(0.2)

        except Exception as e:
            print(f"  ❌ Error: {e}")
            error_count += 1

    print()
    print("=" * 60)
    print(f"Fetch complete: {success_count} success, {error_count} errors")
    print()

    # Save updated holdings
    print(f"Saving to {HOLDINGS_FILE}...")
    with open(HOLDINGS_FILE, 'w') as f:
        json.dump(holdings, f, indent=2)

    # Verify
    new_ciks = get_ciks_in_holdings(holdings)
    print(f"Total manager CIKs in holdings: {len(new_ciks)}")
    print(f"Total tickers covered: {len(holdings)}")
    print()

    # Coverage summary
    covered = registry_ciks & new_ciks
    still_missing = registry_ciks - new_ciks
    print(f"Registry coverage: {len(covered)}/{len(registry_ciks)} ({100*len(covered)/len(registry_ciks):.1f}%)")

    if still_missing:
        print(f"\nStill missing ({len(still_missing)}):")
        for cik in sorted(still_missing):
            mgr = next((m for m in managers if m['cik'].zfill(10) == cik), None)
            if mgr:
                print(f"  {cik}: {mgr['name']}")


if __name__ == '__main__':
    main()
