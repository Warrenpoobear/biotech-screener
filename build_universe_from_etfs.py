#!/usr/bin/env python3
"""
Build Universe from Biotech ETF Constituents

Fetches holdings from major biotech ETFs/indices and creates a combined universe.
Default ETFs: XBI, IBB, NBI (NASDAQ Biotech Index)

Usage:
    $env:MD_AUTH_TOKEN="your-token"
    python build_universe_from_etfs.py
"""

import os
import sys
import json
import csv
from datetime import date
from typing import List, Dict, Set

# Check token
if not os.environ.get('MD_AUTH_TOKEN'):
    print("Error: MD_AUTH_TOKEN not set")
    print("  $env:MD_AUTH_TOKEN='your-token'")
    sys.exit(1)

try:
    import morningstar_data as md
    from morningstar_data.direct.data_type import Frequency
except ImportError:
    print("Error: morningstar-data package not installed")
    sys.exit(1)

# ETF SecIds - biotech ETFs for universe construction
# You may need to look these up using md.direct.investments()
ETF_INFO = {
    'XBI': {'name': 'SPDR S&P Biotech ETF', 'secid': None},
    'IBB': {'name': 'iShares Biotechnology ETF', 'secid': None},
    'NBI': {'name': 'NASDAQ Biotechnology Index', 'secid': None},
}


def find_etf_secids() -> Dict[str, str]:
    """Look up SecIds for ETFs."""
    print("\n[1] Looking up ETF SecIds...")

    secid_map = {}

    for ticker in ETF_INFO.keys():
        try:
            # Try ticker:US format
            results = md.direct.investments([f"{ticker}:US"])

            if results is not None and not results.empty:
                for idx, row in results.iterrows():
                    sec_id = str(row.get('SecId', ''))
                    name = str(row.get('Name', ''))
                    if sec_id:
                        secid_map[ticker] = sec_id
                        print(f"    {ticker}: {sec_id} ({name})")
                        break

            if ticker not in secid_map:
                print(f"    {ticker}: NOT FOUND")

        except Exception as e:
            print(f"    {ticker}: Error - {e}")

    return secid_map


def get_etf_holdings(secid: str, ticker: str) -> List[Dict]:
    """Get holdings for an ETF."""
    print(f"\n    Fetching holdings for {ticker} ({secid})...")

    holdings = []

    try:
        # Try get_holdings or similar method
        df = md.direct.get_holdings(
            investments=[secid],
            # holdings_view might be needed
        )

        if df is not None and not df.empty:
            print(f"      DataFrame columns: {list(df.columns)}")

            # Extract holding info
            for idx, row in df.iterrows():
                holding = {
                    'ticker': str(row.get('Ticker', row.get('HoldingTicker', ''))),
                    'name': str(row.get('Name', row.get('HoldingName', ''))),
                    'secid': str(row.get('SecId', row.get('HoldingSecId', ''))),
                    'weight': float(row.get('Weight', row.get('PercentAssets', 0))) if row.get('Weight') or row.get('PercentAssets') else 0,
                    'etf_source': ticker,
                }
                if holding['ticker'] or holding['secid']:
                    holdings.append(holding)

            print(f"      Found {len(holdings)} holdings")
        else:
            print(f"      No holdings data returned")

    except AttributeError:
        # Try alternative method
        print(f"      get_holdings not available, trying get_portfolio_holdings...")
        try:
            df = md.direct.get_portfolio_holdings(
                investments=[secid],
            )
            if df is not None and not df.empty:
                print(f"      DataFrame columns: {list(df.columns)}")
                for idx, row in df.iterrows():
                    holding = {
                        'ticker': str(row.get('Ticker', '')),
                        'name': str(row.get('Name', '')),
                        'secid': str(row.get('SecId', '')),
                        'weight': 0,
                        'etf_source': ticker,
                    }
                    if holding['ticker'] or holding['secid']:
                        holdings.append(holding)
                print(f"      Found {len(holdings)} holdings")
        except Exception as e2:
            print(f"      Alternative method also failed: {e2}")

    except Exception as e:
        print(f"      Error: {e}")

    return holdings


def main():
    print("="*60)
    print("BUILD UNIVERSE FROM BIOTECH ETF CONSTITUENTS")
    print("="*60)

    # Find ETF SecIds
    etf_secids = find_etf_secids()

    if not etf_secids:
        print("\nError: Could not find any ETF SecIds")
        print("\nLet's try a direct search for biotech ETFs...")

        # Try searching for biotech ETFs
        try:
            results = md.direct.investments("biotech ETF")
            if results is not None and not results.empty:
                print("\nFound ETFs matching 'biotech ETF':")
                print(results[['SecId', 'Ticker', 'Name']].head(10).to_string())
        except Exception as e:
            print(f"Search failed: {e}")

        sys.exit(1)

    # Get holdings for each ETF
    print("\n[2] Fetching ETF holdings...")

    all_holdings = []
    for ticker, secid in etf_secids.items():
        holdings = get_etf_holdings(secid, ticker)
        all_holdings.extend(holdings)

    if not all_holdings:
        print("\nWarning: No holdings retrieved via API")
        print("\nAlternative: Check if your Morningstar Direct subscription includes ETF holdings data.")
        print("You may need to use a different API method or export from the web interface.")

        # Create a sample universe with known biotech tickers
        print("\nCreating sample universe with common biotech tickers...")
        sample_tickers = [
            'VRTX', 'REGN', 'AMGN', 'GILD', 'BIIB', 'MRNA', 'ALNY', 'BMRN',
            'SGEN', 'INCY', 'EXEL', 'NBIX', 'UTHR', 'SRPT', 'IONS', 'RARE',
            'PCVX', 'ARGX', 'LEGN', 'KRYS', 'RCKT', 'IMVT', 'DNLI', 'BEAM',
            'CRSP', 'NTLA', 'EDIT', 'FATE', 'BLUE', 'KRTX', 'HZNP', 'SWTX'
        ]
        all_holdings = [{'ticker': t, 'name': '', 'secid': '', 'weight': 0, 'etf_source': 'manual'} for t in sample_tickers]

    # Deduplicate by ticker
    print("\n[3] Deduplicating holdings...")

    seen_tickers: Set[str] = set()
    unique_holdings = []

    for h in all_holdings:
        ticker = h['ticker'].upper().strip()
        if ticker and ticker not in seen_tickers and len(ticker) <= 5:  # Filter out non-stock entries
            seen_tickers.add(ticker)
            unique_holdings.append(h)

    print(f"    Unique tickers: {len(unique_holdings)}")

    # Save as JSON
    output_json = 'data/universe_biotech_etfs.json'
    os.makedirs('data', exist_ok=True)

    universe_data = {
        'metadata': {
            'created_at': date.today().isoformat(),
            'source_etfs': list(etf_secids.keys()),
            'ticker_count': len(unique_holdings),
        },
        'securities': unique_holdings,
    }

    with open(output_json, 'w') as f:
        json.dump(universe_data, f, indent=2)

    print(f"\n    Saved JSON: {output_json}")

    # Save as CSV (for build_returns_db.py)
    output_csv = 'data/universe_biotech_etfs.csv'

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ticker', 'secid', 'name', 'etf_source'])
        writer.writeheader()
        for h in unique_holdings:
            writer.writerow({
                'ticker': h['ticker'],
                'secid': h.get('secid', ''),
                'name': h.get('name', ''),
                'etf_source': h.get('etf_source', ''),
            })

    print(f"    Saved CSV: {output_csv}")

    # Summary
    print("\n" + "="*60)
    print("UNIVERSE BUILD COMPLETE")
    print("="*60)
    print(f"\nTickers: {len(unique_holdings)}")
    print(f"\nSample tickers: {[h['ticker'] for h in unique_holdings[:10]]}")

    print("\nNext steps:")
    print(f"  python build_returns_db.py --universe {output_csv} --start-date 2020-01-01")


if __name__ == "__main__":
    main()
