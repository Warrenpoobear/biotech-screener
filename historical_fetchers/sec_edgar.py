#!/usr/bin/env python3
"""
SEC EDGAR Historical Financials Fetcher

Fetches 10-Q and 10-K filings from SEC EDGAR to reconstruct
point-in-time financial snapshots.

Usage:
    python sec_edgar.py --ticker VRTX --as-of 2023-01-15
    python sec_edgar.py --tickers VRTX,REGN,ALNY --as-of 2023-01-15 --output financials.json
"""

import argparse
import json
import os
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import urllib.request
import urllib.error

# SEC EDGAR API base URLs
SEC_DATA_URL = "https://data.sec.gov"
SEC_WWW_URL = "https://www.sec.gov"

# Required headers for SEC EDGAR API
SEC_HEADERS = {
    'User-Agent': 'BiotechScreener/1.0 (contact@example.com)',
    'Accept': 'application/json'
}

# Cache directory
CACHE_DIR = Path("data/cache/sec_edgar")


def get_cik_for_ticker(ticker: str) -> Optional[str]:
    """
    Get the CIK (Central Index Key) for a ticker symbol.

    The CIK is required to query SEC EDGAR for company filings.
    Uses the SEC's company tickers JSON file for mapping.
    """
    cache_file = CACHE_DIR / "ticker_to_cik.json"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Check cache first
    if cache_file.exists():
        with open(cache_file, 'r') as f:
            cache = json.load(f)
            if ticker.upper() in cache:
                return cache[ticker.upper()]

    # Fetch the SEC company tickers mapping
    tickers_url = f"{SEC_WWW_URL}/files/company_tickers.json"

    try:
        req = urllib.request.Request(tickers_url, headers=SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode())

            # Build ticker -> CIK mapping
            ticker_map = {}
            for entry in data.values():
                t = entry.get('ticker', '').upper()
                cik = str(entry.get('cik_str', '')).zfill(10)
                if t:
                    ticker_map[t] = cik

            # Save to cache
            with open(cache_file, 'w') as f:
                json.dump(ticker_map, f, indent=2)

            return ticker_map.get(ticker.upper())

    except urllib.error.HTTPError as e:
        print(f"  HTTP Error {e.code} fetching CIK mapping")
        return None
    except Exception as e:
        print(f"  Error fetching CIK for {ticker}: {e}")
        return None


def get_company_filings(cik: str) -> Dict[str, Any]:
    """
    Get all filings for a company by CIK.
    """
    url = f"{SEC_DATA_URL}/submissions/CIK{cik}.json"

    req = urllib.request.Request(url, headers=SEC_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode())


def find_filing_before_date(filings: Dict, form_types: List[str],
                            as_of_date: str) -> Optional[Dict]:
    """
    Find the most recent filing of given type(s) before the as_of_date.

    Returns filing metadata including accession number for fetching details.
    """
    target_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()

    recent_filings = filings.get('filings', {}).get('recent', {})
    forms = recent_filings.get('form', [])
    filing_dates = recent_filings.get('filingDate', [])
    accession_numbers = recent_filings.get('accessionNumber', [])
    primary_docs = recent_filings.get('primaryDocument', [])

    for i, (form, filing_date, accession, doc) in enumerate(
        zip(forms, filing_dates, accession_numbers, primary_docs)
    ):
        if form not in form_types:
            continue

        filed_date = datetime.strptime(filing_date, "%Y-%m-%d").date()
        if filed_date <= target_date:
            return {
                'form': form,
                'filing_date': filing_date,
                'accession_number': accession,
                'primary_document': doc
            }

    return None


def get_company_facts(cik: str) -> Optional[Dict]:
    """
    Get company facts (XBRL data) from SEC EDGAR.

    This includes standardized financial data like:
    - Assets, Liabilities
    - Cash and equivalents
    - Revenue, Net Income
    """
    url = f"{SEC_DATA_URL}/api/xbrl/companyfacts/CIK{cik}.json"

    try:
        req = urllib.request.Request(url, headers=SEC_HEADERS)
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def extract_financial_value(facts: Dict, concept: str, as_of_date: str,
                           namespace: str = 'us-gaap') -> Optional[float]:
    """
    Extract a financial value from company facts as of a specific date.

    Returns the most recent value filed before as_of_date.
    """
    target_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()

    ns_facts = facts.get('facts', {}).get(namespace, {})
    concept_data = ns_facts.get(concept, {})
    units = concept_data.get('units', {})

    # Try USD first, then pure numbers
    values = units.get('USD', []) or units.get('pure', []) or units.get('shares', [])

    best_value = None
    best_date = None

    for entry in values:
        # Get the filing date or end date
        filed = entry.get('filed')
        end = entry.get('end')

        if not filed:
            continue

        filed_date = datetime.strptime(filed, "%Y-%m-%d").date()

        # Must be filed before our target date
        if filed_date > target_date:
            continue

        # Prefer more recent filings
        if best_date is None or filed_date > best_date:
            best_value = entry.get('val')
            best_date = filed_date

    return best_value


def get_historical_financials(ticker: str, as_of_date: str) -> Optional[Dict]:
    """
    Get historical financial data for a ticker as of a specific date.

    Returns cash, debt, and other key metrics from the most recent
    10-Q or 10-K filed before the as_of_date.
    """
    cik = get_cik_for_ticker(ticker)
    if not cik:
        return None

    # Get company facts (XBRL data)
    facts = get_company_facts(cik)
    if not facts:
        return {
            'ticker': ticker,
            'as_of_date': as_of_date,
            'error': 'No XBRL data available'
        }

    # Extract key financial metrics
    # Cash and cash equivalents
    cash = extract_financial_value(facts, 'CashAndCashEquivalentsAtCarryingValue', as_of_date)
    if cash is None:
        cash = extract_financial_value(facts, 'Cash', as_of_date)

    # Marketable securities (add to cash)
    securities = extract_financial_value(facts, 'MarketableSecuritiesCurrent', as_of_date)
    if securities is None:
        securities = extract_financial_value(facts, 'AvailableForSaleSecuritiesCurrent', as_of_date)

    total_cash = (cash or 0) + (securities or 0)

    # Total debt
    long_term_debt = extract_financial_value(facts, 'LongTermDebt', as_of_date) or 0
    short_term_debt = extract_financial_value(facts, 'ShortTermBorrowings', as_of_date) or 0
    total_debt = long_term_debt + short_term_debt

    # Total assets and liabilities
    total_assets = extract_financial_value(facts, 'Assets', as_of_date)
    total_liabilities = extract_financial_value(facts, 'Liabilities', as_of_date)

    # Revenue and net income (for burn rate estimation)
    revenue = extract_financial_value(facts, 'Revenues', as_of_date)
    if revenue is None:
        revenue = extract_financial_value(facts, 'RevenueFromContractWithCustomerExcludingAssessedTax', as_of_date)

    net_income = extract_financial_value(facts, 'NetIncomeLoss', as_of_date)

    # R&D expenses
    rd_expense = extract_financial_value(facts, 'ResearchAndDevelopmentExpense', as_of_date)

    # Operating expenses (for burn rate)
    operating_expenses = extract_financial_value(facts, 'OperatingExpenses', as_of_date)

    # Estimate quarterly burn rate
    quarterly_burn = None
    if net_income is not None and net_income < 0:
        # If losing money, burn rate is the loss
        quarterly_burn = abs(net_income) / 4  # Annualized to quarterly
    elif operating_expenses:
        quarterly_burn = operating_expenses / 4

    # Estimate runway
    runway_months = None
    if total_cash and quarterly_burn and quarterly_burn > 0:
        runway_months = (total_cash / quarterly_burn) * 3  # Convert quarters to months

    return {
        'ticker': ticker,
        'as_of_date': as_of_date,
        'cik': cik,
        'cash': total_cash if total_cash else None,
        'debt': total_debt if total_debt else None,
        'total_assets': total_assets,
        'total_liabilities': total_liabilities,
        'revenue': revenue,
        'net_income': net_income,
        'rd_expense': rd_expense,
        'quarterly_burn': quarterly_burn,
        'runway_months': runway_months,
        'data_source': 'SEC_EDGAR_XBRL'
    }


def fetch_batch(tickers: List[str], as_of_date: str,
                delay: float = 0.1) -> Dict[str, Dict]:
    """
    Fetch historical financials for multiple tickers.

    Respects SEC rate limits (10 requests/second).
    """
    results = {}
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        print(f"  [{i}/{total}] Fetching {ticker}...", end=" ")
        try:
            data = get_historical_financials(ticker, as_of_date)
            if data:
                results[ticker] = data
                cash_str = f"${data.get('cash', 0)/1e6:.0f}M" if data.get('cash') else "N/A"
                print(f"Cash: {cash_str}")
            else:
                print("No data")
        except Exception as e:
            print(f"Error: {e}")
            results[ticker] = {'ticker': ticker, 'error': str(e)}

        # Rate limiting
        time.sleep(delay)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fetch historical financials from SEC EDGAR"
    )
    parser.add_argument('--ticker', type=str, help='Single ticker to fetch')
    parser.add_argument('--tickers', type=str, help='Comma-separated list of tickers')
    parser.add_argument('--tickers-file', type=str, help='File with tickers (one per line)')
    parser.add_argument('--as-of', type=str, required=True,
                        help='Point-in-time date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='Output JSON file')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between requests (default: 0.1s)')

    args = parser.parse_args()

    # Collect tickers
    tickers = []
    if args.ticker:
        tickers = [args.ticker]
    elif args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    elif args.tickers_file:
        with open(args.tickers_file, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
    else:
        parser.error("Specify --ticker, --tickers, or --tickers-file")

    print(f"Fetching historical financials for {len(tickers)} tickers as of {args.as_of}")

    results = fetch_batch(tickers, args.as_of, delay=args.delay)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'as_of_date': args.as_of,
                'fetched_at': datetime.now().isoformat(),
                'count': len(results),
                'tickers': results
            }, f, indent=2)
        print(f"\nSaved to {args.output}")
    else:
        print(json.dumps(results, indent=2))

    # Summary
    with_cash = sum(1 for r in results.values() if r.get('cash'))
    print(f"\nSummary: {with_cash}/{len(results)} tickers with cash data")


if __name__ == "__main__":
    main()
