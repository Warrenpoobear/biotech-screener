#!/usr/bin/env python3
"""
collect_financial_data.py - Collect Financial Data from SEC EDGAR

Fetches 10-K/10-Q filings and extracts key financial metrics.

Usage:
    python collect_financial_data.py --universe production_data/universe.json
"""

import json
import requests
import time
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, Optional
import argparse

# Maximum age in days for financial data to be considered valid
MAX_DATA_AGE_DAYS = 365


def is_data_fresh(date_str: str, max_age_days: int = MAX_DATA_AGE_DAYS) -> bool:
    """Check if data date is within acceptable age range."""
    if not date_str:
        return False
    try:
        data_date = datetime.fromisoformat(date_str)
        age_days = (datetime.now() - data_date).days
        return age_days <= max_age_days
    except (ValueError, TypeError):
        return False


_CIK_CACHE: Optional[Dict[str, str]] = None


def _load_cik_cache() -> Dict[str, str]:
    """Load and cache the SEC ticker→CIK mapping (single request)."""
    global _CIK_CACHE
    if _CIK_CACHE is not None:
        return _CIK_CACHE

    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {'User-Agent': 'WakeRobinCapital research@wakerobincapital.com'}

    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            _CIK_CACHE = {
                entry['ticker'].upper(): str(entry.get('cik_str')).zfill(10)
                for entry in data.values()
                if entry.get('ticker')
            }
        else:
            _CIK_CACHE = {}
    except Exception:
        _CIK_CACHE = {}

    return _CIK_CACHE


def get_cik_from_ticker(ticker: str) -> Optional[str]:
    """Get CIK (Central Index Key) from ticker using cached SEC mapping."""
    cache = _load_cik_cache()
    return cache.get(ticker.upper())


def get_company_facts(cik: str, ticker: str) -> Optional[Dict]:
    """Get company facts from SEC EDGAR API"""
    
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    headers = {
        'User-Agent': 'WakeRobinCapital research@wakerobincapital.com',
        'Accept-Encoding': 'gzip, deflate'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            facts = data.get('facts', {}).get('us-gaap', {})
            
            # Key metrics to extract (single XBRL tag -> friendly name)
            metrics = {
                'Assets': 'Assets',
                'AssetsCurrent': 'CurrentAssets',
                'Liabilities': 'Liabilities',
                'LiabilitiesCurrent': 'CurrentLiabilities',
                'StockholdersEquity': 'ShareholdersEquity',
                'CashAndCashEquivalentsAtCarryingValue': 'Cash',
                'MarketableSecuritiesCurrent': 'MarketableSecurities',
                'ShortTermInvestments': 'ShortTermInvestments',
                'AvailableForSaleSecuritiesCurrent': 'AvailableForSaleSecurities',
                'CostOfRevenue': 'COGS',
                'ResearchAndDevelopmentExpense': 'R&D',
                'NetIncomeLoss': 'NetIncome',
                'LongTermDebt': 'LongTermDebt',
                'LongTermDebtCurrent': 'LongTermDebtCurrent',
                'ConvertibleNotesPayable': 'ConvertibleDebt',
            }

            # Metrics with fallback tags (try in order, use first found)
            metrics_with_fallback = {
                'Revenue': [
                    'RevenueFromContractWithCustomerExcludingAssessedTax',  # ASC 606 (2018+)
                    'Revenues',  # Legacy
                    'SalesRevenueNet',  # Alternative
                    'SalesRevenueGoodsNet',  # Product-specific
                ],
                'CFO': [
                    'NetCashProvidedByUsedInOperatingActivities',  # Standard
                    'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',  # Alternative
                    'CashFlowsFromUsedInOperatingActivities',  # IFRS-style
                ],
                'OperatingExpenses': [
                    'OperatingExpenses',  # Standard
                    'CostsAndExpenses',  # Alternative (includes COGS)
                    'OperatingCostsAndExpenses',  # Alternative
                ],
                'InterestExpense': [
                    'InterestExpense',  # Standard
                    'InterestAndDebtExpense',  # Alternative
                    'InterestExpenseDebt',  # Specific to debt
                ],
            }

            financial_data = {"ticker": ticker, "cik": cik}

            # Extract most recent values (simple metrics) with staleness filter
            for gaap_key, friendly_name in metrics.items():
                if gaap_key in facts:
                    units = facts[gaap_key].get('units', {})

                    if 'USD' in units:
                        values = units['USD']
                        # Sort by date and find most recent FRESH value
                        recent = sorted(values, key=lambda x: x.get('end', ''), reverse=True)

                        for entry in recent:
                            entry_date = entry.get('end')
                            if is_data_fresh(entry_date):
                                financial_data[friendly_name] = entry.get('val')
                                financial_data[f"{friendly_name}_date"] = entry_date
                                break
                        # If no fresh data found, don't include (leave as None)

            # Extract metrics with fallback (use most recent FRESH value across all tags)
            for friendly_name, tag_list in metrics_with_fallback.items():
                best_val, best_date = None, None
                for gaap_key in tag_list:
                    if gaap_key in facts:
                        units = facts[gaap_key].get('units', {})
                        if 'USD' in units:
                            values = units['USD']
                            recent = sorted(values, key=lambda x: x.get('end', ''), reverse=True)
                            for entry in recent:
                                val, dt = entry.get('val'), entry.get('end')
                                if is_data_fresh(dt):
                                    if best_date is None or (dt and dt > best_date):
                                        best_val, best_date = val, dt
                                    break  # Found fresh value for this tag
                if best_val is not None:
                    financial_data[friendly_name] = best_val
                    financial_data[f"{friendly_name}_date"] = best_date
            
            # Aggregate TotalDebt from components if not already present
            if financial_data.get('LongTermDebt') or financial_data.get('LongTermDebtCurrent') or financial_data.get('ConvertibleDebt'):
                total_debt = 0
                debt_components = []

                if financial_data.get('LongTermDebt'):
                    total_debt += financial_data['LongTermDebt']
                    debt_components.append('LongTermDebt')
                if financial_data.get('LongTermDebtCurrent'):
                    total_debt += financial_data['LongTermDebtCurrent']
                    debt_components.append('LongTermDebtCurrent')
                if financial_data.get('ConvertibleDebt'):
                    total_debt += financial_data['ConvertibleDebt']
                    debt_components.append('ConvertibleDebt')

                if total_debt > 0:
                    financial_data['TotalDebt'] = total_debt
                    financial_data['TotalDebt_components'] = debt_components

            # Aggregate CashAndSecurities (Cash + MarketableSecurities + ShortTermInvestments)
            cash = financial_data.get('Cash', 0) or 0
            mkt_sec = financial_data.get('MarketableSecurities', 0) or 0
            st_inv = financial_data.get('ShortTermInvestments', 0) or 0
            avail = financial_data.get('AvailableForSaleSecurities', 0) or 0
            total_liquid = cash + mkt_sec + st_inv + avail
            if total_liquid > 0:
                financial_data['CashAndSecurities'] = total_liquid
                # Use most recent date from components
                dates = [financial_data.get('Cash_date'), financial_data.get('MarketableSecurities_date'),
                         financial_data.get('ShortTermInvestments_date')]
                dates = [d for d in dates if d]
                if dates:
                    financial_data['CashAndSecurities_date'] = max(dates)

            financial_data['collected_at'] = date.today().isoformat()
            return financial_data

        return None

    except Exception as e:
        return None


def collect_all_financial_data(universe_file: Path, output_file: Path):
    """Collect financial data for all tickers"""
    
    print("="*80)
    print("FINANCIAL DATA COLLECTION (SEC EDGAR)")
    print("="*80)
    print(f"Date: {date.today()}")
    print("\n⚠️  Note: SEC rate limits to 10 requests/second")
    print("         This will take 30-60 minutes for 300+ tickers")
    
    # Load universe
    with open(universe_file) as f:
        universe = json.load(f)
    
    tickers = [s['ticker'] for s in universe if s.get('ticker') and s['ticker'] != '_XBI_BENCHMARK_']
    # Build CIK lookup from universe.json (avoids SEC API calls for known CIKs)
    universe_ciks = {}
    for s in universe:
        t = s.get('ticker', '')
        c = s.get('cik', '')
        if t and c and c not in ('', 'None', None):
            # Normalize to 10-digit zero-padded
            universe_ciks[t] = str(c).lstrip('0').zfill(10) if c else ''

    print(f"\nUniverse: {len(tickers)} tickers ({len(universe_ciks)} with CIK in universe.json)")
    print(f"Output: {output_file}")
    print(f"Estimated time: {len(tickers) * 0.2 / 60:.1f} minutes")

    # Pre-load SEC CIK cache (single request)
    print("Loading SEC ticker→CIK mapping...", flush=True)
    _load_cik_cache()
    print(f"  Loaded {len(_CIK_CACHE or {})} ticker→CIK mappings from SEC")

    # Collect
    all_data = []
    stats = {'total': len(tickers), 'successful': 0, 'no_cik': 0, 'no_data': 0}

    print(f"\n{'='*80}")
    print("COLLECTING FINANCIAL DATA")
    print(f"{'='*80}\n")

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:3d}/{len(tickers)}] {ticker:6s}", end=" ", flush=True)

        # Get CIK: prefer universe.json, then SEC cache
        cik = universe_ciks.get(ticker) or get_cik_from_ticker(ticker)
        
        if not cik:
            stats['no_cik'] += 1
            print("❌ No CIK")
            time.sleep(0.1)
            continue
        
        # Get financial data
        data = get_company_facts(cik, ticker)
        
        if data and len(data.keys()) > 3:
            all_data.append(data)
            stats['successful'] += 1
            
            cash = data.get('Cash', 0)
            revenue = data.get('Revenue', 0)
            
            cash_str = f"${cash/1e9:.1f}B" if cash and cash > 1e9 else f"${cash/1e6:.0f}M" if cash else "N/A"
            rev_str = f"${revenue/1e9:.1f}B" if revenue and revenue > 1e9 else f"${revenue/1e6:.0f}M" if revenue else "N/A"
            
            print(f"✅ Cash: {cash_str:>8s}, Rev: {rev_str:>8s}")
        else:
            stats['no_data'] += 1
            print("⚠️  No filings")
        
        # SEC rate limit: 10 req/sec
        time.sleep(0.15)
        
        if i % 50 == 0:
            print(f"\n  Progress: {i}/{len(tickers)} ({i/len(tickers)*100:.1f}%)")
            print(f"  Success: {stats['successful']/i*100:.1f}%")
            print(f"  Time remaining: {(len(tickers)-i)*0.15/60:.1f} minutes\n")
    
    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_data, f, indent=2)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total: {stats['total']}")
    print(f"Successful: {stats['successful']}")
    print(f"No CIK: {stats['no_cik']}")
    print(f"No filings: {stats['no_data']}")
    print(f"Coverage: {stats['successful'] / stats['total'] * 100:.1f}%")
    print(f"✅ Saved to: {output_file}")

    # Data quality summary
    print(f"\n{'='*80}")
    print("DATA QUALITY SUMMARY")
    print(f"{'='*80}")

    # Count key field coverage
    has_cash = sum(1 for d in all_data if d.get('Cash'))
    has_revenue = sum(1 for d in all_data if d.get('Revenue'))
    has_rd = sum(1 for d in all_data if d.get('R&D'))
    has_assets = sum(1 for d in all_data if d.get('Assets'))
    has_cfo = sum(1 for d in all_data if d.get('CFO'))
    has_total_debt = sum(1 for d in all_data if d.get('TotalDebt'))
    has_long_term_debt = sum(1 for d in all_data if d.get('LongTermDebt'))

    collected = len(all_data)
    if collected > 0:
        print(f"Cash:           {has_cash:4d}/{collected} ({has_cash/collected*100:5.1f}%)")
        print(f"Revenue:        {has_revenue:4d}/{collected} ({has_revenue/collected*100:5.1f}%)")
        print(f"R&D:            {has_rd:4d}/{collected} ({has_rd/collected*100:5.1f}%)")
        print(f"Assets:         {has_assets:4d}/{collected} ({has_assets/collected*100:5.1f}%)")
        print(f"CFO:            {has_cfo:4d}/{collected} ({has_cfo/collected*100:5.1f}%)")
        print(f"LongTermDebt:   {has_long_term_debt:4d}/{collected} ({has_long_term_debt/collected*100:5.1f}%)")
        print(f"TotalDebt:      {has_total_debt:4d}/{collected} ({has_total_debt/collected*100:5.1f}%)")

        # Data freshness check
        fresh_count = sum(1 for d in all_data if d.get('Cash_date') and is_data_fresh(d['Cash_date'], 180))
        print(f"\nCash data <6mo: {fresh_count:4d}/{collected} ({fresh_count/collected*100:5.1f}%)")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Collect financial data from SEC EDGAR")
    parser.add_argument('--universe', type=Path, default=Path('production_data/universe.json'))
    parser.add_argument('--output', type=Path, default=Path('production_data/financial_data.json'))
    args = parser.parse_args()
    
    if not args.universe.exists():
        print(f"❌ Universe file not found: {args.universe}")
        return 1
    
    try:
        collect_all_financial_data(args.universe, args.output)
        return 0
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
