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
from datetime import date
from typing import Dict, Optional
import argparse


def get_cik_from_ticker(ticker: str) -> Optional[str]:
    """Get CIK (Central Index Key) from ticker using SEC API"""
    
    url = "https://www.sec.gov/files/company_tickers.json"
    headers = {'User-Agent': 'WakeRobinCapital research@wakerobincapital.com'}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            for entry in data.values():
                if entry.get('ticker', '').upper() == ticker.upper():
                    cik = str(entry.get('cik_str')).zfill(10)
                    return cik
        
        return None
    
    except Exception as e:
        return None


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
            }

            financial_data = {"ticker": ticker, "cik": cik}

            # Extract most recent values (simple metrics)
            for gaap_key, friendly_name in metrics.items():
                if gaap_key in facts:
                    units = facts[gaap_key].get('units', {})

                    if 'USD' in units:
                        values = units['USD']
                        recent = sorted(values, key=lambda x: x.get('end', ''), reverse=True)

                        if recent:
                            financial_data[friendly_name] = recent[0].get('val')
                            financial_data[f"{friendly_name}_date"] = recent[0].get('end')

            # Extract metrics with fallback (use most recent across all tags)
            for friendly_name, tag_list in metrics_with_fallback.items():
                best_val, best_date = None, None
                for gaap_key in tag_list:
                    if gaap_key in facts:
                        units = facts[gaap_key].get('units', {})
                        if 'USD' in units:
                            values = units['USD']
                            recent = sorted(values, key=lambda x: x.get('end', ''), reverse=True)
                            if recent:
                                val, dt = recent[0].get('val'), recent[0].get('end')
                                if best_date is None or (dt and dt > best_date):
                                    best_val, best_date = val, dt
                if best_val is not None:
                    financial_data[friendly_name] = best_val
                    financial_data[f"{friendly_name}_date"] = best_date
            
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
    
    print(f"\nUniverse: {len(tickers)} tickers")
    print(f"Output: {output_file}")
    print(f"Estimated time: {len(tickers) * 0.2 / 60:.1f} minutes")
    
    # Collect
    all_data = []
    stats = {'total': len(tickers), 'successful': 0, 'no_cik': 0, 'no_data': 0}
    
    print(f"\n{'='*80}")
    print("COLLECTING FINANCIAL DATA")
    print(f"{'='*80}\n")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:3d}/{len(tickers)}] {ticker:6s}", end=" ", flush=True)
        
        # Get CIK
        cik = get_cik_from_ticker(ticker)
        
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
