#!/usr/bin/env python3
"""collect_all_universe_data.py - Real data collection"""
import json, time, sys
from datetime import datetime

def collect_financial_data(ticker):
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'cash': info.get('totalCash'),
            'debt': info.get('totalDebt'),
            'net_debt': (info.get('totalDebt', 0) or 0) - (info.get('totalCash', 0) or 0),
            'revenue_ttm': info.get('totalRevenue'),
            'assets': info.get('totalAssets'),
            'liabilities': info.get('totalLiabilities'),
            'equity': info.get('totalStockholderEquity'),
            'currency': info.get('currency', 'USD'),
            'cik': info.get('cik'),
        }
    except Exception as e:
        print(f"    ⚠️ Financial: {e}")
        return None

def collect_market_data(ticker):
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'price': info.get('currentPrice') or info.get('regularMarketPrice'),
            'market_cap': info.get('marketCap'),
            'shares_outstanding': info.get('sharesOutstanding'),
            'volume_avg_30d': info.get('averageVolume'),
            '52_week_high': info.get('fiftyTwoWeekHigh'),
            '52_week_low': info.get('fiftyTwoWeekLow'),
            'pe_ratio': info.get('trailingPE'),
            'company_name': info.get('longName'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
        }
    except Exception as e:
        print(f"    ⚠️ Market: {e}")
        return None

def collect_clinical_data(ticker, company_name=None):
    try:
        import requests
        search_terms = [ticker]
        if company_name and ' ' in company_name:
            search_terms.append(company_name.split()[0])
        
        all_trials = []
        for term in search_terms[:2]:
            url = "https://clinicaltrials.gov/api/v2/studies"
            params = {'query.term': term, 'format': 'json', 'pageSize': 20}
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                studies = response.json().get('studies', [])
                for study in studies:
                    protocol = study.get('protocolSection', {})
                    trial = {
                        'nct_id': protocol.get('identificationModule', {}).get('nctId'),
                        'title': protocol.get('identificationModule', {}).get('briefTitle'),
                        'status': protocol.get('statusModule', {}).get('overallStatus'),
                        'phase': (protocol.get('designModule', {}).get('phases', ['N/A']) or ['N/A'])[0],
                    }
                    if trial['nct_id'] and not any(t.get('nct_id') == trial['nct_id'] for t in all_trials):
                        all_trials.append(trial)
            time.sleep(0.5)
        
        active = sum(1 for t in all_trials if 'ACTIVE' in t.get('status', '').upper() or 'RECRUITING' in t.get('status', '').upper())
        completed = sum(1 for t in all_trials if 'COMPLETED' in t.get('status', '').upper())
        
        return {
            'total_trials': len(all_trials),
            'active_trials': active,
            'completed_trials': completed,
            'lead_stage': 'unknown',
            'by_phase': {},
            'conditions': [],
            'top_trials': all_trials[:5],
        }
    except Exception as e:
        print(f"    ⚠️ Clinical: {e}")
        return None

def collect_single_stock(security):
    ticker = security.get('ticker', 'UNKNOWN')
    print(f"  [{ticker}]", end=" ")
    
    if security.get('financial_data') and security.get('clinical_data'):
        print("Already complete")
        return security
    
    company_name = None
    if security.get('market_data'):
        company_name = security['market_data'].get('company_name')
    
    if not security.get('financial_data'):
        fin = collect_financial_data(ticker)
        if fin: security['financial_data'] = fin
        time.sleep(0.5)
    
    if not security.get('market_data'):
        mkt = collect_market_data(ticker)
        if mkt:
            security['market_data'] = mkt
            company_name = mkt.get('company_name')
        time.sleep(0.5)
    
    if not security.get('clinical_data'):
        clin = collect_clinical_data(ticker, company_name)
        if clin:
            security['clinical_data'] = clin
            print(f"✓ ({clin['total_trials']} trials)")
        else:
            print("✓")
    else:
        print("✓")
    
    return security

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="production_data/universe.json")
    parser.add_argument("--output", default="production_data/universe.json")
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()
    
    print("="*80)
    print("BIOTECH UNIVERSE DATA COLLECTION")
    print("="*80)
    print()
    
    with open(args.universe) as f:
        universe = json.load(f)
    
    total = len(universe)
    print(f"Total: {total} stocks")
    print()
    
    start_time = datetime.now()
    
    for i, security in enumerate(universe):
        print(f"[{i+1}/{total}]", end=" ")
        universe[i] = collect_single_stock(security)
        
        if (i + 1) % args.save_every == 0:
            with open(args.output, 'w') as f:
                json.dump(universe, f, indent=2)
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (total - i - 1) / rate if rate > 0 else 0
            print(f"\n  💾 Saved ({i+1}/{total}) ~{remaining/60:.0f}min remaining\n")
        
        time.sleep(1)
    
    with open(args.output, 'w') as f:
        json.dump(universe, f, indent=2)
    
    fin_count = sum(1 for s in universe if s.get('financial_data'))
    clin_count = sum(1 for s in universe if s.get('clinical_data'))
    
    print(f"\n{'='*80}")
    print(f"DONE: Financial {fin_count}/{total}, Clinical {clin_count}/{total}")
    print(f"Time: {(datetime.now() - start_time).total_seconds()/60:.1f} minutes")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
