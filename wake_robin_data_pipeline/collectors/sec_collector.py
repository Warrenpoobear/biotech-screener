"""
sec_collector.py - Collect financial data from SEC EDGAR
Free, no API key required. Rate limit: 10 req/sec per IP (we use 1 req/sec)
"""
import json
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import hashlib

# SEC requires User-Agent header with contact info
USER_AGENT = "Wake Robin Research contact@wakerobincapital.com"

def get_cache_path(identifier: str, data_type: str = "financials") -> Path:
    """Get cache file path."""
    cache_dir = Path(__file__).parent.parent / "cache" / "sec"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{identifier}_{data_type}.json"

def is_cache_valid(cache_path: Path, max_age_hours: int = 24) -> bool:
    """Check if cache is fresh enough."""
    if not cache_path.exists():
        return False
    
    age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
    return age < timedelta(hours=max_age_hours)

def ticker_to_cik(ticker: str) -> Optional[str]:
    """
    Resolve ticker to CIK using SEC's company tickers JSON.
    Returns 10-digit CIK string or None if not found.
    """
    cache_path = get_cache_path(ticker, "cik_mapping")
    
    # Check cache first
    if is_cache_valid(cache_path, max_age_hours=168):  # Cache for 1 week
        with open(cache_path) as f:
            cached = json.load(f)
            return cached.get('cik')
    
    try:
        # SEC maintains a ticker->CIK mapping file
        url = "https://www.sec.gov/files/company_tickers.json"
        headers = {'User-Agent': USER_AGENT}
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        # Search for ticker (case-insensitive)
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get('ticker', '').upper() == ticker_upper:
                cik = str(entry['cik_str']).zfill(10)
                
                # Cache the mapping
                with open(cache_path, 'w') as f:
                    json.dump({
                        'ticker': ticker,
                        'cik': cik,
                        'company_name': entry.get('title', ''),
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
                
                return cik
        
        # Not found
        with open(cache_path, 'w') as f:
            json.dump({
                'ticker': ticker,
                'cik': None,
                'error': 'Ticker not found in SEC database',
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        return None
        
    except Exception as e:
        print(f"  Warning: CIK resolution failed: {e}")
        return None

def extract_latest_metric(facts_data: dict, metric_name: str, unit: str = 'USD') -> Optional[float]:
    """
    Extract latest value for a GAAP metric from SEC company facts.
    
    Args:
        facts_data: Full company facts JSON from SEC
        metric_name: GAAP metric name (e.g., 'Assets', 'Cash')
        unit: Unit type (default 'USD', also can be 'shares', 'USD/shares')
    
    Returns:
        Latest value as float, or None if not found
    """
    try:
        # Navigate to us-gaap facts
        us_gaap = facts_data.get('facts', {}).get('us-gaap', {})
        
        if metric_name not in us_gaap:
            return None
        
        # Get units data
        metric_data = us_gaap[metric_name]
        units_data = metric_data.get('units', {})
        
        # Try to find the right unit
        if unit in units_data:
            values = units_data[unit]
        elif 'USD' in units_data:
            values = units_data['USD']
        else:
            # Try first available unit
            if units_data:
                values = list(units_data.values())[0]
            else:
                return None
        
        # Sort by date and get most recent
        sorted_values = sorted(values, key=lambda x: x.get('end', ''), reverse=True)
        
        if sorted_values:
            return float(sorted_values[0].get('val', 0))
        
        return None
        
    except Exception:
        return None

def fetch_sec_financials(ticker: str, cik: Optional[str] = None) -> dict:
    """
    Fetch financial data from SEC EDGAR company facts API.
    
    Args:
        ticker: Stock ticker
        cik: 10-digit CIK (will auto-resolve if None)
        
    Returns:
        dict with financial data and provenance
    """
    # Resolve CIK if not provided
    if not cik:
        cik = ticker_to_cik(ticker)
        if not cik:
            return {
                "ticker": ticker,
                "success": False,
                "error": "Could not resolve ticker to CIK",
                "timestamp": datetime.now().isoformat()
            }
    
    try:
        # Fetch company facts
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        headers = {'User-Agent': USER_AGENT}
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        facts_data = response.json()
        
        # Extract key metrics (try common variations)
        cash_metrics = ['CashAndCashEquivalentsAtCarryingValue', 'Cash', 'CashAndCashEquivalents']
        debt_metrics = ['LongTermDebt', 'LongTermDebtNoncurrent', 'DebtCurrent']
        revenue_metrics = ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet']
        assets_metrics = ['Assets']
        liabilities_metrics = ['Liabilities']
        
        cash = None
        for metric in cash_metrics:
            cash = extract_latest_metric(facts_data, metric)
            if cash:
                break
        
        debt = None
        for metric in debt_metrics:
            debt = extract_latest_metric(facts_data, metric)
            if debt:
                break
        
        revenue = None
        for metric in revenue_metrics:
            revenue = extract_latest_metric(facts_data, metric)
            if revenue:
                break
        
        assets = extract_latest_metric(facts_data, 'Assets')
        liabilities = extract_latest_metric(facts_data, 'Liabilities')
        
        # Calculate derived metrics
        net_debt = None
        if cash is not None and debt is not None:
            net_debt = debt - cash
        elif debt is not None:
            net_debt = debt
        
        equity = None
        if assets is not None and liabilities is not None:
            equity = assets - liabilities
        
        data = {
            "ticker": ticker,
            "cik": cik,
            "success": True,
            "financials": {
                "cash": cash,
                "debt": debt,
                "net_debt": net_debt,
                "revenue_ttm": revenue,
                "assets": assets,
                "liabilities": liabilities,
                "equity": equity,
                "currency": "USD"
            },
            "coverage": {
                "has_cash": cash is not None,
                "has_debt": debt is not None,
                "has_revenue": revenue is not None,
                "has_balance_sheet": assets is not None and liabilities is not None,
                "pct_complete": sum([
                    cash is not None,
                    debt is not None,
                    revenue is not None,
                    assets is not None
                ]) / 4 * 100
            },
            "provenance": {
                "source": "SEC EDGAR Company Facts API",
                "timestamp": datetime.now().isoformat(),
                "url": f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                "cik": cik
            }
        }
        
        return data
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            error = f"CIK {cik} not found in SEC database"
        else:
            error = f"HTTP {e.response.status_code}: {str(e)}"
        
        return {
            "ticker": ticker,
            "cik": cik,
            "success": False,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "ticker": ticker,
            "cik": cik,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def collect_sec_data(ticker: str, force_refresh: bool = False) -> dict:
    """
    Main entry point: collect SEC financial data with caching.
    """
    cache_path = get_cache_path(ticker, "financials")
    
    # Check cache first
    if not force_refresh and is_cache_valid(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
            cached['from_cache'] = True
            return cached
    
    # Fetch fresh data
    data = fetch_sec_financials(ticker)
    
    # Cache successful results
    if data.get('success'):
        with open(cache_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    data['from_cache'] = False
    return data

def collect_batch(tickers: list[str], delay_seconds: float = 1.0) -> dict:
    """Collect SEC data for multiple tickers with rate limiting."""
    results = {}
    total = len(tickers)
    
    print(f"\n📄 Collecting SEC EDGAR data for {total} tickers...")
    
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{total}] Fetching {ticker}...", end=" ")
        
        data = collect_sec_data(ticker)
        results[ticker] = data
        
        if data.get('success'):
            fin = data['financials']
            coverage = data['coverage']['pct_complete']
            cash_str = f"${fin['cash']/1e6:.0f}M" if fin['cash'] else "N/A"
            cached = " (cached)" if data.get('from_cache') else ""
            print(f"✓ Cash: {cash_str}, Coverage: {coverage:.0f}%{cached}")
        else:
            print(f"✗ {data.get('error', 'Unknown error')}")
        
        # Rate limiting
        if i < total and not data.get('from_cache'):
            time.sleep(delay_seconds)
    
    successful = sum(1 for d in results.values() if d.get('success'))
    print(f"\n✓ Successfully collected data for {successful}/{total} tickers")
    
    return results

if __name__ == "__main__":
    # Test with a single ticker
    test_ticker = "VRTX"
    print(f"Testing SEC collector with {test_ticker}...")
    
    data = collect_sec_data(test_ticker, force_refresh=True)
    print(json.dumps(data, indent=2))
