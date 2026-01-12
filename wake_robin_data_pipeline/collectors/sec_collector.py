"""
sec_collector.py - Collect financial data from SEC EDGAR
Free, no API key required. Rate limit: 10 req/sec per IP (we use 1 req/sec)

Environment Variables:
    SEC_USER_AGENT: Override default User-Agent
    SEC_CACHE_DIR: Override default cache directory
"""
import json
import logging
import os
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import hashlib

logger = logging.getLogger(__name__)

# SEC requires User-Agent header with contact info
# Can be overridden via environment variable
USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "Wake Robin Research contact@wakerobincapital.com"
)

# Default cache directory (can be overridden)
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "cache" / "sec"

# Data staleness thresholds (days)
STALENESS_WARNING_DAYS = 90
STALENESS_CRITICAL_DAYS = 180


def get_cache_dir() -> Path:
    """Get cache directory from env or default."""
    cache_dir_str = os.environ.get("SEC_CACHE_DIR")
    if cache_dir_str:
        cache_dir = Path(cache_dir_str)
    else:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_cache_path(identifier: str, data_type: str = "financials") -> Path:
    """Get cache file path."""
    return get_cache_dir() / f"{identifier}_{data_type}.json"

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

def extract_latest_metric(facts_data: dict, metric_name: str, unit: str = 'USD') -> tuple[Optional[float], Optional[str]]:
    """
    Extract latest value for a GAAP metric from SEC company facts.

    Args:
        facts_data: Full company facts JSON from SEC
        metric_name: GAAP metric name (e.g., 'Assets', 'Cash')
        unit: Unit type (default 'USD', also can be 'shares', 'USD/shares')

    Returns:
        Tuple of (value, end_date) where:
        - value: Latest value as float, or None if not found
        - end_date: Date of the data point (YYYY-MM-DD), or None
    """
    try:
        # Navigate to us-gaap facts
        us_gaap = facts_data.get('facts', {}).get('us-gaap', {})

        if metric_name not in us_gaap:
            return None, None

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
                return None, None

        # Sort by date and get most recent
        sorted_values = sorted(values, key=lambda x: x.get('end', ''), reverse=True)

        if sorted_values:
            latest = sorted_values[0]
            return float(latest.get('val', 0)), latest.get('end')

        return None, None

    except Exception as e:
        logger.debug(f"Error extracting {metric_name}: {e}")
        return None, None

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

        # Track dates for staleness validation
        data_dates = {}

        cash, cash_date = None, None
        for metric in cash_metrics:
            cash, cash_date = extract_latest_metric(facts_data, metric)
            if cash is not None:
                data_dates['cash'] = cash_date
                break

        debt, debt_date = None, None
        for metric in debt_metrics:
            debt, debt_date = extract_latest_metric(facts_data, metric)
            if debt is not None:
                data_dates['debt'] = debt_date
                break

        revenue, revenue_date = None, None
        for metric in revenue_metrics:
            revenue, revenue_date = extract_latest_metric(facts_data, metric)
            if revenue is not None:
                data_dates['revenue'] = revenue_date
                break

        assets, assets_date = extract_latest_metric(facts_data, 'Assets')
        liabilities, liabilities_date = extract_latest_metric(facts_data, 'Liabilities')

        if assets is not None:
            data_dates['assets'] = assets_date
        if liabilities is not None:
            data_dates['liabilities'] = liabilities_date

        # Calculate derived metrics
        net_debt = None
        if cash is not None and debt is not None:
            net_debt = debt - cash
        elif debt is not None:
            net_debt = debt

        equity = None
        if assets is not None and liabilities is not None:
            equity = assets - liabilities

        # Determine most recent and oldest data dates for staleness check
        valid_dates = [d for d in data_dates.values() if d]
        most_recent_date = max(valid_dates) if valid_dates else None
        oldest_date = min(valid_dates) if valid_dates else None

        # Check for stale data
        staleness_flags = []
        if oldest_date:
            try:
                oldest_dt = datetime.fromisoformat(oldest_date)
                age_days = (datetime.now() - oldest_dt).days
                if age_days > STALENESS_CRITICAL_DAYS:
                    staleness_flags.append(f"critical_staleness:{age_days}d")
                    logger.warning(f"{ticker}: Financial data is {age_days} days old (critical)")
                elif age_days > STALENESS_WARNING_DAYS:
                    staleness_flags.append(f"stale_data:{age_days}d")
                    logger.info(f"{ticker}: Financial data is {age_days} days old (warning)")
            except ValueError:
                pass

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
            "data_dates": data_dates,
            "data_freshness": {
                "most_recent_date": most_recent_date,
                "oldest_date": oldest_date,
                "staleness_flags": staleness_flags,
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


def extract_metric_history(facts_data: dict, metric_name: str, unit: str = 'USD') -> list:
    """
    Extract ALL historical values for a GAAP metric from SEC company facts.

    Returns list of dicts: [{'date': 'YYYY-MM-DD', 'value': float, 'form': '10-K/10-Q'}, ...]
    """
    try:
        us_gaap = facts_data.get('facts', {}).get('us-gaap', {})
        if metric_name not in us_gaap:
            return []

        metric_data = us_gaap[metric_name]
        units_data = metric_data.get('units', {})

        if unit in units_data:
            values = units_data[unit]
        elif 'USD' in units_data:
            values = units_data['USD']
        else:
            if units_data:
                values = list(units_data.values())[0]
            else:
                return []

        # Extract all values with dates
        history = []
        seen_dates = set()
        for v in values:
            end_date = v.get('end')
            if end_date and end_date not in seen_dates:
                seen_dates.add(end_date)
                history.append({
                    'date': end_date,
                    'value': float(v.get('val', 0)),
                    'form': v.get('form', ''),
                    'filed': v.get('filed', '')
                })

        # Sort by date
        history.sort(key=lambda x: x['date'])
        return history

    except Exception as e:
        logger.debug(f"Error extracting history for {metric_name}: {e}")
        return []


def fetch_historical_financials(ticker: str, years: int = 5) -> dict:
    """
    Fetch historical financial data from SEC EDGAR for PIT backtesting.

    Args:
        ticker: Stock ticker
        years: Number of years of history to fetch

    Returns:
        dict with time series of financial snapshots
    """
    cik = ticker_to_cik(ticker)
    if not cik:
        return {"ticker": ticker, "success": False, "error": "Could not resolve CIK", "snapshots": []}

    try:
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        headers = {'User-Agent': USER_AGENT}

        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        facts_data = response.json()

        # Extract history for key metrics
        cash_metrics = ['CashAndCashEquivalentsAtCarryingValue', 'Cash', 'CashAndCashEquivalents']
        debt_metrics = ['LongTermDebt', 'LongTermDebtNoncurrent', 'DebtCurrent']

        cash_history = []
        for metric in cash_metrics:
            cash_history = extract_metric_history(facts_data, metric)
            if cash_history:
                break

        debt_history = []
        for metric in debt_metrics:
            debt_history = extract_metric_history(facts_data, metric)
            if debt_history:
                break

        assets_history = extract_metric_history(facts_data, 'Assets')
        liabilities_history = extract_metric_history(facts_data, 'Liabilities')
        rd_history = extract_metric_history(facts_data, 'ResearchAndDevelopmentExpense')

        # Build snapshots by date
        all_dates = set()
        for h in [cash_history, debt_history, assets_history, liabilities_history, rd_history]:
            for item in h:
                all_dates.add(item['date'])

        # Filter to last N years
        cutoff_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
        all_dates = {d for d in all_dates if d >= cutoff_date}

        # Create lookup dicts
        def make_lookup(history):
            return {item['date']: item['value'] for item in history}

        cash_lookup = make_lookup(cash_history)
        debt_lookup = make_lookup(debt_history)
        assets_lookup = make_lookup(assets_history)
        liabilities_lookup = make_lookup(liabilities_history)
        rd_lookup = make_lookup(rd_history)

        # Build snapshots
        snapshots = []
        for date in sorted(all_dates):
            snapshot = {
                'ticker': ticker,
                'date': date,
                'cash': cash_lookup.get(date),
                'debt': debt_lookup.get(date),
                'assets': assets_lookup.get(date),
                'liabilities': liabilities_lookup.get(date),
                'rd_expense': rd_lookup.get(date),
            }
            # Only include if we have at least cash or assets
            if snapshot['cash'] is not None or snapshot['assets'] is not None:
                snapshots.append(snapshot)

        return {
            "ticker": ticker,
            "cik": cik,
            "success": True,
            "snapshots": snapshots,
            "date_range": {
                "start": min(all_dates) if all_dates else None,
                "end": max(all_dates) if all_dates else None,
                "count": len(snapshots)
            }
        }

    except Exception as e:
        return {"ticker": ticker, "success": False, "error": str(e), "snapshots": []}


def collect_historical_batch(tickers: list[str], years: int = 5, delay_seconds: float = 1.0) -> dict:
    """
    Collect historical financial data for multiple tickers.

    Returns dict mapping ticker -> historical snapshots
    """
    results = {}
    total = len(tickers)

    print(f"\n📄 Collecting {years}-year historical SEC data for {total} tickers...")

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{total}] Fetching {ticker}...", end=" ")

        data = fetch_historical_financials(ticker, years)
        results[ticker] = data

        if data.get('success'):
            count = data['date_range']['count']
            date_range = f"{data['date_range']['start']} to {data['date_range']['end']}"
            print(f"✓ {count} snapshots ({date_range})")
        else:
            print(f"✗ {data.get('error', 'Unknown error')}")

        if i < total:
            time.sleep(delay_seconds)

    successful = sum(1 for d in results.values() if d.get('success'))
    total_snapshots = sum(len(d.get('snapshots', [])) for d in results.values())
    print(f"\n✓ Collected {total_snapshots} historical snapshots for {successful}/{total} tickers")

    return results

if __name__ == "__main__":
    # Test with a single ticker
    test_ticker = "VRTX"
    print(f"Testing SEC collector with {test_ticker}...")
    
    data = collect_sec_data(test_ticker, force_refresh=True)
    print(json.dumps(data, indent=2))
