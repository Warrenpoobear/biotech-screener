"""
data_merger.py - Merge financial data from multiple sources (SEC, Yahoo)

Prioritization strategy:
1. Prefer more recent data (within 1 year)
2. For same freshness, prefer SEC (authoritative) over Yahoo
3. Fill gaps with any available source
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Cache directories
SEC_CACHE_DIR = Path(__file__).parent.parent / "cache" / "sec"
YAHOO_CACHE_DIR = Path(__file__).parent.parent / "cache" / "yahoo"

# Maximum age for data to be considered valid (days)
MAX_DATA_AGE_DAYS = 365


def load_sec_data(ticker: str) -> Optional[dict]:
    """Load SEC financial data for a ticker."""
    cache_file = SEC_CACHE_DIR / f"{ticker}_financials.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def load_yahoo_data(ticker: str) -> Optional[dict]:
    """Load Yahoo Finance data for a ticker."""
    cache_file = YAHOO_CACHE_DIR / f"{ticker}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)
    return None


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string to datetime."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace('Z', '+00:00').split('+')[0])
    except (ValueError, AttributeError):
        return None


def is_fresh(date_str: str, max_age_days: int = MAX_DATA_AGE_DAYS) -> bool:
    """Check if a date is within the acceptable age range."""
    dt = parse_date(date_str)
    if not dt:
        return False
    age = datetime.now() - dt
    return age.days <= max_age_days


def merge_financial_data(ticker: str) -> Dict[str, Any]:
    """
    Merge SEC and Yahoo financial data for a ticker.

    Returns a consolidated dict with:
    - All available financial fields
    - Provenance information for each field
    - Data freshness indicators
    """
    sec_data = load_sec_data(ticker)
    yahoo_data = load_yahoo_data(ticker)

    result = {
        "ticker": ticker,
        "success": False,
        "financials": {},
        "data_dates": {},
        "data_sources": {},
        "provenance": {
            "merged_at": datetime.now().isoformat(),
            "sec_available": sec_data is not None and sec_data.get("success", False),
            "yahoo_available": yahoo_data is not None and yahoo_data.get("success", False)
        }
    }

    # Define field mappings: (result_field, sec_path, yahoo_path, sec_date_path, yahoo_date_path)
    field_mappings = [
        # Cash and liquidity
        ("cash",
         ("financials", "cash"),
         ("balance_sheet", "cash"),
         ("data_dates", "cash"),
         ("balance_sheet", "period_date")),
        ("marketable_securities",
         ("financials", "marketable_securities"),
         ("balance_sheet", "marketable_securities"),
         ("data_dates", "marketable_securities"),
         ("balance_sheet", "period_date")),
        ("total_liquidity",
         ("financials", "total_liquidity"),
         ("balance_sheet", "total_liquidity"),
         ("data_dates", "cash"),  # Use cash date for SEC total_liquidity
         ("balance_sheet", "period_date")),
        # Debt
        ("total_debt",
         ("financials", "debt"),
         ("balance_sheet", "total_debt"),
         ("data_dates", "debt"),
         ("balance_sheet", "period_date")),
        ("long_term_debt",
         None,  # SEC doesn't have separate long-term debt in our extraction
         ("balance_sheet", "long_term_debt"),
         None,
         ("balance_sheet", "period_date")),
        # Balance sheet
        ("total_assets",
         ("financials", "assets"),
         ("balance_sheet", "total_assets"),
         ("data_dates", "assets"),
         ("balance_sheet", "period_date")),
        ("total_liabilities",
         ("financials", "liabilities"),
         ("balance_sheet", "total_liabilities"),
         ("data_dates", "liabilities"),
         ("balance_sheet", "period_date")),
        ("stockholders_equity",
         ("financials", "equity"),
         ("balance_sheet", "stockholders_equity"),
         ("data_dates", "assets"),  # Use assets date for equity
         ("balance_sheet", "period_date")),
        # Revenue (SEC only typically)
        ("revenue_ttm",
         ("financials", "revenue_ttm"),
         None,
         ("data_dates", "revenue"),
         None),
    ]

    def get_nested(data: dict, path: tuple) -> Any:
        """Get nested value from dict."""
        if data is None or path is None:
            return None
        result = data
        for key in path:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return None
        return result

    for field_name, sec_path, yahoo_path, sec_date_path, yahoo_date_path in field_mappings:
        # Get values from both sources
        sec_value = get_nested(sec_data, sec_path)
        yahoo_value = get_nested(yahoo_data, yahoo_path)
        sec_date = get_nested(sec_data, sec_date_path)
        yahoo_date = get_nested(yahoo_data, yahoo_date_path)

        # Determine which source to use
        chosen_value = None
        chosen_source = None
        chosen_date = None

        sec_fresh = is_fresh(sec_date) if sec_date else False
        yahoo_fresh = is_fresh(yahoo_date) if yahoo_date else False

        # Decision logic:
        # 1. Both fresh: prefer SEC (authoritative)
        # 2. Only one fresh: use the fresh one
        # 3. Neither fresh but available: prefer more recent
        # 4. Only one available: use it

        if sec_value is not None and yahoo_value is not None:
            if sec_fresh and yahoo_fresh:
                # Both fresh, prefer SEC
                chosen_value = sec_value
                chosen_source = "SEC"
                chosen_date = sec_date
            elif sec_fresh:
                chosen_value = sec_value
                chosen_source = "SEC"
                chosen_date = sec_date
            elif yahoo_fresh:
                chosen_value = yahoo_value
                chosen_source = "Yahoo"
                chosen_date = yahoo_date
            else:
                # Neither fresh, compare dates
                sec_dt = parse_date(sec_date)
                yahoo_dt = parse_date(yahoo_date)
                if sec_dt and yahoo_dt:
                    if sec_dt >= yahoo_dt:
                        chosen_value = sec_value
                        chosen_source = "SEC"
                        chosen_date = sec_date
                    else:
                        chosen_value = yahoo_value
                        chosen_source = "Yahoo"
                        chosen_date = yahoo_date
                elif sec_dt:
                    chosen_value = sec_value
                    chosen_source = "SEC"
                    chosen_date = sec_date
                else:
                    chosen_value = yahoo_value
                    chosen_source = "Yahoo"
                    chosen_date = yahoo_date
        elif sec_value is not None:
            chosen_value = sec_value
            chosen_source = "SEC"
            chosen_date = sec_date
        elif yahoo_value is not None:
            chosen_value = yahoo_value
            chosen_source = "Yahoo"
            chosen_date = yahoo_date

        if chosen_value is not None:
            result["financials"][field_name] = chosen_value
            result["data_sources"][field_name] = chosen_source
            if chosen_date:
                result["data_dates"][field_name] = chosen_date

    # Calculate derived fields
    financials = result["financials"]

    # Net debt = total_debt - total_liquidity
    if financials.get("total_debt") is not None and financials.get("total_liquidity") is not None:
        financials["net_debt"] = financials["total_debt"] - financials["total_liquidity"]
    elif financials.get("total_debt") is not None and financials.get("cash") is not None:
        financials["net_debt"] = financials["total_debt"] - financials["cash"]

    # Mark success if we have at least some core data
    if financials.get("cash") or financials.get("total_assets") or financials.get("total_liabilities"):
        result["success"] = True

    return result


def merge_all_tickers(tickers: list) -> Dict[str, Dict]:
    """Merge financial data for all tickers."""
    results = {}
    for ticker in tickers:
        results[ticker] = merge_financial_data(ticker)
    return results


def analyze_data_coverage(tickers: list) -> dict:
    """Analyze data coverage across sources."""
    stats = {
        "total": len(tickers),
        "sec_only": 0,
        "yahoo_only": 0,
        "both_sources": 0,
        "neither_source": 0,
        "by_field": {}
    }

    fields = ["cash", "total_liquidity", "total_debt", "total_assets", "total_liabilities", "stockholders_equity"]
    for field in fields:
        stats["by_field"][field] = {"sec": 0, "yahoo": 0, "merged": 0}

    for ticker in tickers:
        merged = merge_financial_data(ticker)

        sec_has = merged["provenance"]["sec_available"]
        yahoo_has = merged["provenance"]["yahoo_available"]

        if sec_has and yahoo_has:
            stats["both_sources"] += 1
        elif sec_has:
            stats["sec_only"] += 1
        elif yahoo_has:
            stats["yahoo_only"] += 1
        else:
            stats["neither_source"] += 1

        for field in fields:
            source = merged["data_sources"].get(field)
            if source == "SEC":
                stats["by_field"][field]["sec"] += 1
                stats["by_field"][field]["merged"] += 1
            elif source == "Yahoo":
                stats["by_field"][field]["yahoo"] += 1
                stats["by_field"][field]["merged"] += 1

    return stats


if __name__ == "__main__":
    # Test with a few tickers
    test_tickers = ["AMGN", "GOSS", "VRTX", "AKRO"]

    print("Testing data merger...")
    for ticker in test_tickers:
        merged = merge_financial_data(ticker)
        print(f"\n=== {ticker} ===")
        print(f"Success: {merged['success']}")
        print(f"SEC available: {merged['provenance']['sec_available']}")
        print(f"Yahoo available: {merged['provenance']['yahoo_available']}")
        print(f"Financials:")
        for k, v in merged['financials'].items():
            source = merged['data_sources'].get(k, '?')
            date = merged['data_dates'].get(k, '?')
            if v is not None:
                print(f"  {k}: {v:,.0f} (source: {source}, date: {date})")
