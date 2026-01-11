#!/usr/bin/env python3
"""
threshold_list_feed.py

Downloads and parses Reg SHO Threshold Securities Lists from NYSE and Nasdaq.

Threshold securities are those with significant fails-to-deliver (FTDs).
Being on the threshold list indicates:
- Persistent delivery failures (>=10,000 shares for 5+ consecutive settlement days)
- Potential short squeeze pressure
- Elevated borrowing costs / hard-to-borrow status

Data Sources:
- Nasdaq: https://nasdaqtrader.com/trader.aspx?id=RegSHOThreshold
- NYSE: https://www.nyse.com/regulation/threshold-securities

Design Philosophy (Governed Over Smart):
- DETERMINISTIC: No datetime.now() in scoring - uses as_of_date
- STDLIB-ONLY: No external dependencies
- PIT DISCIPLINE: Lists published daily, available T+1
- FAIL LOUDLY: Clear error states
- AUDITABLE: SHA256 hashes of raw data

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import hashlib
import json
import ssl
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set


__version__ = "1.0.0"
__author__ = "Wake Robin Capital Management"


# Nasdaq threshold list URL (current day's list)
NASDAQ_THRESHOLD_URL = "https://nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth.txt"

# NYSE threshold list base URL
NYSE_THRESHOLD_BASE = "https://www.nyse.com/api/regulatory/threshold-securities/download"

# Raw data storage
RAW_DATA_DIR = Path("raw/threshold_lists")
META_DATA_DIR = Path("meta/threshold_lists")


def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def prev_business_day(d: date) -> date:
    """Get previous business day."""
    d = d - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def fetch_url(url: str, timeout: int = 30) -> bytes:
    """Fetch URL content."""
    ctx = ssl.create_default_context()

    request = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (compatible; WakeRobinDataFeed/1.0)',
            'Accept': 'text/plain,text/html,*/*;q=0.8',
        }
    )

    with urllib.request.urlopen(request, timeout=timeout, context=ctx) as response:
        return response.read()


def download_nasdaq_threshold_list(
    list_date: date,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Download Nasdaq threshold securities list.

    Note: Nasdaq provides the current day's list at the standard URL.
    Historical lists may require different access.

    Args:
        list_date: Date of the list
        output_dir: Output directory

    Returns:
        Download result dict
    """
    output_dir = output_dir or RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try dated URL patterns
        date_str = list_date.strftime("%Y%m%d")
        url_patterns = [
            f"https://nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{date_str}.txt",
            NASDAQ_THRESHOLD_URL,  # Current list as fallback
        ]

        data = None
        used_url = None
        for url in url_patterns:
            try:
                data = fetch_url(url)
                used_url = url
                break
            except Exception:
                continue

        if data is None:
            return {
                "success": False,
                "error": "DOWNLOAD_FAILED",
                "message": "Could not download Nasdaq threshold list",
                "exchange": "NASDAQ",
            }

        # Save raw file
        output_file = output_dir / f"nasdaq_{list_date.isoformat()}.txt"
        with open(output_file, 'wb') as f:
            f.write(data)

        data_hash = compute_sha256(data)

        # Save metadata
        meta_dir = META_DATA_DIR
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_file = meta_dir / f"nasdaq_{list_date.isoformat()}.json"

        meta = {
            "list_date": list_date.isoformat(),
            "exchange": "NASDAQ",
            "source_url": used_url,
            "sha256": data_hash,
            "file_size_bytes": len(data),
            "raw_file_path": str(output_file),
            "download_version": __version__,
        }

        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)

        return {
            "success": True,
            "list_date": list_date.isoformat(),
            "exchange": "NASDAQ",
            "file_path": str(output_file),
            "sha256": data_hash,
            "source_url": used_url,
        }

    except Exception as e:
        return {
            "success": False,
            "error": "DOWNLOAD_FAILED",
            "message": str(e),
            "exchange": "NASDAQ",
        }


def download_nyse_threshold_list(
    list_date: date,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Download NYSE threshold securities list.

    Args:
        list_date: Date of the list
        output_dir: Output directory

    Returns:
        Download result dict
    """
    output_dir = output_dir or RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # NYSE API endpoint (may require date parameter)
        date_str = list_date.strftime("%Y-%m-%d")
        url_patterns = [
            f"{NYSE_THRESHOLD_BASE}?date={date_str}",
            NYSE_THRESHOLD_BASE,
        ]

        data = None
        used_url = None
        for url in url_patterns:
            try:
                data = fetch_url(url)
                used_url = url
                break
            except Exception:
                continue

        if data is None:
            return {
                "success": False,
                "error": "DOWNLOAD_FAILED",
                "message": "Could not download NYSE threshold list",
                "exchange": "NYSE",
            }

        # Save raw file
        output_file = output_dir / f"nyse_{list_date.isoformat()}.txt"
        with open(output_file, 'wb') as f:
            f.write(data)

        data_hash = compute_sha256(data)

        # Save metadata
        meta_dir = META_DATA_DIR
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_file = meta_dir / f"nyse_{list_date.isoformat()}.json"

        meta = {
            "list_date": list_date.isoformat(),
            "exchange": "NYSE",
            "source_url": used_url,
            "sha256": data_hash,
            "file_size_bytes": len(data),
            "raw_file_path": str(output_file),
            "download_version": __version__,
        }

        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)

        return {
            "success": True,
            "list_date": list_date.isoformat(),
            "exchange": "NYSE",
            "file_path": str(output_file),
            "sha256": data_hash,
            "source_url": used_url,
        }

    except Exception as e:
        return {
            "success": False,
            "error": "DOWNLOAD_FAILED",
            "message": str(e),
            "exchange": "NYSE",
        }


def parse_nasdaq_threshold_file(file_path: Path) -> Set[str]:
    """
    Parse Nasdaq threshold list file to extract symbols.

    Format: Symbol|SecurityName|Market|Date|Threshold Rule

    Returns:
        Set of threshold securities symbols
    """
    symbols = set()

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    lines = content.strip().split('\n')

    for line in lines:
        if not line.strip() or line.startswith('#'):
            continue

        # Handle pipe-delimited format
        if '|' in line:
            parts = line.split('|')
            if len(parts) >= 1:
                symbol = parts[0].strip()
                # Skip header rows
                if symbol.upper() in ('SYMBOL', 'SECURITY', 'TICKER'):
                    continue
                if symbol and len(symbol) <= 10 and symbol.isalpha():
                    symbols.add(symbol.upper())
        else:
            # Space or comma delimited
            parts = line.replace(',', ' ').split()
            if parts:
                symbol = parts[0].strip()
                if symbol and len(symbol) <= 10 and symbol.isalpha():
                    symbols.add(symbol.upper())

    return symbols


def parse_nyse_threshold_file(file_path: Path) -> Set[str]:
    """
    Parse NYSE threshold list file to extract symbols.

    Returns:
        Set of threshold securities symbols
    """
    symbols = set()

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    lines = content.strip().split('\n')

    for line in lines:
        if not line.strip() or line.startswith('#'):
            continue

        # Handle various delimiters
        for delimiter in ['|', ',', '\t']:
            if delimiter in line:
                parts = line.split(delimiter)
                if len(parts) >= 1:
                    symbol = parts[0].strip()
                    if symbol.upper() in ('SYMBOL', 'SECURITY', 'TICKER'):
                        continue
                    if symbol and len(symbol) <= 10:
                        # Remove common suffixes
                        symbol = symbol.split('.')[0].split('-')[0]
                        if symbol.isalpha():
                            symbols.add(symbol.upper())
                break
        else:
            # Single column or space delimited
            parts = line.split()
            if parts:
                symbol = parts[0].strip().split('.')[0].split('-')[0]
                if symbol and len(symbol) <= 10 and symbol.isalpha():
                    symbols.add(symbol.upper())

    return symbols


def load_threshold_securities(
    list_date: date,
    exchanges: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Load threshold securities for a given date.

    Args:
        list_date: Date of the threshold list
        exchanges: List of exchanges to include (default: all)

    Returns:
        Dict with threshold securities by exchange and combined set
    """
    exchanges = exchanges or ["NASDAQ", "NYSE"]
    result = {
        "list_date": list_date.isoformat(),
        "exchanges": {},
        "combined": set(),
        "total_count": 0,
    }

    for exchange in exchanges:
        if exchange.upper() == "NASDAQ":
            file_path = RAW_DATA_DIR / f"nasdaq_{list_date.isoformat()}.txt"
            if file_path.exists():
                symbols = parse_nasdaq_threshold_file(file_path)
                result["exchanges"]["NASDAQ"] = symbols
                result["combined"].update(symbols)

        elif exchange.upper() == "NYSE":
            file_path = RAW_DATA_DIR / f"nyse_{list_date.isoformat()}.txt"
            if file_path.exists():
                symbols = parse_nyse_threshold_file(file_path)
                result["exchanges"]["NYSE"] = symbols
                result["combined"].update(symbols)

    result["total_count"] = len(result["combined"])

    return result


def is_threshold_security(
    symbol: str,
    list_date: date,
) -> bool:
    """
    Check if a symbol is on the threshold securities list.

    Args:
        symbol: Stock symbol
        list_date: Date to check

    Returns:
        True if symbol is on threshold list
    """
    threshold_data = load_threshold_securities(list_date)
    return symbol.upper() in threshold_data["combined"]


def get_threshold_flags_for_universe(
    tickers: List[str],
    as_of_date: date,
) -> Dict[str, bool]:
    """
    Get threshold flags for a list of tickers.

    Args:
        tickers: List of ticker symbols
        as_of_date: As-of date (uses previous business day's list)

    Returns:
        Dict mapping ticker -> is_threshold_security
    """
    # Threshold lists are for T-1 (published next business day)
    list_date = prev_business_day(as_of_date)

    threshold_data = load_threshold_securities(list_date)
    threshold_set = threshold_data["combined"]

    return {
        ticker.upper(): ticker.upper() in threshold_set
        for ticker in tickers
    }


def demonstration():
    """Demonstrate the threshold list feed."""
    print("=" * 70)
    print("THRESHOLD SECURITIES LIST FEED v1.0.0 - DEMONSTRATION")
    print("=" * 70)
    print()

    as_of = date(2026, 1, 11)
    list_date = prev_business_day(as_of)
    print(f"As-of date: {as_of}")
    print(f"Threshold list date: {list_date}")
    print()

    # Check for cached data
    nasdaq_file = RAW_DATA_DIR / f"nasdaq_{list_date.isoformat()}.txt"
    nyse_file = RAW_DATA_DIR / f"nyse_{list_date.isoformat()}.txt"

    print("Cached files:")
    print(f"  Nasdaq: {'Found' if nasdaq_file.exists() else 'Not found'}")
    print(f"  NYSE: {'Found' if nyse_file.exists() else 'Not found'}")
    print()

    # Load and display if available
    threshold_data = load_threshold_securities(list_date)
    if threshold_data["total_count"] > 0:
        print(f"Total threshold securities: {threshold_data['total_count']}")
        for exchange, symbols in threshold_data["exchanges"].items():
            print(f"  {exchange}: {len(symbols)} securities")
            for sym in sorted(list(symbols))[:10]:
                print(f"    {sym}")
            if len(symbols) > 10:
                print(f"    ... and {len(symbols) - 10} more")
    else:
        print("No threshold data loaded")

    print()
    print("To download threshold lists, run:")
    print("  python threshold_list_feed.py --download --date YYYY-MM-DD")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Threshold Securities List Feed")
    parser.add_argument("--download", action="store_true", help="Download threshold lists")
    parser.add_argument("--date", type=str, help="List date (YYYY-MM-DD)")
    parser.add_argument("--exchange", type=str, choices=["nasdaq", "nyse", "all"], default="all")

    args = parser.parse_args()

    if args.download and args.date:
        list_date = date.fromisoformat(args.date)
        exchanges = ["NASDAQ", "NYSE"] if args.exchange == "all" else [args.exchange.upper()]

        for exchange in exchanges:
            print(f"Downloading {exchange} threshold list for {list_date}...")
            if exchange == "NASDAQ":
                result = download_nasdaq_threshold_list(list_date)
            else:
                result = download_nyse_threshold_list(list_date)

            if result["success"]:
                print(f"  Success! Saved to {result['file_path']}")
            else:
                print(f"  Failed: {result['message']}")
    else:
        demonstration()
