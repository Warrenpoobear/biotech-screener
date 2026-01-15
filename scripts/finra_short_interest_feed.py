#!/usr/bin/env python3
"""
finra_short_interest_feed.py

Downloads and parses FINRA Equity Short Interest data.

FINRA publishes short interest data twice monthly (settlement dates around
mid-month and end-of-month). Data is disseminated ~11 business days after
settlement date.

Design Philosophy (Governed Over Smart):
- DETERMINISTIC: No datetime.now() in scoring - uses as_of_date
- STDLIB-ONLY: No external dependencies (uses urllib)
- PIT DISCIPLINE: Tracks settlement_date vs dissemination_date
- FAIL LOUDLY: Clear error states
- AUDITABLE: SHA256 hashes of raw data

Data Source: https://www.finra.org/finra-data/browse-catalog/equity-short-interest/files

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import csv
import hashlib
import io
import json
import os
import re
import ssl
import urllib.request
import zipfile
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from html.parser import HTMLParser


__version__ = "1.0.0"
__author__ = "Wake Robin Capital Management"


# FINRA catalog URL for equity short interest files
FINRA_SI_CATALOG_URL = "https://www.finra.org/finra-data/browse-catalog/equity-short-interest/files"

# FINRA direct download base (files are typically named by settlement date)
FINRA_SI_DOWNLOAD_BASE = "https://cdn.finra.org/equity/otcmarket/biweekly"

# Settlement to dissemination lag (conservative estimate: 11 business days)
# FINRA rule: data must be disseminated by T+11 business days
DISSEMINATION_LAG_BUSINESS_DAYS = 11

# Raw data storage paths
RAW_DATA_DIR = Path("raw/finra_equity_short_interest")
META_DATA_DIR = Path("meta/finra_equity_short_interest")


class FINRACatalogParser(HTMLParser):
    """Parse FINRA catalog page to find available SI files."""

    def __init__(self):
        super().__init__()
        self.files = []
        self.in_link = False
        self.current_href = None

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for name, value in attrs:
                if name == 'href' and value and ('short' in value.lower() or '.zip' in value.lower() or '.txt' in value.lower()):
                    self.current_href = value
                    self.in_link = True

    def handle_endtag(self, tag):
        if tag == 'a':
            self.in_link = False
            self.current_href = None

    def handle_data(self, data):
        if self.in_link and self.current_href:
            self.files.append({
                'url': self.current_href,
                'text': data.strip()
            })


def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def business_days_between(start: date, end: date) -> int:
    """Count business days between two dates (excluding weekends)."""
    days = 0
    current = start
    while current < end:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Monday = 0, Friday = 4
            days += 1
    return days


def add_business_days(start: date, days: int) -> date:
    """Add business days to a date."""
    current = start
    added = 0
    while added < days:
        current += timedelta(days=1)
        if current.weekday() < 5:
            added += 1
    return current


def estimate_dissemination_date(settlement_date: date) -> date:
    """
    Estimate when SI data becomes available (dissemination date).

    FINRA rule: data disseminated by T+11 business days after settlement.
    We use this as the "available_date" for PIT purposes.
    """
    return add_business_days(settlement_date, DISSEMINATION_LAG_BUSINESS_DAYS)


def is_data_available(settlement_date: date, as_of_date: date) -> bool:
    """
    Check if SI data for a settlement date would be available as of a given date.

    PIT DISCIPLINE: Data is only "known" after dissemination.
    """
    dissemination_date = estimate_dissemination_date(settlement_date)
    return as_of_date >= dissemination_date


def get_latest_available_settlement_date(as_of_date: date) -> date:
    """
    Get the most recent settlement date whose data would be available.

    FINRA reports SI twice monthly:
    - Mid-month settlement: ~15th
    - End-month settlement: last business day

    Returns the most recent settlement date that would be disseminated by as_of_date.
    """
    # Work backwards from as_of_date to find latest available settlement
    # Subtract dissemination lag to get approximate settlement window
    earliest_settlement = as_of_date - timedelta(days=DISSEMINATION_LAG_BUSINESS_DAYS + 7)

    # Find mid-month and end-month settlement dates for relevant months
    candidates = []

    for month_offset in range(3):  # Check last 3 months
        year = as_of_date.year
        month = as_of_date.month - month_offset
        if month < 1:
            month += 12
            year -= 1

        # Mid-month: 15th (or nearest business day)
        mid_month = date(year, month, 15)
        while mid_month.weekday() > 4:  # If weekend, go to Friday
            mid_month -= timedelta(days=1)

        # End-month: last business day
        if month == 12:
            next_month_1st = date(year + 1, 1, 1)
        else:
            next_month_1st = date(year, month + 1, 1)
        end_month = next_month_1st - timedelta(days=1)
        while end_month.weekday() > 4:
            end_month -= timedelta(days=1)

        candidates.extend([mid_month, end_month])

    # Filter to those that would be available
    available = [d for d in candidates if is_data_available(d, as_of_date)]

    if available:
        return max(available)

    # Fallback: approximate
    return as_of_date - timedelta(days=15)


def fetch_url(url: str, timeout: int = 30) -> bytes:
    """Fetch URL content with proper SSL handling."""
    # Create SSL context that handles most certificate scenarios
    ctx = ssl.create_default_context()

    request = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (compatible; WakeRobinDataFeed/1.0)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
    )

    with urllib.request.urlopen(request, timeout=timeout, context=ctx) as response:
        return response.read()


def download_finra_si_file(
    settlement_date: date,
    output_dir: Optional[Path] = None,
    as_of_date: Optional[date] = None
) -> Dict[str, Any]:
    """
    Download FINRA short interest file for a specific settlement date.

    Args:
        settlement_date: The SI settlement date to fetch
        output_dir: Directory to save raw files (default: RAW_DATA_DIR)
        as_of_date: For PIT validation (optional)

    Returns:
        Dict with download result and metadata
    """
    output_dir = output_dir or RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # PIT check
    if as_of_date and not is_data_available(settlement_date, as_of_date):
        return {
            "success": False,
            "error": "PIT_VIOLATION",
            "message": f"Settlement {settlement_date} data not yet disseminated as of {as_of_date}",
            "settlement_date": settlement_date.isoformat(),
            "estimated_available": estimate_dissemination_date(settlement_date).isoformat(),
        }

    # FINRA file naming convention: CNMSshvol{YYYYMMDD}.txt or similar
    date_str = settlement_date.strftime("%Y%m%d")

    # Try multiple URL patterns (FINRA has changed formats over time)
    url_patterns = [
        f"{FINRA_SI_DOWNLOAD_BASE}/CNMSequity{date_str}.txt",
        f"{FINRA_SI_DOWNLOAD_BASE}/CNMSshvol{date_str}.txt",
        f"{FINRA_SI_DOWNLOAD_BASE}/FNRAshvol{date_str}.txt",
        f"https://cdn.finra.org/equity/otcmarket/biweekly/shvol{date_str}.txt",
    ]

    last_error = None
    for url in url_patterns:
        try:
            data = fetch_url(url)

            # Save raw file
            output_file = output_dir / f"{settlement_date.isoformat()}.txt"
            with open(output_file, 'wb') as f:
                f.write(data)

            # Compute hash
            data_hash = compute_sha256(data)

            # Save metadata
            meta_dir = META_DATA_DIR
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_file = meta_dir / f"{settlement_date.isoformat()}.json"

            meta = {
                "settlement_date": settlement_date.isoformat(),
                "dissemination_date_estimate": estimate_dissemination_date(settlement_date).isoformat(),
                "source_url": url,
                "sha256": data_hash,
                "file_size_bytes": len(data),
                "raw_file_path": str(output_file),
                "download_version": __version__,
            }

            with open(meta_file, 'w') as f:
                json.dump(meta, f, indent=2)

            return {
                "success": True,
                "settlement_date": settlement_date.isoformat(),
                "file_path": str(output_file),
                "sha256": data_hash,
                "source_url": url,
                "records_hint": data.count(b'\n'),
            }

        except urllib.error.HTTPError as e:
            last_error = f"HTTP {e.code}: {e.reason}"
            continue
        except urllib.error.URLError as e:
            last_error = f"URL Error: {e.reason}"
            continue
        except Exception as e:
            last_error = str(e)
            continue

    return {
        "success": False,
        "error": "DOWNLOAD_FAILED",
        "message": f"Could not download SI file for {settlement_date}: {last_error}",
        "settlement_date": settlement_date.isoformat(),
        "tried_urls": url_patterns,
    }


def parse_finra_si_file(
    file_path: Path,
    settlement_date: date
) -> List[Dict[str, Any]]:
    """
    Parse a FINRA short interest file into records.

    FINRA SI file format (pipe-delimited):
    SYMBOL|SECURITY_NAME|MARKET|CURRENT_SHORT_POSITION|PREVIOUS_SHORT_POSITION|CHANGE|AVG_DAILY_VOLUME|DAYS_TO_COVER

    Returns:
        List of parsed SI records
    """
    records = []

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Detect delimiter (pipe or comma)
    if '|' in content[:500]:
        delimiter = '|'
    else:
        delimiter = ','

    lines = content.strip().split('\n')

    # Find header row
    header_idx = 0
    for i, line in enumerate(lines):
        if 'SYMBOL' in line.upper() or 'TICKER' in line.upper():
            header_idx = i
            break

    if header_idx >= len(lines) - 1:
        return records

    # Parse header
    header = [h.strip().upper() for h in lines[header_idx].split(delimiter)]

    # Map common column names
    col_map = {
        'SYMBOL': ['SYMBOL', 'TICKER', 'ISSUESYMBOL'],
        'NAME': ['SECURITY_NAME', 'ISSUENAME', 'NAME', 'SECURITYNAME'],
        'MARKET': ['MARKET', 'MARKETCATEGORY', 'EXCHANGE'],
        'CURRENT_SI': ['CURRENT_SHORT_POSITION', 'CURRENTSHORTPOSITION', 'SHORTINTEREST', 'SI'],
        'PREV_SI': ['PREVIOUS_SHORT_POSITION', 'PREVIOUSSHORTPOSITION', 'PREVSI'],
        'CHANGE': ['CHANGE', 'CHANGE_PCT', 'CHANGEINSHORTINTEREST'],
        'ADV': ['AVG_DAILY_VOLUME', 'AVGDAILYVOLUME', 'ADV'],
        'DTC': ['DAYS_TO_COVER', 'DAYSTOCOVER', 'DTC'],
    }

    def find_col_idx(target_names):
        for name in target_names:
            if name in header:
                return header.index(name)
        return -1

    idx_symbol = find_col_idx(col_map['SYMBOL'])
    idx_name = find_col_idx(col_map['NAME'])
    idx_market = find_col_idx(col_map['MARKET'])
    idx_current = find_col_idx(col_map['CURRENT_SI'])
    idx_prev = find_col_idx(col_map['PREV_SI'])
    idx_change = find_col_idx(col_map['CHANGE'])
    idx_adv = find_col_idx(col_map['ADV'])
    idx_dtc = find_col_idx(col_map['DTC'])

    # Parse data rows
    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue

        parts = line.split(delimiter)
        if len(parts) < max(idx_symbol, idx_current, 0) + 1:
            continue

        try:
            symbol = parts[idx_symbol].strip() if idx_symbol >= 0 else ""
            if not symbol or len(symbol) > 10:  # Skip invalid symbols
                continue

            # Parse numeric fields safely
            def parse_num(idx, default=None):
                if idx < 0 or idx >= len(parts):
                    return default
                val = parts[idx].strip().replace(',', '')
                if not val or val == '-' or val.upper() == 'NA':
                    return default
                try:
                    return float(val)
                except ValueError:
                    return default

            current_si = parse_num(idx_current)
            if current_si is None:
                continue

            record = {
                "symbol": symbol,
                "security_name": parts[idx_name].strip() if idx_name >= 0 and idx_name < len(parts) else "",
                "market": parts[idx_market].strip() if idx_market >= 0 and idx_market < len(parts) else "",
                "short_interest_shares": int(current_si),
                "previous_si_shares": int(parse_num(idx_prev, 0)),
                "si_change_shares": int(parse_num(idx_change, 0)),
                "avg_daily_volume": int(parse_num(idx_adv, 0)) if parse_num(idx_adv) else None,
                "days_to_cover": parse_num(idx_dtc),
                "settlement_date": settlement_date.isoformat(),
                "source": "FINRA",
            }

            # Compute change percentage if we have previous
            if record["previous_si_shares"] and record["previous_si_shares"] > 0:
                change_pct = ((record["short_interest_shares"] - record["previous_si_shares"])
                              / record["previous_si_shares"]) * 100
                record["si_change_pct"] = round(change_pct, 2)
            else:
                record["si_change_pct"] = None

            records.append(record)

        except (ValueError, IndexError) as e:
            continue  # Skip malformed rows

    return records


def load_cached_si_data(settlement_date: date) -> Optional[List[Dict[str, Any]]]:
    """Load previously downloaded and parsed SI data."""
    raw_file = RAW_DATA_DIR / f"{settlement_date.isoformat()}.txt"
    if raw_file.exists():
        return parse_finra_si_file(raw_file, settlement_date)
    return None


def get_available_settlement_dates() -> List[date]:
    """Get list of settlement dates we have cached data for."""
    dates = []
    if RAW_DATA_DIR.exists():
        for f in RAW_DATA_DIR.glob("*.txt"):
            try:
                d = date.fromisoformat(f.stem)
                dates.append(d)
            except ValueError:
                continue
    return sorted(dates, reverse=True)


def demonstration():
    """Demonstrate the FINRA SI feed."""
    print("=" * 70)
    print("FINRA SHORT INTEREST FEED v1.0.0 - DEMONSTRATION")
    print("=" * 70)
    print()

    as_of = date(2026, 1, 11)
    print(f"As-of date: {as_of}")

    # Get latest available settlement date
    latest_settlement = get_latest_available_settlement_date(as_of)
    print(f"Latest available settlement: {latest_settlement}")
    print(f"Estimated dissemination: {estimate_dissemination_date(latest_settlement)}")
    print()

    # Check cached data
    cached_dates = get_available_settlement_dates()
    if cached_dates:
        print(f"Cached settlement dates: {len(cached_dates)}")
        for d in cached_dates[:5]:
            print(f"  {d}")
    else:
        print("No cached data found")

    print()
    print("To download fresh data, run:")
    print("  python finra_short_interest_feed.py --download --settlement-date YYYY-MM-DD")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FINRA Short Interest Data Feed")
    parser.add_argument("--download", action="store_true", help="Download SI data")
    parser.add_argument("--settlement-date", type=str, help="Settlement date (YYYY-MM-DD)")
    parser.add_argument("--as-of-date", type=str, help="As-of date for PIT validation")
    parser.add_argument("--output", type=Path, help="Output directory")

    args = parser.parse_args()

    if args.download and args.settlement_date:
        settlement = date.fromisoformat(args.settlement_date)
        as_of = date.fromisoformat(args.as_of_date) if args.as_of_date else None

        print(f"Downloading SI data for settlement {settlement}...")
        result = download_finra_si_file(settlement, args.output, as_of)

        if result["success"]:
            print(f"Success! Saved to {result['file_path']}")
            print(f"SHA256: {result['sha256']}")
            print(f"Estimated records: {result['records_hint']}")
        else:
            print(f"Failed: {result['message']}")
    else:
        demonstration()
