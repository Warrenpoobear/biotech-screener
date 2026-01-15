#!/usr/bin/env python3
"""
finra_short_volume_feed.py

Downloads and parses FINRA Daily Short Sale Volume data.

This provides daily short selling activity (not positions), which complements
the bi-weekly short interest data with more responsive signals.

Data Source: https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files

Design Philosophy (Governed Over Smart):
- DETERMINISTIC: No datetime.now() in scoring - uses as_of_date
- STDLIB-ONLY: No external dependencies (uses urllib)
- PIT DISCIPLINE: Data available T+1 (next business day)
- FAIL LOUDLY: Clear error states
- AUDITABLE: SHA256 hashes of raw data

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import csv
import hashlib
import io
import json
import ssl
import urllib.request
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Any, Optional


__version__ = "1.0.0"
__author__ = "Wake Robin Capital Management"


# FINRA daily short sale volume download base
# Format: CNMSshvol{YYYYMMDD}.txt
FINRA_SV_DOWNLOAD_BASE = "https://cdn.finra.org/equity/regsho/daily"

# Data availability lag (T+1 typically)
AVAILABILITY_LAG_DAYS = 1

# Raw data storage paths
RAW_DATA_DIR = Path("raw/finra_daily_short_volume")
META_DATA_DIR = Path("meta/finra_daily_short_volume")


def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of data."""
    return hashlib.sha256(data).hexdigest()


def is_business_day(d: date) -> bool:
    """Check if date is a business day (Mon-Fri)."""
    return d.weekday() < 5


def prev_business_day(d: date) -> date:
    """Get previous business day."""
    d = d - timedelta(days=1)
    while not is_business_day(d):
        d -= timedelta(days=1)
    return d


def get_available_trade_date(as_of_date: date) -> date:
    """
    Get the most recent trade date with available short volume data.

    PIT DISCIPLINE: Data for trade date T is available on T+1.
    """
    # As of as_of_date, we can see data up to (as_of_date - 1 business day)
    latest = prev_business_day(as_of_date)
    return latest


def fetch_url(url: str, timeout: int = 30) -> bytes:
    """Fetch URL content with proper SSL handling."""
    ctx = ssl.create_default_context()

    request = urllib.request.Request(
        url,
        headers={
            'User-Agent': 'Mozilla/5.0 (compatible; WakeRobinDataFeed/1.0)',
            'Accept': 'text/plain,*/*;q=0.8',
        }
    )

    with urllib.request.urlopen(request, timeout=timeout, context=ctx) as response:
        return response.read()


def download_finra_short_volume(
    trade_date: date,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Download FINRA daily short sale volume file for a specific trade date.

    Args:
        trade_date: The trading date to fetch
        output_dir: Directory to save raw files

    Returns:
        Dict with download result and metadata
    """
    output_dir = output_dir or RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = trade_date.strftime("%Y%m%d")

    # FINRA file naming patterns
    url_patterns = [
        f"{FINRA_SV_DOWNLOAD_BASE}/CNMSshvol{date_str}.txt",
        f"{FINRA_SV_DOWNLOAD_BASE}/FNRAshvol{date_str}.txt",
        f"{FINRA_SV_DOWNLOAD_BASE}/REGSHOshvol{date_str}.txt",
    ]

    last_error = None
    for url in url_patterns:
        try:
            data = fetch_url(url)

            # Save raw file
            output_file = output_dir / f"{trade_date.isoformat()}.txt"
            with open(output_file, 'wb') as f:
                f.write(data)

            # Compute hash
            data_hash = compute_sha256(data)

            # Save metadata
            meta_dir = META_DATA_DIR
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_file = meta_dir / f"{trade_date.isoformat()}.json"

            meta = {
                "trade_date": trade_date.isoformat(),
                "available_date": (trade_date + timedelta(days=AVAILABILITY_LAG_DAYS)).isoformat(),
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
                "trade_date": trade_date.isoformat(),
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
        "message": f"Could not download short volume for {trade_date}: {last_error}",
        "trade_date": trade_date.isoformat(),
        "tried_urls": url_patterns,
    }


def parse_finra_short_volume_file(
    file_path: Path,
    trade_date: date
) -> List[Dict[str, Any]]:
    """
    Parse a FINRA daily short volume file into records.

    FINRA short volume file format (pipe-delimited):
    Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market

    Returns:
        List of parsed short volume records
    """
    records = []

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Detect delimiter
    if '|' in content[:500]:
        delimiter = '|'
    else:
        delimiter = ','

    lines = content.strip().split('\n')

    # Find header row
    header_idx = 0
    for i, line in enumerate(lines):
        lower = line.lower()
        if 'symbol' in lower or 'date' in lower:
            header_idx = i
            break

    if header_idx >= len(lines) - 1:
        return records

    header = [h.strip().upper() for h in lines[header_idx].split(delimiter)]

    # Column mapping
    def find_col(names):
        for n in names:
            if n in header:
                return header.index(n)
        return -1

    idx_date = find_col(['DATE', 'TRADEDATE'])
    idx_symbol = find_col(['SYMBOL', 'TICKER'])
    idx_short = find_col(['SHORTVOLUME', 'SHORT_VOLUME', 'SHORTVOL'])
    idx_exempt = find_col(['SHORTEXEMPTVOLUME', 'SHORT_EXEMPT_VOLUME', 'EXEMPTVOLUME'])
    idx_total = find_col(['TOTALVOLUME', 'TOTAL_VOLUME', 'TOTALVOL'])
    idx_market = find_col(['MARKET', 'MARKETCENTER', 'EXCHANGE'])

    for line in lines[header_idx + 1:]:
        if not line.strip():
            continue

        parts = line.split(delimiter)
        if len(parts) < 3:
            continue

        try:
            symbol = parts[idx_symbol].strip() if idx_symbol >= 0 else ""
            if not symbol or len(symbol) > 10:
                continue

            def parse_int(idx, default=0):
                if idx < 0 or idx >= len(parts):
                    return default
                val = parts[idx].strip().replace(',', '')
                if not val or val == '-':
                    return default
                try:
                    return int(float(val))
                except ValueError:
                    return default

            short_vol = parse_int(idx_short)
            exempt_vol = parse_int(idx_exempt)
            total_vol = parse_int(idx_total)

            # Skip if no meaningful data
            if total_vol == 0:
                continue

            # Compute short volume ratio
            short_vol_ratio = (short_vol / total_vol) if total_vol > 0 else 0

            record = {
                "symbol": symbol,
                "trade_date": trade_date.isoformat(),
                "short_volume": short_vol,
                "short_exempt_volume": exempt_vol,
                "total_volume": total_vol,
                "short_vol_ratio": round(short_vol_ratio, 4),
                "market": parts[idx_market].strip() if idx_market >= 0 and idx_market < len(parts) else "",
                "source": "FINRA",
            }

            records.append(record)

        except (ValueError, IndexError):
            continue

    return records


def load_cached_short_volume(trade_date: date) -> Optional[List[Dict[str, Any]]]:
    """Load previously downloaded short volume data."""
    raw_file = RAW_DATA_DIR / f"{trade_date.isoformat()}.txt"
    if raw_file.exists():
        return parse_finra_short_volume_file(raw_file, trade_date)
    return None


def get_available_trade_dates() -> List[date]:
    """Get list of trade dates we have cached data for."""
    dates = []
    if RAW_DATA_DIR.exists():
        for f in RAW_DATA_DIR.glob("*.txt"):
            try:
                d = date.fromisoformat(f.stem)
                dates.append(d)
            except ValueError:
                continue
    return sorted(dates, reverse=True)


def compute_short_volume_stats(
    records_by_date: Dict[date, List[Dict[str, Any]]],
    symbol: str,
    as_of_date: date,
    lookback_days: int = 20
) -> Dict[str, Any]:
    """
    Compute rolling short volume statistics for a symbol.

    Args:
        records_by_date: Dict mapping trade_date -> list of records
        symbol: Symbol to compute stats for
        as_of_date: As-of date
        lookback_days: Number of trading days to look back

    Returns:
        Dict with short volume statistics
    """
    symbol = symbol.upper()

    # Collect recent data points
    data_points = []
    dates_checked = 0
    current = prev_business_day(as_of_date)

    while dates_checked < lookback_days and len(data_points) < lookback_days:
        if current in records_by_date:
            for rec in records_by_date[current]:
                if rec["symbol"].upper() == symbol:
                    data_points.append({
                        "date": current,
                        "short_vol_ratio": rec["short_vol_ratio"],
                        "short_volume": rec["short_volume"],
                        "total_volume": rec["total_volume"],
                    })
                    break
        current = prev_business_day(current)
        dates_checked += 1

    if not data_points:
        return {
            "symbol": symbol,
            "data_points": 0,
            "short_vol_ratio_latest": None,
            "short_vol_ratio_avg": None,
            "short_vol_ratio_std": None,
            "short_vol_ratio_zscore": None,
        }

    ratios = [dp["short_vol_ratio"] for dp in data_points]
    latest = ratios[0] if ratios else None

    avg_ratio = sum(ratios) / len(ratios) if ratios else None

    # Compute std dev
    if len(ratios) > 1 and avg_ratio is not None:
        variance = sum((r - avg_ratio) ** 2 for r in ratios) / len(ratios)
        std_ratio = variance ** 0.5
    else:
        std_ratio = None

    # Compute z-score of latest vs lookback
    zscore = None
    if latest is not None and avg_ratio is not None and std_ratio and std_ratio > 0:
        zscore = (latest - avg_ratio) / std_ratio

    return {
        "symbol": symbol,
        "data_points": len(data_points),
        "short_vol_ratio_latest": latest,
        "short_vol_ratio_avg": round(avg_ratio, 4) if avg_ratio else None,
        "short_vol_ratio_std": round(std_ratio, 4) if std_ratio else None,
        "short_vol_ratio_zscore": round(zscore, 2) if zscore else None,
        "lookback_days": lookback_days,
    }


def demonstration():
    """Demonstrate the FINRA short volume feed."""
    print("=" * 70)
    print("FINRA DAILY SHORT VOLUME FEED v1.0.0 - DEMONSTRATION")
    print("=" * 70)
    print()

    as_of = date(2026, 1, 11)
    print(f"As-of date: {as_of}")

    available_trade_date = get_available_trade_date(as_of)
    print(f"Latest available trade date: {available_trade_date}")
    print()

    cached_dates = get_available_trade_dates()
    if cached_dates:
        print(f"Cached trade dates: {len(cached_dates)}")
        for d in cached_dates[:5]:
            print(f"  {d}")
    else:
        print("No cached data found")

    print()
    print("To download fresh data, run:")
    print("  python finra_short_volume_feed.py --download --trade-date YYYY-MM-DD")
    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FINRA Daily Short Volume Feed")
    parser.add_argument("--download", action="store_true", help="Download short volume data")
    parser.add_argument("--trade-date", type=str, help="Trade date (YYYY-MM-DD)")
    parser.add_argument("--output", type=Path, help="Output directory")

    args = parser.parse_args()

    if args.download and args.trade_date:
        trade = date.fromisoformat(args.trade_date)
        print(f"Downloading short volume for trade date {trade}...")
        result = download_finra_short_volume(trade, args.output)

        if result["success"]:
            print(f"Success! Saved to {result['file_path']}")
            print(f"SHA256: {result['sha256']}")
        else:
            print(f"Failed: {result['message']}")
    else:
        demonstration()
