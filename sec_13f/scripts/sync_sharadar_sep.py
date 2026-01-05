"""
Sharadar SEP Daily Sync Script (Windows Compatible)

Maintains a local mirror of Sharadar SEP (Security End-of-day Prices) data
with incremental daily updates.

REQUIREMENTS:
  pip install requests

SETUP:
  1. Set environment variable: NASDAQ_DATA_LINK_API_KEY=your_api_key
  2. Or create: data/config/nasdaq_api_key.txt (one line, just the key)

USAGE:
  python scripts/sync_sharadar_sep.py              # Incremental sync
  python scripts/sync_sharadar_sep.py --full       # Full rebuild (slow)
  python scripts/sync_sharadar_sep.py --days 30    # Sync last N days

OUTPUT:
  data/raw/sharadar_sep/ingest_YYYY-MM-DD.csv      # Immutable raw slices
  data/sharadar_sep.csv                            # Curated (rebuilt from raw)
  data/curated/sharadar/state.json                 # Sync state
  data/curated/sharadar/ingest_manifest.json       # Audit manifest

SCHEDULING (Windows Task Scheduler):
  Run: sync_sharadar.bat (created alongside this script)
  Trigger: Daily at 6:00 AM (after market data is finalized)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backtest.data_readiness import validate_and_load_csv, SchemaValidationError

# ============================================================================
# CONFIGURATION
# ============================================================================

# Nasdaq Data Link API endpoint for Sharadar SEP
NASDAQ_API_BASE = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/SEP"

# Default overlap days for incremental sync (safety margin)
DEFAULT_OVERLAP_DAYS = 5

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "sharadar_sep"
CURATED_DIR = DATA_DIR / "curated" / "sharadar"
CONFIG_DIR = DATA_DIR / "config"

STATE_FILE = CURATED_DIR / "state.json"
MANIFEST_FILE = CURATED_DIR / "ingest_manifest.json"
CURATED_CSV = DATA_DIR / "sharadar_sep.csv"

# API key locations (checked in order)
API_KEY_ENV_VAR = "NASDAQ_DATA_LINK_API_KEY"
API_KEY_FILE = CONFIG_DIR / "nasdaq_api_key.txt"


# ============================================================================
# API KEY MANAGEMENT
# ============================================================================

def get_api_key() -> str:
    """Get Nasdaq Data Link API key from environment or file."""
    # Try environment variable first
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if api_key:
        return api_key.strip()
    
    # Try file
    if API_KEY_FILE.exists():
        with open(API_KEY_FILE, "r") as f:
            api_key = f.read().strip()
            if api_key:
                return api_key
    
    raise ValueError(
        f"Nasdaq Data Link API key not found.\n"
        f"Set environment variable: {API_KEY_ENV_VAR}\n"
        f"Or create file: {API_KEY_FILE}"
    )


# ============================================================================
# STATE MANAGEMENT
# ============================================================================

def load_state() -> Dict[str, Any]:
    """Load sync state from file."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {
        "last_sync_date": None,
        "last_data_date": None,
        "last_data_hash": None,
        "total_rows": 0,
        "syncs": [],
    }


def save_state(state: Dict[str, Any]) -> None:
    """Save sync state to file."""
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ============================================================================
# NASDAQ DATA LINK API
# ============================================================================

def fetch_sep_data(
    api_key: str,
    date_gte: Optional[str] = None,
    date_lte: Optional[str] = None,
    tickers: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch SEP data from Nasdaq Data Link API.
    
    Args:
        api_key: Nasdaq Data Link API key
        date_gte: Filter dates >= this (YYYY-MM-DD)
        date_lte: Filter dates <= this (YYYY-MM-DD)
        tickers: Optional list of tickers to filter
        
    Returns:
        List of row dictionaries
    """
    all_rows = []
    cursor = None
    page = 0
    
    while True:
        # Build URL with parameters
        params = [f"api_key={api_key}"]
        
        if date_gte:
            params.append(f"date.gte={date_gte}")
        if date_lte:
            params.append(f"date.lte={date_lte}")
        if tickers:
            params.append(f"ticker={','.join(tickers)}")
        if cursor:
            params.append(f"qopts.cursor_id={cursor}")
        
        # Select only needed columns
        params.append("qopts.columns=ticker,date,closeadj")
        
        url = f"{NASDAQ_API_BASE}.json?{'&'.join(params)}"
        
        print(f"  Fetching page {page + 1}...", end=" ", flush=True)
        
        try:
            req = Request(url)
            with urlopen(req, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429:
                print(f"\n⚠ Rate limited. Wait and retry.")
                raise
            elif e.code == 401:
                print(f"\n✗ Invalid API key")
                raise ValueError("Invalid Nasdaq Data Link API key")
            else:
                print(f"\n✗ HTTP error: {e.code}")
                raise
        except URLError as e:
            print(f"\n✗ Network error: {e.reason}")
            raise
        
        # Parse response
        datatable = data.get("datatable", {})
        columns = [c["name"] for c in datatable.get("columns", [])]
        rows = datatable.get("data", [])
        
        # Convert to dicts
        for row in rows:
            all_rows.append(dict(zip(columns, row)))
        
        print(f"{len(rows)} rows")
        page += 1
        
        # Check for more pages
        meta = data.get("meta", {})
        cursor = meta.get("next_cursor_id")
        
        if not cursor or not rows:
            break
    
    return all_rows


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def compute_file_hash(filepath: Path) -> str:
    """Compute SHA256 hash of file contents."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def write_raw_slice(rows: List[Dict[str, Any]], ingest_date: str) -> Path:
    """Write raw data slice (immutable)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    filepath = RAW_DIR / f"ingest_{ingest_date}.csv"
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "date", "closeadj"])
        writer.writeheader()
        
        # Sort for determinism
        sorted_rows = sorted(rows, key=lambda x: (x["ticker"], x["date"]))
        writer.writerows(sorted_rows)
    
    return filepath


def rebuild_curated_csv() -> Dict[str, Any]:
    """
    Rebuild curated CSV from all raw slices.
    
    Deduplicates by (ticker, date), keeping the most recent ingest.
    """
    # Collect all raw slices
    raw_files = sorted(RAW_DIR.glob("ingest_*.csv"))
    
    if not raw_files:
        return {"error": "No raw slices found", "rows": 0}
    
    # Load and dedupe (later ingests win)
    data: Dict[tuple, Dict] = {}  # (ticker, date) -> row
    
    for raw_file in raw_files:
        with open(raw_file, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["ticker"], row["date"])
                data[key] = row
    
    # Write curated CSV
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(CURATED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "date", "closeadj"])
        writer.writeheader()
        
        # Sort for determinism
        for key in sorted(data.keys()):
            writer.writerow(data[key])
    
    # Compute stats
    dates = [row["date"] for row in data.values()]
    tickers = set(row["ticker"] for row in data.values())
    
    return {
        "rows": len(data),
        "tickers": len(tickers),
        "min_date": min(dates) if dates else None,
        "max_date": max(dates) if dates else None,
        "raw_slices": len(raw_files),
        "file_hash": compute_file_hash(CURATED_CSV),
    }


def update_manifest(
    ingest_date: str,
    rows_fetched: int,
    curated_stats: Dict[str, Any],
    sync_type: str,
) -> None:
    """Update ingest manifest with audit info."""
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing manifest
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {"ingests": []}
    
    # Add this ingest
    manifest["ingests"].append({
        "ingest_date": ingest_date,
        "sync_type": sync_type,
        "rows_fetched": rows_fetched,
        "timestamp": datetime.now().isoformat(),
    })
    
    # Update summary
    manifest["curated"] = curated_stats
    manifest["last_updated"] = datetime.now().isoformat()
    
    # Keep only last 100 ingests
    manifest["ingests"] = manifest["ingests"][-100:]
    
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


# ============================================================================
# SYNC OPERATIONS
# ============================================================================

def sync_incremental(
    api_key: str,
    overlap_days: int = DEFAULT_OVERLAP_DAYS,
    tickers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Incremental sync: fetch only new/recent data.
    """
    state = load_state()
    ingest_date = date.today().isoformat()
    
    # Determine start date
    if state["last_data_date"]:
        last_date = date.fromisoformat(state["last_data_date"])
        start_date = (last_date - timedelta(days=overlap_days)).isoformat()
    else:
        # First sync: get last 2 years
        start_date = (date.today() - timedelta(days=730)).isoformat()
    
    print(f"\n[Incremental Sync]")
    print(f"  Start date: {start_date}")
    print(f"  Ingest date: {ingest_date}")
    
    # Fetch data
    rows = fetch_sep_data(api_key, date_gte=start_date, tickers=tickers)
    
    if not rows:
        print("  No new data found.")
        return {"status": "no_data", "rows": 0}
    
    print(f"  Total rows fetched: {len(rows)}")
    
    # Write raw slice
    raw_file = write_raw_slice(rows, ingest_date)
    print(f"  Raw slice: {raw_file}")
    
    # Rebuild curated
    curated_stats = rebuild_curated_csv()
    print(f"  Curated CSV: {CURATED_CSV}")
    print(f"    Rows: {curated_stats['rows']}")
    print(f"    Tickers: {curated_stats['tickers']}")
    print(f"    Date range: {curated_stats['min_date']} -> {curated_stats['max_date']}")
    
    # Update state
    state["last_sync_date"] = ingest_date
    state["last_data_date"] = curated_stats["max_date"]
    state["last_data_hash"] = curated_stats["file_hash"]
    state["total_rows"] = curated_stats["rows"]
    state["syncs"].append({
        "date": ingest_date,
        "type": "incremental",
        "rows_fetched": len(rows),
    })
    state["syncs"] = state["syncs"][-50:]  # Keep last 50
    save_state(state)
    
    # Update manifest
    update_manifest(ingest_date, len(rows), curated_stats, "incremental")
    
    # Validate with schema validator
    print("\n[Validation]")
    try:
        _, diagnostics = validate_and_load_csv(str(CURATED_CSV))
        print(f"  OK Schema valid")
        print(f"  Rows valid: {diagnostics['rows_valid']}")
        if diagnostics.get("warnings"):
            print(f"  Warnings: {len(diagnostics['warnings'])}")
    except SchemaValidationError as e:
        print(f"  FAIL Schema validation failed: {e}")
    
    return {
        "status": "success",
        "rows_fetched": len(rows),
        "curated": curated_stats,
    }


def sync_full(
    api_key: str,
    start_date: str = "2015-01-01",
    tickers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Full sync: rebuild everything from scratch.
    """
    ingest_date = date.today().isoformat()
    
    print(f"\n[Full Sync]")
    print(f"  Start date: {start_date}")
    print(f"  Ingest date: {ingest_date}")
    print(f"  WARNING: This may take several minutes...")
    
    # Fetch all data
    rows = fetch_sep_data(api_key, date_gte=start_date, tickers=tickers)
    
    if not rows:
        print("  No data found.")
        return {"status": "no_data", "rows": 0}
    
    print(f"  Total rows fetched: {len(rows)}")
    
    # Clear existing raw slices
    if RAW_DIR.exists():
        for f in RAW_DIR.glob("ingest_*.csv"):
            f.unlink()
        print(f"  Cleared existing raw slices")
    
    # Write single raw slice
    raw_file = write_raw_slice(rows, ingest_date)
    print(f"  Raw slice: {raw_file}")
    
    # Rebuild curated
    curated_stats = rebuild_curated_csv()
    print(f"  Curated CSV: {CURATED_CSV}")
    print(f"    Rows: {curated_stats['rows']}")
    print(f"    Tickers: {curated_stats['tickers']}")
    print(f"    Date range: {curated_stats['min_date']} -> {curated_stats['max_date']}")
    
    # Reset state
    state = {
        "last_sync_date": ingest_date,
        "last_data_date": curated_stats["max_date"],
        "last_data_hash": curated_stats["file_hash"],
        "total_rows": curated_stats["rows"],
        "syncs": [{
            "date": ingest_date,
            "type": "full",
            "rows_fetched": len(rows),
        }],
    }
    save_state(state)
    
    # Update manifest
    update_manifest(ingest_date, len(rows), curated_stats, "full")
    
    return {
        "status": "success",
        "rows_fetched": len(rows),
        "curated": curated_stats,
    }


def sync_days(
    api_key: str,
    days: int,
    tickers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Sync last N days.
    """
    start_date = (date.today() - timedelta(days=days)).isoformat()
    ingest_date = date.today().isoformat()
    
    print(f"\n[Sync Last {days} Days]")
    print(f"  Start date: {start_date}")
    print(f"  Ingest date: {ingest_date}")
    
    # Fetch data
    rows = fetch_sep_data(api_key, date_gte=start_date, tickers=tickers)
    
    if not rows:
        print("  No data found.")
        return {"status": "no_data", "rows": 0}
    
    print(f"  Total rows fetched: {len(rows)}")
    
    # Write raw slice
    raw_file = write_raw_slice(rows, ingest_date)
    print(f"  Raw slice: {raw_file}")
    
    # Rebuild curated
    curated_stats = rebuild_curated_csv()
    print(f"  Curated CSV: {CURATED_CSV}")
    
    # Update state
    state = load_state()
    state["last_sync_date"] = ingest_date
    state["last_data_date"] = curated_stats["max_date"]
    state["last_data_hash"] = curated_stats["file_hash"]
    state["total_rows"] = curated_stats["rows"]
    state["syncs"].append({
        "date": ingest_date,
        "type": f"days_{days}",
        "rows_fetched": len(rows),
    })
    state["syncs"] = state["syncs"][-50:]
    save_state(state)
    
    # Update manifest
    update_manifest(ingest_date, len(rows), curated_stats, f"days_{days}")
    
    return {
        "status": "success",
        "rows_fetched": len(rows),
        "curated": curated_stats,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sync Sharadar SEP data from Nasdaq Data Link"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full rebuild (slow, clears existing data)"
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Sync last N days"
    )
    parser.add_argument(
        "--start-date",
        default="2015-01-01",
        help="Start date for full sync (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--tickers",
        help="Comma-separated list of tickers to filter"
    )
    
    args = parser.parse_args()
    
    # Parse tickers
    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    
    print("=" * 60)
    print("SHARADAR SEP SYNC")
    print("=" * 60)
    
    # Get API key
    try:
        api_key = get_api_key()
        print(f"API key: {'*' * 8}{api_key[-4:]}")
    except ValueError as e:
        print(f"FAIL {e}")
        sys.exit(1)
    
    # Run sync
    try:
        if args.full:
            result = sync_full(api_key, args.start_date, tickers)
        elif args.days:
            result = sync_days(api_key, args.days, tickers)
        else:
            result = sync_incremental(api_key, tickers=tickers)
    except Exception as e:
        print(f"\nFAIL Sync failed: {e}")
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    if result["status"] == "success":
        print("OK SYNC COMPLETE")
        print(f"  Rows fetched: {result['rows_fetched']}")
        print(f"  Curated file: {CURATED_CSV}")
        print(f"  Ready for backtest:")
        print(f"    python scripts/run_sharadar_backtest.py --prices {CURATED_CSV}")
    else:
        print("WARNING SYNC COMPLETE (no new data)")
    print("=" * 60)


if __name__ == "__main__":
    main()
