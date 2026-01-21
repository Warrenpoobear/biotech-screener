"""
Sharadar CSV Schema Validator and Data Readiness Preflight

Strict contract for Sharadar SEP (Security End-of-day Prices) data.

SCHEMA CONTRACT:
  Required columns (case-insensitive, normalized):
    - ticker: string, uppercased, stripped
    - date: YYYY-MM-DD format
    - closeadj: Decimal, positive (adjusted close price)
    
  Column name mappings (auto-normalized):
    - "adj_close" -> "closeadj"
    - "close_adj" -> "closeadj"
    - "adjclose" -> "closeadj"
    
  Validation rules:
    - Reject non-positive prices
    - Reject malformed dates
    - Reject duplicate (ticker, date) rows (keep last, log warning)
    - Parse closeadj to Decimal for precision

PREFLIGHT OUTPUT:
  data_readiness.json with:
    - date_coverage: min/max date in file
    - ticker_coverage: % of required trading days present per ticker
    - missingness_breakdown: by reason
    - trading_calendar: inferred from file
    - gate_passed: bool (coverage >= 80%)
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ============================================================================
# SCHEMA CONTRACT
# ============================================================================

# Required columns (normalized names)
REQUIRED_COLUMNS = {"ticker", "date", "closeadj"}

# Column name mappings (source -> normalized)
COLUMN_MAPPINGS = {
    # ticker variations
    "ticker": "ticker",
    "symbol": "ticker",
    "sym": "ticker",
    # date variations
    "date": "date",
    "trade_date": "date",
    "tradedate": "date",
    # closeadj variations
    "closeadj": "closeadj",
    "adj_close": "closeadj",
    "close_adj": "closeadj",
    "adjclose": "closeadj",
    "adjusted_close": "closeadj",
}

# Validation thresholds
MIN_COVERAGE_GATE = 0.80  # 80% coverage required to proceed


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

class SchemaValidationError(Exception):
    """Raised when CSV schema validation fails."""
    pass


def normalize_column_name(col: str) -> str:
    """Normalize column name to standard form."""
    col_lower = col.strip().lower().replace(" ", "_")
    return COLUMN_MAPPINGS.get(col_lower, col_lower)


def validate_and_load_csv(
    filepath: str,
    ticker_filter: Optional[List[str]] = None,
) -> Tuple[Dict[str, Dict[str, Decimal]], Dict[str, Any]]:
    """
    Validate and load Sharadar CSV with strict schema enforcement.
    
    Args:
        filepath: Path to CSV file
        ticker_filter: Optional list of tickers to include
        
    Returns:
        (data, diagnostics)
        - data: {ticker: {date: closeadj}}
        - diagnostics: validation results and warnings
    """
    diagnostics = {
        "filepath": str(filepath),
        "rows_read": 0,
        "rows_valid": 0,
        "rows_skipped": 0,
        "warnings": [],
        "errors": [],
        "column_mapping": {},
        "duplicates_found": 0,
        "invalid_prices": 0,
        "invalid_dates": 0,
        "tickers_loaded": 0,
    }
    
    path = Path(filepath)
    if not path.exists():
        raise SchemaValidationError(f"File not found: {filepath}")
    
    # Normalize ticker filter
    if ticker_filter:
        ticker_filter_set = {t.upper().strip() for t in ticker_filter}
    else:
        ticker_filter_set = None
    
    data: Dict[str, Dict[str, Decimal]] = defaultdict(dict)
    seen_pairs: Set[Tuple[str, str]] = set()
    
    # Read with proper encoding handling
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        
        if reader.fieldnames is None:
            raise SchemaValidationError("CSV has no header row")
        
        # Normalize column names
        original_cols = list(reader.fieldnames)
        normalized_cols = [normalize_column_name(c) for c in original_cols]
        
        diagnostics["column_mapping"] = dict(zip(original_cols, normalized_cols))
        
        # Check required columns
        normalized_set = set(normalized_cols)
        missing_cols = REQUIRED_COLUMNS - normalized_set
        if missing_cols:
            raise SchemaValidationError(
                f"Missing required columns: {missing_cols}. "
                f"Found: {original_cols}"
            )
        
        # Find column indices
        col_indices = {normalized_cols[i]: original_cols[i] for i in range(len(original_cols))}
        
        for row in reader:
            diagnostics["rows_read"] += 1
            
            try:
                # Extract and normalize values
                ticker = row[col_indices["ticker"]].upper().strip()
                date_str = row[col_indices["date"]].strip()
                price_str = row[col_indices["closeadj"]].strip()
                
                # Filter tickers
                if ticker_filter_set and ticker not in ticker_filter_set:
                    diagnostics["rows_skipped"] += 1
                    continue
                
                # Validate date
                try:
                    date.fromisoformat(date_str)
                except ValueError:
                    diagnostics["invalid_dates"] += 1
                    diagnostics["warnings"].append(f"Invalid date: {date_str} for {ticker}")
                    continue
                
                # Validate price
                try:
                    price = Decimal(price_str)
                    if price <= 0:
                        diagnostics["invalid_prices"] += 1
                        diagnostics["warnings"].append(f"Non-positive price: {price} for {ticker} on {date_str}")
                        continue
                except (InvalidOperation, ValueError):
                    diagnostics["invalid_prices"] += 1
                    diagnostics["warnings"].append(f"Invalid price: {price_str} for {ticker} on {date_str}")
                    continue
                
                # Check for duplicates
                pair = (ticker, date_str)
                if pair in seen_pairs:
                    diagnostics["duplicates_found"] += 1
                    diagnostics["warnings"].append(f"Duplicate: {ticker} on {date_str} (keeping last)")
                seen_pairs.add(pair)
                
                # Store (overwrites duplicates - keeps last)
                data[ticker][date_str] = price
                diagnostics["rows_valid"] += 1
                
            except KeyError as e:
                diagnostics["errors"].append(f"Missing column in row: {e}")
                continue
    
    diagnostics["tickers_loaded"] = len(data)
    
    # Truncate warnings if too many
    if len(diagnostics["warnings"]) > 100:
        diagnostics["warnings"] = diagnostics["warnings"][:100] + [
            f"... and {len(diagnostics['warnings']) - 100} more warnings"
        ]
    
    return dict(data), diagnostics


# ============================================================================
# TRADING CALENDAR
# ============================================================================

def infer_trading_calendar(data: Dict[str, Dict[str, Decimal]]) -> Dict[str, Any]:
    """
    Infer trading calendar from price data.
    
    Uses global trading calendar (dates present across all tickers).
    """
    if not data:
        return {
            "min_date": None,
            "max_date": None,
            "trading_days": [],
            "n_trading_days": 0,
        }
    
    # Collect all dates across all tickers
    all_dates: Set[str] = set()
    for ticker_dates in data.values():
        all_dates.update(ticker_dates.keys())
    
    sorted_dates = sorted(all_dates)
    
    return {
        "min_date": sorted_dates[0] if sorted_dates else None,
        "max_date": sorted_dates[-1] if sorted_dates else None,
        "trading_days": sorted_dates,
        "n_trading_days": len(sorted_dates),
    }


def compute_next_trading_day(
    calendar: Dict[str, Any],
    from_date: str,
    offset_days: int = 1,
) -> Optional[str]:
    """
    Find next trading day from a given date.
    
    Uses global trading calendar (dates present in the file overall).
    """
    trading_days = calendar.get("trading_days", [])
    if not trading_days:
        return None
    
    # Find first trading day >= from_date + offset
    target = date.fromisoformat(from_date) + timedelta(days=offset_days)
    target_str = target.isoformat()
    
    for td in trading_days:
        if td >= target_str:
            return td
    
    return None


# ============================================================================
# DATA READINESS PREFLIGHT
# ============================================================================

def compute_ticker_coverage(
    data: Dict[str, Dict[str, Decimal]],
    calendar: Dict[str, Any],
    start_date: str,
    end_date: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute per-ticker coverage over a date range.
    """
    trading_days = calendar.get("trading_days", [])
    
    # Filter trading days to range
    required_days = [d for d in trading_days if start_date <= d <= end_date]
    n_required = len(required_days)
    
    coverage = {}
    
    for ticker, prices in data.items():
        available_days = [d for d in prices.keys() if start_date <= d <= end_date]
        n_available = len(available_days)
        
        coverage[ticker] = {
            "required_days": n_required,
            "available_days": n_available,
            "coverage_pct": round(n_available / n_required * 100, 1) if n_required > 0 else 0,
            "missing_days": n_required - n_available,
        }
    
    return coverage


def run_data_readiness_preflight(
    filepath: str,
    ticker_filter: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run data readiness preflight before any backtest math.
    
    Returns:
        {
            "gate_passed": bool,
            "gate_reason": str,
            "schema_validation": {...},
            "date_coverage": {"min": str, "max": str},
            "ticker_coverage": {...},
            "missingness_breakdown": {...},
            "trading_calendar": {...},
        }
    """
    # Step 1: Validate and load
    try:
        data, schema_diagnostics = validate_and_load_csv(filepath, ticker_filter)
    except SchemaValidationError as e:
        return {
            "gate_passed": False,
            "gate_reason": f"Schema validation failed: {e}",
            "schema_validation": {"error": str(e)},
        }
    
    if not data:
        return {
            "gate_passed": False,
            "gate_reason": "No valid data rows loaded",
            "schema_validation": schema_diagnostics,
        }
    
    # Step 2: Infer trading calendar
    calendar = infer_trading_calendar(data)
    
    # Use date range from data if not specified
    if start_date is None:
        start_date = calendar["min_date"]
    if end_date is None:
        end_date = calendar["max_date"]
    
    # Step 3: Compute ticker coverage
    ticker_cov = compute_ticker_coverage(data, calendar, start_date, end_date)
    
    # Step 4: Compute overall metrics
    if ticker_cov:
        coverage_pcts = [t["coverage_pct"] for t in ticker_cov.values()]
        avg_coverage = sum(coverage_pcts) / len(coverage_pcts)
        min_coverage = min(coverage_pcts)
        
        # Tickers below threshold
        low_coverage_tickers = [
            ticker for ticker, stats in ticker_cov.items()
            if stats["coverage_pct"] < MIN_COVERAGE_GATE * 100
        ]
    else:
        avg_coverage = 0
        min_coverage = 0
        low_coverage_tickers = []
    
    # Step 5: Missingness breakdown
    missingness = {
        "ticker_absent_entirely": len(ticker_filter or []) - len(data) if ticker_filter else 0,
        "rows_with_invalid_price": schema_diagnostics["invalid_prices"],
        "rows_with_invalid_date": schema_diagnostics["invalid_dates"],
        "duplicate_rows_found": schema_diagnostics["duplicates_found"],
        "low_coverage_tickers": len(low_coverage_tickers),
    }
    
    # Step 6: Gate decision
    gate_passed = avg_coverage >= MIN_COVERAGE_GATE * 100
    
    if not gate_passed:
        gate_reason = f"Average coverage {avg_coverage:.1f}% < {MIN_COVERAGE_GATE*100:.0f}% threshold"
    elif low_coverage_tickers:
        gate_reason = f"Passed but {len(low_coverage_tickers)} tickers below threshold"
        gate_passed = True  # Still pass, but warn
    else:
        gate_reason = "All coverage checks passed"
    
    return {
        "gate_passed": gate_passed,
        "gate_reason": gate_reason,
        "schema_validation": schema_diagnostics,
        "date_coverage": {
            "min_date": calendar["min_date"],
            "max_date": calendar["max_date"],
            "n_trading_days": calendar["n_trading_days"],
            "requested_range": {"start": start_date, "end": end_date},
        },
        "ticker_coverage": {
            "n_tickers": len(ticker_cov),
            "avg_coverage_pct": round(avg_coverage, 1),
            "min_coverage_pct": round(min_coverage, 1),
            "low_coverage_tickers": low_coverage_tickers[:10],  # Top 10
            "per_ticker": ticker_cov,
        },
        "missingness_breakdown": missingness,
        "trading_calendar": {
            "type": "global",  # Using global calendar, not per-ticker
            "n_days": calendar["n_trading_days"],
            "sample_days": calendar["trading_days"][:5] + ["..."] + calendar["trading_days"][-5:] 
                           if len(calendar["trading_days"]) > 10 else calendar["trading_days"],
        },
    }


def print_preflight_report(preflight: Dict[str, Any]) -> None:
    """Print formatted preflight report."""
    print("\n" + "=" * 70)
    print("DATA READINESS PREFLIGHT")
    print("=" * 70)
    
    # Gate status
    if preflight["gate_passed"]:
        print(f"\n✓ GATE PASSED: {preflight['gate_reason']}")
    else:
        print(f"\n✗ GATE FAILED: {preflight['gate_reason']}")
    
    # Schema validation
    schema = preflight.get("schema_validation", {})
    print(f"\nSchema Validation:")
    print(f"  Rows read:     {schema.get('rows_read', 'N/A')}")
    print(f"  Rows valid:    {schema.get('rows_valid', 'N/A')}")
    print(f"  Tickers loaded:{schema.get('tickers_loaded', 'N/A')}")
    
    if schema.get("column_mapping"):
        print(f"  Column mapping: {schema['column_mapping']}")
    
    # Date coverage
    date_cov = preflight.get("date_coverage", {})
    print(f"\nDate Coverage:")
    print(f"  Min date:      {date_cov.get('min_date', 'N/A')}")
    print(f"  Max date:      {date_cov.get('max_date', 'N/A')}")
    print(f"  Trading days:  {date_cov.get('n_trading_days', 'N/A')}")
    
    # Ticker coverage
    ticker_cov = preflight.get("ticker_coverage", {})
    print(f"\nTicker Coverage:")
    print(f"  Tickers:       {ticker_cov.get('n_tickers', 'N/A')}")
    print(f"  Avg coverage:  {ticker_cov.get('avg_coverage_pct', 'N/A')}%")
    print(f"  Min coverage:  {ticker_cov.get('min_coverage_pct', 'N/A')}%")
    
    low_cov = ticker_cov.get("low_coverage_tickers", [])
    if low_cov:
        print(f"  ⚠ Low coverage: {low_cov[:5]}{'...' if len(low_cov) > 5 else ''}")
    
    # Missingness
    miss = preflight.get("missingness_breakdown", {})
    print(f"\nMissingness Breakdown:")
    print(f"  Tickers absent:    {miss.get('ticker_absent_entirely', 0)}")
    print(f"  Invalid prices:    {miss.get('rows_with_invalid_price', 0)}")
    print(f"  Invalid dates:     {miss.get('rows_with_invalid_date', 0)}")
    print(f"  Duplicates:        {miss.get('duplicate_rows_found', 0)}")
    
    # Trading calendar
    cal = preflight.get("trading_calendar", {})
    print(f"\nTrading Calendar:")
    print(f"  Type: {cal.get('type', 'unknown')} (dates present in file)")
    print(f"  Days: {cal.get('n_days', 0)}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example usage
    import sys
    
    filepath = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/biotech_screener/data/daily_prices.csv"
    
    # Note: Removed delisted tickers (SGEN acquired by Pfizer, BLUE acquired by Carlyle/SK Capital)
    BIOTECH_UNIVERSE = [
        "AMGN", "GILD", "VRTX", "REGN", "BIIB",
        "ALNY", "BMRN", "INCY", "EXEL", "JAZZ",
        "MRNA", "BNTX", "IONS", "SRPT", "RARE",
        "FOLD", "ACAD", "HALO", "KRTX", "LEGN",
        "IMVT", "ARWR", "PCVX", "BEAM", "EDIT",
    ]
    
    preflight = run_data_readiness_preflight(
        filepath,
        ticker_filter=BIOTECH_UNIVERSE,
        start_date="2023-01-01",
        end_date="2024-12-31",
    )
    
    print_preflight_report(preflight)
    
    # Save to file
    output_path = Path("/home/claude/biotech_screener/output/data_readiness.json")
    with open(output_path, "w") as f:
        # Remove per-ticker detail for readability
        output = {k: v for k, v in preflight.items()}
        output["ticker_coverage"] = {
            k: v for k, v in preflight["ticker_coverage"].items() 
            if k != "per_ticker"
        }
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {output_path}")
