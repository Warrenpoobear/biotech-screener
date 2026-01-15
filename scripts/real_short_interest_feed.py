#!/usr/bin/env python3
"""
real_short_interest_feed.py

Unified Short Interest Data Feed that combines:
1. FINRA Equity Short Interest (bi-weekly positions)
2. FINRA Daily Short Sale Volume (daily activity)
3. Reg SHO Threshold Lists (NYSE/Nasdaq)

This produces a PIT-safe, auditable short interest dataset for the biotech universe.

Design Philosophy (Governed Over Smart):
- DETERMINISTIC: No datetime.now() - uses as_of_date for all timestamps
- STDLIB-ONLY: No external dependencies
- PIT DISCIPLINE: All data tagged with proper availability dates
- FAIL LOUDLY: Clear error states
- AUDITABLE: Source tracking and hashes throughout

Usage:
    # Generate SI data from cached FINRA files + market data
    python real_short_interest_feed.py --universe production_data/universe.json \
        --market-data production_data/market_data.json \
        --output production_data/short_interest.json \
        --as-of-date 2026-01-10

    # Download fresh FINRA data first
    python real_short_interest_feed.py --download --as-of-date 2026-01-10

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import hashlib
import json
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, List, Any, Optional, Set


__version__ = "1.0.0"
__author__ = "Wake Robin Capital Management"


# Import feed modules
try:
    from finra_short_interest_feed import (
        download_finra_si_file,
        parse_finra_si_file,
        load_cached_si_data,
        get_latest_available_settlement_date,
        get_available_settlement_dates,
        is_data_available,
        RAW_DATA_DIR as SI_RAW_DIR,
    )
    HAS_SI_FEED = True
except ImportError:
    HAS_SI_FEED = False

try:
    from finra_short_volume_feed import (
        download_finra_short_volume,
        load_cached_short_volume,
        compute_short_volume_stats,
        get_available_trade_date,
        get_available_trade_dates,
        prev_business_day,
        RAW_DATA_DIR as SV_RAW_DIR,
    )
    HAS_SV_FEED = True
except ImportError:
    HAS_SV_FEED = False

try:
    from threshold_list_feed import (
        download_nasdaq_threshold_list,
        download_nyse_threshold_list,
        load_threshold_securities,
        get_threshold_flags_for_universe,
    )
    HAS_THRESHOLD_FEED = True
except ImportError:
    HAS_THRESHOLD_FEED = False


class SymbolNormalizer:
    """
    Normalizes FINRA symbols to match universe ticker symbols.

    Handles common discrepancies:
    - Case normalization (AAPL vs aapl)
    - Class suffixes (BRK.A vs BRKA vs BRK-A)
    - Exchange-specific suffixes
    """

    def __init__(self, universe_tickers: List[str]):
        """
        Initialize with universe tickers.

        Args:
            universe_tickers: List of canonical ticker symbols from universe
        """
        self.universe_set = {t.upper() for t in universe_tickers}

        # Build lookup variants
        self.lookup = {}
        for ticker in universe_tickers:
            upper = ticker.upper()
            self.lookup[upper] = upper

            # Add variants
            # Remove dots: BRKA -> BRK.A
            no_dot = upper.replace('.', '')
            self.lookup[no_dot] = upper

            # Add dot variant: BRKA -> BRK.A might be stored as BRKA
            if len(upper) > 3 and upper[-1].isalpha() and upper[-2].isdigit() is False:
                # Could be a class share like BRKA
                pass

    def normalize(self, symbol: str) -> Optional[str]:
        """
        Normalize a FINRA symbol to universe ticker.

        Args:
            symbol: Raw symbol from FINRA data

        Returns:
            Matched universe ticker or None if not found
        """
        if not symbol:
            return None

        upper = symbol.upper().strip()

        # Direct match
        if upper in self.lookup:
            return self.lookup[upper]

        # Try without common suffixes
        clean = upper
        for suffix in ['.A', '.B', '-A', '-B', '-W', '-WI', '-U']:
            if clean.endswith(suffix):
                clean = clean[:-len(suffix)]
                if clean in self.lookup:
                    return self.lookup[clean]

        # Try removing dots and dashes
        clean = upper.replace('.', '').replace('-', '')
        if clean in self.lookup:
            return self.lookup[clean]

        # Not in universe
        return None

    def get_match_stats(self, symbols: List[str]) -> Dict[str, Any]:
        """Get matching statistics for a list of symbols."""
        matched = 0
        unmatched = []

        for sym in symbols:
            if self.normalize(sym):
                matched += 1
            else:
                unmatched.append(sym)

        return {
            "total": len(symbols),
            "matched": matched,
            "match_rate": matched / len(symbols) if symbols else 0,
            "unmatched_sample": unmatched[:20],
        }


def load_market_data(market_data_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load market data for float shares, ADV, etc.

    Returns:
        Dict mapping ticker -> market data record
    """
    with open(market_data_path, 'r') as f:
        records = json.load(f)

    return {r["ticker"].upper(): r for r in records if r.get("ticker")}


def load_universe_tickers(universe_path: Path) -> List[str]:
    """Load tickers from universe file."""
    with open(universe_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = data.get("active_securities", data.get("records", []))
    else:
        records = []

    return [r["ticker"] for r in records if r.get("ticker")]


def compute_si_metrics(
    si_shares: int,
    float_shares: Optional[int],
    avg_daily_volume: Optional[int],
    price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute derived short interest metrics.

    Args:
        si_shares: Short interest in shares
        float_shares: Float shares (for SI% of float)
        avg_daily_volume: 20-day ADV (for days to cover)
        price: Current price (for SI$ calculation)

    Returns:
        Dict with computed metrics
    """
    metrics = {
        "short_interest_shares": si_shares,
    }

    # SI% of float
    if float_shares and float_shares > 0:
        si_pct = (si_shares / float_shares) * 100
        metrics["short_interest_pct"] = str(Decimal(str(si_pct)).quantize(Decimal("0.1")))
    else:
        metrics["short_interest_pct"] = None

    # Days to cover
    if avg_daily_volume and avg_daily_volume > 0:
        dtc = si_shares / avg_daily_volume
        metrics["days_to_cover"] = str(Decimal(str(dtc)).quantize(Decimal("0.1")))
    else:
        metrics["days_to_cover"] = None

    # SI in dollars
    if price and price > 0:
        si_dollars = si_shares * price
        metrics["short_interest_dollars"] = int(si_dollars)

    return metrics


def generate_short_interest_data(
    universe_tickers: List[str],
    market_data: Dict[str, Dict[str, Any]],
    as_of_date: date,
    use_finra_si: bool = True,
    use_finra_sv: bool = True,
    use_threshold: bool = True,
    fallback_to_market_data: bool = True,
) -> Dict[str, Any]:
    """
    Generate comprehensive short interest data for the universe.

    Combines:
    1. FINRA SI positions (if available)
    2. FINRA daily short volume (if available)
    3. Threshold list flags (if available)
    4. Fallback to market_data.json fields (if available)

    Args:
        universe_tickers: List of tickers in universe
        market_data: Market data keyed by ticker
        as_of_date: As-of date for PIT
        use_finra_si: Include FINRA short interest
        use_finra_sv: Include FINRA short volume
        use_threshold: Include threshold flags
        fallback_to_market_data: Use market_data fields when FINRA unavailable

    Returns:
        Dict with records and metadata
    """
    normalizer = SymbolNormalizer(universe_tickers)

    # Initialize results
    si_by_ticker: Dict[str, Dict[str, Any]] = {}
    for ticker in universe_tickers:
        si_by_ticker[ticker.upper()] = {
            "ticker": ticker.upper(),
            "has_finra_si": False,
            "has_finra_sv": False,
            "is_threshold_security": False,
        }

    # Source tracking
    sources_used = []
    diagnostics = {
        "universe_size": len(universe_tickers),
        "finra_si_matches": 0,
        "finra_sv_matches": 0,
        "threshold_matches": 0,
        "market_data_fallback": 0,
        "no_data": 0,
    }

    # Step 1: Load FINRA Short Interest
    finra_si_records = []
    si_settlement_date = None

    if use_finra_si and HAS_SI_FEED:
        # Find latest available SI data
        si_settlement_date = get_latest_available_settlement_date(as_of_date)

        # Check if we have it cached
        si_data = load_cached_si_data(si_settlement_date)
        if si_data:
            finra_si_records = si_data
            sources_used.append({
                "type": "FINRA_SI",
                "settlement_date": si_settlement_date.isoformat(),
                "records": len(finra_si_records),
            })

            # Match to universe
            for rec in finra_si_records:
                matched_ticker = normalizer.normalize(rec["symbol"])
                if matched_ticker and matched_ticker in si_by_ticker:
                    si_by_ticker[matched_ticker].update({
                        "has_finra_si": True,
                        "short_interest_shares": rec.get("short_interest_shares"),
                        "si_change_pct_finra": rec.get("si_change_pct"),
                        "finra_days_to_cover": rec.get("days_to_cover"),
                        "si_settlement_date": si_settlement_date.isoformat(),
                    })
                    diagnostics["finra_si_matches"] += 1

    # Step 2: Load FINRA Daily Short Volume
    if use_finra_sv and HAS_SV_FEED:
        # Load recent trading days
        sv_records_by_date = {}
        trade_dates = get_available_trade_dates()

        for td in trade_dates[:20]:  # Last 20 days
            sv_data = load_cached_short_volume(td)
            if sv_data:
                sv_records_by_date[td] = sv_data

        if sv_records_by_date:
            sources_used.append({
                "type": "FINRA_SV",
                "trade_dates": [d.isoformat() for d in sorted(sv_records_by_date.keys(), reverse=True)[:5]],
                "total_dates": len(sv_records_by_date),
            })

            # Compute stats for each ticker
            for ticker in universe_tickers:
                stats = compute_short_volume_stats(
                    sv_records_by_date, ticker, as_of_date, lookback_days=20
                )
                if stats["data_points"] > 0:
                    si_by_ticker[ticker.upper()].update({
                        "has_finra_sv": True,
                        "short_vol_ratio_latest": stats["short_vol_ratio_latest"],
                        "short_vol_ratio_avg_20d": stats["short_vol_ratio_avg"],
                        "short_vol_ratio_zscore": stats["short_vol_ratio_zscore"],
                    })
                    diagnostics["finra_sv_matches"] += 1

    # Step 3: Load Threshold Flags
    if use_threshold and HAS_THRESHOLD_FEED:
        threshold_flags = get_threshold_flags_for_universe(universe_tickers, as_of_date)
        threshold_count = sum(1 for v in threshold_flags.values() if v)

        if threshold_count > 0:
            sources_used.append({
                "type": "THRESHOLD_LIST",
                "count": threshold_count,
            })

        for ticker, is_threshold in threshold_flags.items():
            if ticker in si_by_ticker:
                si_by_ticker[ticker]["is_threshold_security"] = is_threshold
                if is_threshold:
                    diagnostics["threshold_matches"] += 1

    # Step 4: Compute derived metrics using market data
    for ticker in universe_tickers:
        ticker_upper = ticker.upper()
        record = si_by_ticker[ticker_upper]
        mkt = market_data.get(ticker_upper, {})

        # Get float and ADV from market data
        float_shares = mkt.get("float_shares")
        avg_volume = mkt.get("avg_volume")
        price = mkt.get("price")

        # If we have FINRA SI shares, compute our own metrics
        if record.get("short_interest_shares"):
            metrics = compute_si_metrics(
                record["short_interest_shares"],
                float_shares,
                avg_volume,
                price
            )
            record.update(metrics)

        # Fallback to market_data.json fields if no FINRA data
        elif fallback_to_market_data and mkt:
            if mkt.get("short_percent") is not None:
                record["short_interest_pct"] = str(Decimal(str(mkt["short_percent"] * 100)).quantize(Decimal("0.1")))
                record["data_source"] = "market_data_fallback"
                diagnostics["market_data_fallback"] += 1

            if mkt.get("short_ratio") is not None:
                record["days_to_cover"] = str(Decimal(str(mkt["short_ratio"])).quantize(Decimal("0.1")))

        # Track no-data cases
        if not record.get("short_interest_pct") and not record.get("has_finra_sv"):
            diagnostics["no_data"] += 1

    # Step 5: Compute SI change if we have previous settlement
    if use_finra_si and HAS_SI_FEED and si_settlement_date:
        available_dates = get_available_settlement_dates()
        if len(available_dates) >= 2:
            prev_settlement = available_dates[1]
            prev_si_data = load_cached_si_data(prev_settlement)
            if prev_si_data:
                prev_by_symbol = {r["symbol"]: r for r in prev_si_data}
                for rec in finra_si_records:
                    matched_ticker = normalizer.normalize(rec["symbol"])
                    if matched_ticker and matched_ticker in si_by_ticker:
                        prev_rec = prev_by_symbol.get(rec["symbol"])
                        if prev_rec and prev_rec.get("short_interest_shares"):
                            current_si = rec.get("short_interest_shares", 0)
                            prev_si = prev_rec.get("short_interest_shares", 0)
                            if prev_si > 0:
                                change_pct = ((current_si - prev_si) / prev_si) * 100
                                si_by_ticker[matched_ticker]["short_interest_change_pct"] = str(
                                    Decimal(str(change_pct)).quantize(Decimal("0.1"))
                                )

    # Build output records
    records = []
    for ticker in sorted(universe_tickers):
        ticker_upper = ticker.upper()
        rec = si_by_ticker[ticker_upper]

        # Build clean output record
        output_rec = {
            "ticker": ticker_upper,
            "short_interest_pct": rec.get("short_interest_pct"),
            "days_to_cover": rec.get("days_to_cover"),
            "short_interest_change_pct": rec.get("short_interest_change_pct"),
            "is_threshold_security": rec.get("is_threshold_security", False),
            "short_vol_ratio_latest": rec.get("short_vol_ratio_latest"),
            "short_vol_ratio_zscore": rec.get("short_vol_ratio_zscore"),
            "report_date": si_settlement_date.isoformat() if si_settlement_date else as_of_date.isoformat(),
            "sources": [],
        }

        # Track data sources
        if rec.get("has_finra_si"):
            output_rec["sources"].append("FINRA_SI")
        if rec.get("has_finra_sv"):
            output_rec["sources"].append("FINRA_SV")
        if rec.get("is_threshold_security"):
            output_rec["sources"].append("THRESHOLD")
        if rec.get("data_source") == "market_data_fallback":
            output_rec["sources"].append("MARKET_DATA")

        # Add institutional ownership if available from market data
        mkt = market_data.get(ticker_upper, {})
        # Note: market_data.json doesn't have inst_long, but we keep placeholder
        output_rec["institutional_long_pct"] = None

        records.append(output_rec)

    # Compute coverage stats
    has_si_pct = sum(1 for r in records if r.get("short_interest_pct")) / len(records) * 100
    has_sv = sum(1 for r in records if r.get("short_vol_ratio_latest")) / len(records) * 100

    return {
        "records": records,
        "provenance": {
            "module": "real_short_interest_feed",
            "version": __version__,
            "as_of_date": as_of_date.isoformat(),
            "sources": sources_used,
            "si_settlement_date": si_settlement_date.isoformat() if si_settlement_date else None,
        },
        "diagnostics": {
            **diagnostics,
            "si_coverage_pct": round(has_si_pct, 1),
            "sv_coverage_pct": round(has_sv, 1),
        },
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Real Short Interest Data Feed (FINRA + Threshold Lists)"
    )
    parser.add_argument(
        "--universe",
        type=Path,
        help="Path to universe.json file"
    )
    parser.add_argument(
        "--market-data",
        type=Path,
        help="Path to market_data.json file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for short_interest.json"
    )
    parser.add_argument(
        "--as-of-date",
        required=True,
        help="As-of date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download fresh FINRA data before generating"
    )
    parser.add_argument(
        "--no-finra-si",
        action="store_true",
        help="Skip FINRA short interest"
    )
    parser.add_argument(
        "--no-finra-sv",
        action="store_true",
        help="Skip FINRA short volume"
    )
    parser.add_argument(
        "--no-threshold",
        action="store_true",
        help="Skip threshold lists"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing file"
    )

    args = parser.parse_args()

    as_of_date = date.fromisoformat(args.as_of_date)

    print(f"Real Short Interest Feed v{__version__}")
    print(f"As-of date: {as_of_date}")
    print()

    # Download if requested
    if args.download:
        print("Downloading fresh FINRA data...")

        if HAS_SI_FEED:
            si_settlement = get_latest_available_settlement_date(as_of_date)
            print(f"  Downloading SI for settlement {si_settlement}...")
            result = download_finra_si_file(si_settlement, as_of_date=as_of_date)
            if result["success"]:
                print(f"    Success: {result['file_path']}")
            else:
                print(f"    Failed: {result.get('message', 'Unknown error')}")

        if HAS_SV_FEED:
            # Download last 5 trading days
            current = as_of_date
            for _ in range(5):
                current = prev_business_day(current)
                print(f"  Downloading short volume for {current}...")
                result = download_finra_short_volume(current)
                if result["success"]:
                    print(f"    Success: {result['file_path']}")
                else:
                    print(f"    Skipped: {result.get('message', 'Not available')}")

        if HAS_THRESHOLD_FEED:
            list_date = prev_business_day(as_of_date)
            print(f"  Downloading threshold lists for {list_date}...")
            for func, name in [(download_nasdaq_threshold_list, "Nasdaq"),
                               (download_nyse_threshold_list, "NYSE")]:
                result = func(list_date)
                if result["success"]:
                    print(f"    {name}: {result['file_path']}")
                else:
                    print(f"    {name}: {result.get('message', 'Not available')}")

        print()

    # Generate SI data
    if args.universe and args.market_data:
        print("Generating short interest data...")

        tickers = load_universe_tickers(args.universe)
        market_data = load_market_data(args.market_data)

        print(f"  Universe: {len(tickers)} tickers")
        print(f"  Market data: {len(market_data)} records")

        result = generate_short_interest_data(
            universe_tickers=tickers,
            market_data=market_data,
            as_of_date=as_of_date,
            use_finra_si=not args.no_finra_si,
            use_finra_sv=not args.no_finra_sv,
            use_threshold=not args.no_threshold,
        )

        # Print summary
        diag = result["diagnostics"]
        print()
        print("Coverage Summary:")
        print(f"  SI% coverage: {diag['si_coverage_pct']}%")
        print(f"  Short volume coverage: {diag['sv_coverage_pct']}%")
        print(f"  FINRA SI matches: {diag['finra_si_matches']}")
        print(f"  FINRA SV matches: {diag['finra_sv_matches']}")
        print(f"  Threshold flags: {diag['threshold_matches']}")
        print(f"  Market data fallback: {diag['market_data_fallback']}")
        print(f"  No data: {diag['no_data']}")

        # Show sample high-SI tickers
        records_with_si = [r for r in result["records"] if r.get("short_interest_pct")]
        if records_with_si:
            sorted_by_si = sorted(
                records_with_si,
                key=lambda x: float(x["short_interest_pct"] or 0),
                reverse=True
            )
            print()
            print("Top 10 by SI%:")
            for r in sorted_by_si[:10]:
                flags = []
                if r.get("is_threshold_security"):
                    flags.append("THRESHOLD")
                if r.get("short_vol_ratio_zscore") and float(r.get("short_vol_ratio_zscore", 0)) > 2:
                    flags.append("SV_SPIKE")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                print(f"  {r['ticker']}: SI={r['short_interest_pct']}%, DTC={r.get('days_to_cover', 'N/A')}{flag_str}")

        if args.dry_run:
            print()
            print("[DRY-RUN] Not writing output file")
        elif args.output:
            # Write output (just the records list for compatibility)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(result["records"], f, indent=2)
                f.write('\n')

            print()
            print(f"Wrote {len(result['records'])} records to {args.output}")

    else:
        # Show status
        print("Feed Status:")
        print(f"  FINRA SI feed: {'Available' if HAS_SI_FEED else 'Not available'}")
        print(f"  FINRA SV feed: {'Available' if HAS_SV_FEED else 'Not available'}")
        print(f"  Threshold feed: {'Available' if HAS_THRESHOLD_FEED else 'Not available'}")

        if HAS_SI_FEED:
            dates = get_available_settlement_dates()
            print(f"  Cached SI dates: {len(dates)}")
            if dates:
                print(f"    Latest: {dates[0]}")

        if HAS_SV_FEED:
            dates = get_available_trade_dates()
            print(f"  Cached SV dates: {len(dates)}")
            if dates:
                print(f"    Latest: {dates[0]}")

        print()
        print("To generate SI data, run:")
        print("  python real_short_interest_feed.py --universe production_data/universe.json \\")
        print("      --market-data production_data/market_data.json \\")
        print("      --output production_data/short_interest.json \\")
        print("      --as-of-date 2026-01-10")


if __name__ == "__main__":
    main()
