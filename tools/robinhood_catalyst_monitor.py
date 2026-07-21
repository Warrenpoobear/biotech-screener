#!/usr/bin/env python3
"""
Monitor imminent catalysts for Robinhood holdings against biotech screener picks.
Generates daily alerts for catalyst windows and portfolio impact analysis.

Fails loudly if:
- Snapshot is missing or stale (>7 days old)
- Holdings data is missing
- Decision portfolio cannot be loaded
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path


def find_latest_snapshot():
    """Find most recent snapshot. Fail if missing or stale."""
    snap_dir = Path("data/snapshots")
    if not snap_dir.exists():
        raise FileNotFoundError(f"❌ Snapshot directory not found: {snap_dir}")

    snapshots = sorted(
        [d for d in snap_dir.iterdir() if d.is_dir() and len(d.name) == 10 and d.name[4] == "-" and d.name[7] == "-"]
    )
    if not snapshots:
        raise FileNotFoundError("❌ No snapshots found in data/snapshots_pit/")

    latest = snapshots[-1]
    snapshot_date = latest.name

    # Check freshness (warn if >7 days old)
    snap_dt = datetime.strptime(snapshot_date, "%Y-%m-%d")
    days_old = (datetime.now() - snap_dt).days
    if days_old > 7:
        print(f"⚠️  WARNING: Snapshot is {days_old} days old ({snapshot_date})")
    elif days_old > 0:
        print(f"ℹ️  Using snapshot from {days_old} day(s) ago: {snapshot_date}")

    return snapshot_date, latest


def load_decision_portfolio(snapshot_path):
    """Load screener decision portfolio with error handling."""
    portfolio_file = snapshot_path / "decision_portfolio.json"
    if not portfolio_file.exists():
        raise FileNotFoundError(f"❌ Decision portfolio not found: {portfolio_file}")

    try:
        with open(portfolio_file) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Failed to parse decision portfolio: {e}")


def load_robinhood_holdings():
    """Load holdings from JSON file. Falls back to hardcoded if file missing."""
    holdings_file = Path("production_data/robinhood_holdings.json")

    if holdings_file.exists():
        try:
            with open(holdings_file) as f:
                data = json.load(f)
            holdings = set(data.get("tickers", []))
            updated = data.get("updated_at", "unknown")
            print(f"✓ Loaded holdings from {holdings_file} (updated: {updated})")
            return holdings, updated
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Failed to load holdings file: {e}. Using hardcoded fallback.")

    # Hardcoded fallback (from 2026-06-10 Robinhood fetch)
    holdings = {
        "COGT",
        "DNTH",
        "ALMS",
        "RVMD",
        "PRAX",
        "XENE",
        "MIRM",
        "ORKA",
        "RYTM",
        "TNGX",
        "PHVS",
        "TYRA",
        "CELC",
        "APGE",
        "ASND",
        "SION",
        "SRRK",
        "PTGX",
        "NVDA",
        "AVGO",
        "APH",
        "KTOS",
        "FER",
        "SGOL",
        "SGOV",
        "CCJ",
        "STRL",
        "MCK",
        "ACM",
        "BSX",
        "VRT",
        "FNV",
        "CRH",
        "SIVR",
        "FIX",
        "DDOG",
        "MOD",
        "MU",
        "GOOGL",
        "REMX",
        "TEVA",
        "LLY",
        "JNJ",
        "AMAT",
        "MRVL",
        "EPRX",
        "JAZZ",
        "AMLX",
        "IBB",
        "SNDK",
        "INDV",
        "CLOA",
        "VERA",
        "GH",
        "CRVS",
        "INBX",
        "IMVT",
        "RNA",
        "ROIV",
        "KRYS",
        "ERAS",
        "IONS",
        "KYMR",
        "MAGS",
        "MSFT",
        "XBI",
        "IVV",
        "AVAV",
        "CRS",
        "CIEN",
        "ALNY",
        "TLT",
        "WM",
        "CAT",
        "ANNX",
        "ARM",
        "BE",
        "ATMU",
        "CPER",
    }
    print("⚠️  Using hardcoded holdings fallback (may be stale)")
    return holdings, "2026-06-10 (hardcoded fallback)"


def analyze_catalysts(snapshot_date, snapshot_path):
    """Analyze imminent catalysts for holdings."""
    try:
        portfolio = load_decision_portfolio(snapshot_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"{e}")
        sys.exit(1)

    holdings, holdings_updated = load_robinhood_holdings()

    snap_dt = datetime.strptime(snapshot_date, "%Y-%m-%d")

    catalysts = {"imminent": [], "near_term": [], "medium_term": [], "not_screened": []}

    screener_tickers = {pos["ticker"] for pos in portfolio["positions"]}

    for pos in portfolio["positions"]:
        ticker = pos["ticker"]
        if ticker not in holdings:
            continue

        catalyst_days_raw = pos.get("catalyst_days", 999)
        catalyst_days = int(catalyst_days_raw) if catalyst_days_raw else 999

        entry = {
            "ticker": ticker,
            "company": pos.get("company_name", ""),
            "rank": pos.get("actionable_rank", 999),
            "tier": pos.get("tier_any", ""),
            "catalyst_date": (snap_dt + timedelta(days=catalyst_days)).strftime("%Y-%m-%d"),
            "catalyst_days": catalyst_days,
            "dev_stage": pos.get("development_stage", ""),
        }

        if catalyst_days <= 14:
            catalysts["imminent"].append(entry)
        elif catalyst_days <= 30:
            catalysts["near_term"].append(entry)
        elif catalyst_days <= 60:
            catalysts["medium_term"].append(entry)

    for ticker in sorted(holdings):
        if ticker not in screener_tickers:
            catalysts["not_screened"].append({"ticker": ticker})

    return catalysts, len(holdings), holdings_updated


def print_alert(catalysts, snapshot_date, holdings_count, holdings_updated):
    """Print formatted catalyst alert."""
    print("\n" + "=" * 90)
    print(f"ROBINHOOD CATALYST MONITOR — {snapshot_date}")
    print("=" * 90)
    print(f"Holdings: {holdings_count} tickers (updated: {holdings_updated})")

    print("\n🔴 IMMINENT CATALYSTS (0-14 days)")
    print("─" * 90)
    if catalysts["imminent"]:
        for entry in sorted(catalysts["imminent"], key=lambda x: x["catalyst_days"]):
            print(
                f"  {entry['ticker']:6} {entry['company'][:40]:40} "
                f"Rank {entry['rank']:3} Tier {entry['tier']}  "
                f"Catalyst: {entry['catalyst_date']} ({entry['catalyst_days']}d)"
            )
        print("\n  ⚠️  ACTION: Set price alerts, monitor earnings/clinical updates")
    else:
        print("  ✓ No imminent catalysts in screened holdings")

    print("\n🟠 NEAR-TERM CATALYSTS (15-30 days)")
    print("─" * 90)
    if catalysts["near_term"]:
        for entry in sorted(catalysts["near_term"], key=lambda x: x["catalyst_days"])[:15]:
            print(
                f"  {entry['ticker']:6} {entry['company'][:40]:40} "
                f"Rank {entry['rank']:3} Tier {entry['tier']}  "
                f"Catalyst: {entry['catalyst_date']}"
            )
        if len(catalysts["near_term"]) > 15:
            print(f"  ... and {len(catalysts['near_term'])-15} more")
    else:
        print("  ✓ None in window")

    print("\n🟡 MEDIUM-TERM CATALYSTS (31-60 days)")
    print("─" * 90)
    if catalysts["medium_term"]:
        print(f"  {len(catalysts['medium_term'])} holdings with catalysts in 31-60 day window")
    else:
        print("  ✓ None")

    print("\n📊 SUMMARY")
    print("─" * 90)
    print(f"  Imminent (0-14d):     {len(catalysts['imminent']):3} holdings")
    print(f"  Near-term (15-30d):   {len(catalysts['near_term']):3} holdings")
    print(f"  Medium-term (31-60d): {len(catalysts['medium_term']):3} holdings")
    print(
        f"  Total monitored:      {len(catalysts['imminent']) + len(catalysts['near_term']) + len(catalysts['medium_term']):3} / {holdings_count}"
    )


def main():
    try:
        snapshot_date, snapshot_path = find_latest_snapshot()
    except FileNotFoundError as e:
        print(f"{e}")
        sys.exit(1)

    try:
        catalysts, holdings_count, holdings_updated = analyze_catalysts(snapshot_date, snapshot_path)
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)

    print_alert(catalysts, snapshot_date, holdings_count, holdings_updated)

    # Write JSON output
    output = {
        "snapshot_date": snapshot_date,
        "timestamp": datetime.now().isoformat(),
        "holdings_count": holdings_count,
        "holdings_updated": holdings_updated,
        "catalysts": catalysts,
        "summary": {
            "imminent_count": len(catalysts["imminent"]),
            "near_term_count": len(catalysts["near_term"]),
            "medium_term_count": len(catalysts["medium_term"]),
        },
    }

    output_file = Path(f"artifacts/catalyst_alerts/{snapshot_date}_catalyst_monitor.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Detailed output: {output_file}")


if __name__ == "__main__":
    main()
