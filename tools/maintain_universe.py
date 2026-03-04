"""
Universe Maintenance Utility — tools/maintain_universe.py

Three sub-commands for managing production_data/universe.json:

  audit   — report tickers that may need attention (never modifies files)
  add     — add a new ticker with minimal required fields
  retire  — soft-delete a ticker (sets status to the given reason type)

XBI is a protected benchmark ticker and is always excluded from audit flags
and cannot be retired.

Usage:
    python3 tools/maintain_universe.py audit
    python3 tools/maintain_universe.py add AAPL --name "Apple Inc." --sector biotech
    python3 tools/maintain_universe.py retire XXXX --reason "acquired by Pfizer 2026-02-01"
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

UNIVERSE_JSON = PROJECT_ROOT / "production_data" / "universe.json"
AUDIT_LOG = PROJECT_ROOT / "production_data" / "universe_audit_log.jsonl"
SNAPSHOTS_ROOT = PROJECT_ROOT / "data" / "snapshots"

# Tickers that must never be touched — benchmark / infrastructure
_PROTECTED = {"XBI"}

# How many recent snapshots to check for data-gap detection
_DATA_GAP_LOOKBACK = 5


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_universe() -> List[Dict]:
    if not UNIVERSE_JSON.is_file():
        raise FileNotFoundError(f"universe.json not found: {UNIVERSE_JSON}")
    return json.loads(UNIVERSE_JSON.read_text())


def _save_universe(universe: List[Dict]) -> None:
    UNIVERSE_JSON.write_text(json.dumps(universe, indent=2))


def _append_audit_log(entry: Dict) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _discover_snapshot_dates(n: int = _DATA_GAP_LOOKBACK) -> List[str]:
    """Return the most recent N snapshot date strings, sorted descending."""
    if not SNAPSHOTS_ROOT.is_dir():
        return []
    dates = []
    for d in SNAPSHOTS_ROOT.iterdir():
        name = d.name
        if len(name) == 10 and name[4] == "-" and name[7] == "-":
            try:
                datetime.strptime(name, "%Y-%m-%d")
                dates.append(name)
            except ValueError:
                pass
    return sorted(dates, reverse=True)[:n]


def _tickers_in_snapshot(snap_date: str) -> Set[str]:
    """Return set of tickers present in a snapshot's rankings.csv."""
    csv_path = SNAPSHOTS_ROOT / snap_date / "rankings.csv"
    if not csv_path.is_file():
        return set()
    tickers: Set[str] = set()
    import csv
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tk = row.get("ticker", "").strip().upper()
            if tk:
                tickers.add(tk)
    return tickers


def _eligible_tickers_in_snapshot(snap_date: str) -> Set[str]:
    """Return tickers marked eligible=1 in a snapshot."""
    csv_path = SNAPSHOTS_ROOT / snap_date / "rankings.csv"
    if not csv_path.is_file():
        return set()
    tickers: Set[str] = set()
    import csv
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("eligible", "").strip() == "1":
                tk = row.get("ticker", "").strip().upper()
                if tk:
                    tickers.add(tk)
    return tickers


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def cmd_audit(args: argparse.Namespace) -> None:
    """Report tickers that may need attention. Never modifies files."""
    universe = _load_universe()
    snap_dates = _discover_snapshot_dates(_DATA_GAP_LOOKBACK)

    # Build per-ticker presence sets from recent snapshots
    snap_tickers: List[Set[str]] = [_tickers_in_snapshot(d) for d in snap_dates]
    snap_eligible: List[Set[str]] = [_eligible_tickers_in_snapshot(d) for d in snap_dates]

    flags: List[Dict] = []

    for entry in universe:
        ticker = entry.get("ticker", "").upper()
        if ticker in _PROTECTED:
            continue

        status = entry.get("status", "active")

        # Flag 1: non-active status still in universe
        if status not in ("active", ""):
            flags.append({
                "ticker": ticker,
                "flag": "non_active_status",
                "detail": f"status={status!r}",
            })

        # Flag 2: missing from all recent snapshots (data gap)
        if snap_dates and all(ticker not in s for s in snap_tickers):
            flags.append({
                "ticker": ticker,
                "flag": "missing_from_recent_snapshots",
                "detail": f"absent from last {len(snap_dates)} snapshot(s): {snap_dates}",
            })

        # Flag 3: always ineligible across all recent snapshots
        if snap_dates and all(
            (ticker in s and ticker not in e)
            for s, e in zip(snap_tickers, snap_eligible)
            if ticker in s
        ) and any(ticker in s for s in snap_tickers):
            flags.append({
                "ticker": ticker,
                "flag": "always_ineligible",
                "detail": f"present but ineligible in all {len(snap_dates)} recent snapshot(s)",
            })

    if not flags:
        print(f"[audit] Universe looks clean ({len(universe)} tickers). No flags.")
        return

    print(f"[audit] {len(flags)} flag(s) across {len(universe)} tickers:\n")
    by_flag: Dict[str, List] = {}
    for f in flags:
        by_flag.setdefault(f["flag"], []).append(f)

    for flag_type, items in sorted(by_flag.items()):
        print(f"  {flag_type} ({len(items)}):")
        for item in items:
            print(f"    {item['ticker']:8s}  {item['detail']}")
        print()

    print(f"[audit] To retire: python3 tools/maintain_universe.py retire TICKER --reason '...'")
    print(f"[audit] Protected tickers (never flagged): {sorted(_PROTECTED)}")


# ---------------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------------

def cmd_add(args: argparse.Namespace) -> None:
    """Add a new ticker to universe.json."""
    ticker = args.ticker.upper().strip()

    if ticker in _PROTECTED:
        print(f"[add] {ticker} is a protected benchmark ticker — cannot modify.", file=sys.stderr)
        sys.exit(1)

    universe = _load_universe()

    # Check for duplicate
    existing_tickers = {e.get("ticker", "").upper() for e in universe}
    if ticker in existing_tickers:
        print(f"[add] {ticker} already exists in universe.json. No change made.")
        return

    entry: Dict[str, Any] = {
        "ticker": ticker,
        "name": args.name or ticker,
        "exchange": args.exchange or "",
        "sector": args.sector or "Biotechnology",
        "status": "active",
        "added_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "added_from_etf": False,
        "description": args.description or f"Added by maintain_universe.py",
    }
    if args.cik:
        entry["cik"] = args.cik

    universe.append(entry)
    _save_universe(universe)

    log_entry = {
        "timestamp": _now_iso(),
        "action": "add",
        "ticker": ticker,
        "operator": args.operator or "manual",
        "fields": entry,
    }
    _append_audit_log(log_entry)

    print(f"[add] Added {ticker} to universe.json ({len(universe)} total).")


# ---------------------------------------------------------------------------
# Retire
# ---------------------------------------------------------------------------

def cmd_retire(args: argparse.Namespace) -> None:
    """Soft-delete a ticker (set status, never physically remove)."""
    ticker = args.ticker.upper().strip()

    if ticker in _PROTECTED:
        print(f"[retire] {ticker} is a protected benchmark ticker — cannot retire.", file=sys.stderr)
        sys.exit(1)

    universe = _load_universe()
    matched = [e for e in universe if e.get("ticker", "").upper() == ticker]
    if not matched:
        print(f"[retire] {ticker} not found in universe.json.", file=sys.stderr)
        sys.exit(1)

    entry = matched[0]
    prior_status = entry.get("status", "active")

    # Determine new status from reason keyword, or use explicit status flag
    reason = args.reason or ""
    if args.status:
        new_status = args.status
    elif "acqui" in reason.lower() or "merger" in reason.lower():
        new_status = "excluded_acquired"
    elif "delist" in reason.lower() or "bankrupt" in reason.lower():
        new_status = "delisted"
    else:
        new_status = "retired"

    entry["status"] = new_status
    entry["retire_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if reason:
        entry["retire_reason"] = reason

    _save_universe(universe)

    log_entry = {
        "timestamp": _now_iso(),
        "action": "retire",
        "ticker": ticker,
        "prior_status": prior_status,
        "new_status": new_status,
        "reason": reason,
        "operator": args.operator or "manual",
    }
    _append_audit_log(log_entry)

    print(f"[retire] {ticker}: status {prior_status!r} → {new_status!r}")
    if reason:
        print(f"  Reason: {reason}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Universe maintenance utility for production_data/universe.json.",
    )
    parser.add_argument(
        "--universe-path", type=Path, default=None,
        help="Override path to universe.json (for testing)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── audit ────────────────────────────────────────────────────────
    p_audit = sub.add_parser("audit", help="Report tickers that may need attention")
    p_audit.set_defaults(func=cmd_audit)

    # ── add ──────────────────────────────────────────────────────────
    p_add = sub.add_parser("add", help="Add a ticker to the universe")
    p_add.add_argument("ticker", help="Ticker symbol (case-insensitive)")
    p_add.add_argument("--name", default=None, help="Company name")
    p_add.add_argument("--sector", default="Biotechnology")
    p_add.add_argument("--exchange", default=None)
    p_add.add_argument("--cik", default=None, help="SEC CIK (optional, e.g. 0001234567)")
    p_add.add_argument("--description", default=None)
    p_add.add_argument("--operator", default=None, help="Operator name for audit log")
    p_add.set_defaults(func=cmd_add)

    # ── retire ───────────────────────────────────────────────────────
    p_retire = sub.add_parser("retire", help="Soft-delete a ticker (sets status)")
    p_retire.add_argument("ticker", help="Ticker symbol to retire")
    p_retire.add_argument("--reason", default="", help="Human-readable reason (e.g. 'acquired by Pfizer 2026-02-01')")
    p_retire.add_argument("--status", default=None,
                          choices=["delisted", "excluded_acquired", "retired"],
                          help="Explicit status (inferred from --reason if omitted)")
    p_retire.add_argument("--operator", default=None, help="Operator name for audit log")
    p_retire.set_defaults(func=cmd_retire)

    args = parser.parse_args()

    # Allow test override of universe path
    if args.universe_path:
        global UNIVERSE_JSON
        UNIVERSE_JSON = args.universe_path

    args.func(args)


if __name__ == "__main__":
    main()
