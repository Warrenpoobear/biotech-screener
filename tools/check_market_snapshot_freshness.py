"""Staleness + validity gate for market_snapshot.json.

Classification: REGIME_SNAPSHOT_REFRESH_AND_STALENESS_GATE_NO_MODEL_CHANGE

Usage:
    python tools/check_market_snapshot_freshness.py
    python tools/check_market_snapshot_freshness.py --as-of-date 2026-06-26
    python tools/check_market_snapshot_freshness.py --snapshot-path /path/to/market_snapshot.json

Exit codes:
    0  — snapshot is fresh and valid
    1  — snapshot is stale or invalid (fail closed)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT_PATH = PROJECT_ROOT / "data" / "market_snapshot.json"

log = logging.getLogger("check_market_snapshot_freshness")

MAX_STALE_TRADING_DAYS = 2

# Fields the regime engine uses as primary inputs; must not all be null/zero
REGIME_SIGNAL_FIELDS = (
    "vix",
    "xbi_vs_spy_30d",
    "xbi_momentum_10d",
    "spy_momentum_10d",
    "xbi_realized_vol_20d",
)


def _count_trading_days(from_date: date, to_date: date) -> int:
    """Count business days (Mon-Fri) between two dates, inclusive of endpoints."""
    if from_date > to_date:
        return 0
    count = 0
    cur = from_date
    while cur <= to_date:
        if cur.weekday() < 5:  # Mon=0 … Fri=4
            count += 1
        cur += timedelta(days=1)
    return count


def check_freshness(
    snapshot_path: Path | str = DEFAULT_SNAPSHOT_PATH,
    reference_date: date | str | None = None,
    max_stale_trading_days: int = MAX_STALE_TRADING_DAYS,
) -> dict:
    """Check whether market_snapshot.json is fresh and valid.

    Args:
        snapshot_path: Path to the snapshot file.
        reference_date: The date to measure staleness against (defaults to today).
        max_stale_trading_days: Maximum allowed trading-day gap before FAIL.

    Returns:
        {
            "ok": bool,
            "age_trading_days": int | None,
            "snapshot_as_of_date": str | None,
            "reference_date": str,
            "issues": list[str],
            "feeds": dict,          # snapshot feeds status
            "signal_fields": dict,  # field name → value (for diagnostics)
        }
    """
    snapshot_path = Path(snapshot_path)
    issues: list[str] = []

    if isinstance(reference_date, str):
        reference_date = date.fromisoformat(reference_date)
    if reference_date is None:
        reference_date = date.today()

    ref_str = reference_date.isoformat()

    # ------------------------------------------------------------------ #
    # 1. File exists                                                       #
    # ------------------------------------------------------------------ #
    if not snapshot_path.exists():
        issues.append(f"snapshot file not found: {snapshot_path}")
        return {
            "ok": False,
            "age_trading_days": None,
            "snapshot_as_of_date": None,
            "reference_date": ref_str,
            "issues": issues,
            "feeds": {},
            "signal_fields": {},
        }

    # ------------------------------------------------------------------ #
    # 2. Parse JSON                                                        #
    # ------------------------------------------------------------------ #
    try:
        snap = json.loads(snapshot_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        issues.append(f"cannot parse snapshot JSON: {exc}")
        return {
            "ok": False,
            "age_trading_days": None,
            "snapshot_as_of_date": None,
            "reference_date": ref_str,
            "issues": issues,
            "feeds": {},
            "signal_fields": {},
        }

    prov = snap.get("provenance", {})
    snap_date_str = prov.get("as_of_date", "")

    # ------------------------------------------------------------------ #
    # 3. Staleness check                                                   #
    # ------------------------------------------------------------------ #
    age_trading_days: int | None = None
    if not snap_date_str:
        issues.append("provenance.as_of_date is missing")
    else:
        try:
            snap_date = date.fromisoformat(snap_date_str)
            # Age = trading days from snap_date to reference_date (exclusive of snap_date)
            age_trading_days = max(0, _count_trading_days(snap_date, reference_date) - 1)
            if age_trading_days > max_stale_trading_days:
                issues.append(
                    f"snapshot is {age_trading_days} trading days old "
                    f"(as_of={snap_date_str}, reference={ref_str}); "
                    f"limit is {max_stale_trading_days}"
                )
        except ValueError:
            issues.append(f"provenance.as_of_date is not a valid date: {snap_date_str!r}")

    # ------------------------------------------------------------------ #
    # 4. VIX validity                                                      #
    # ------------------------------------------------------------------ #
    vix_raw = snap.get("vix")
    signal_fields: dict[str, str | None] = {}
    for f in REGIME_SIGNAL_FIELDS:
        signal_fields[f] = snap.get(f)

    if vix_raw is None:
        issues.append("vix field is null — VIX feed was absent")
    else:
        try:
            vix_val = float(vix_raw)
            if vix_val == 0.0:
                issues.append(
                    "vix = 0 — impossible for live market data; "
                    "this indicates a feed failure that was silently zeroed"
                )
            elif vix_val < 5.0 or vix_val > 90.0:
                issues.append(f"vix = {vix_val} is outside plausible range [5, 90]")
        except (ValueError, TypeError):
            issues.append(f"vix = {vix_raw!r} cannot be parsed as a float")

    # ------------------------------------------------------------------ #
    # 5. All-zero signal check                                             #
    # ------------------------------------------------------------------ #
    non_null_vals = [v for f in REGIME_SIGNAL_FIELDS if (v := snap.get(f)) is not None]
    if non_null_vals:
        try:
            if all(float(v) == 0.0 for v in non_null_vals):
                issues.append(
                    "all regime signal fields are 0.0 — indicates wholesale feed "
                    "failure where missing values were silently zeroed; "
                    "regime detector cannot produce a valid classification"
                )
        except (ValueError, TypeError):
            pass

    feeds = snap.get("feeds", {})
    return {
        "ok": len(issues) == 0,
        "age_trading_days": age_trading_days,
        "snapshot_as_of_date": snap_date_str,
        "reference_date": ref_str,
        "issues": issues,
        "feeds": feeds,
        "signal_fields": signal_fields,
    }


def count_present_feeds(feeds: dict) -> int:
    """Count feeds that produced a value.

    Live refresh statuses contain "live". Reconstructed snapshots use "pit"
    and "pit_volume_proxy". A "failed" status stays absent.
    """
    present = 0
    for status in feeds.values():
        text = str(status)
        if "live" in text or text.startswith("pit"):
            present += 1
    return present


def emit_diagnostic(result: dict) -> None:
    """Log a human-readable diagnostic for the freshness check result."""
    status = "FRESH" if result["ok"] else "STALE/INVALID"
    age = result.get("age_trading_days")
    age_str = f"{age} trading day(s)" if age is not None else "unknown age"
    log.info(
        "Snapshot %s | %s old | as_of=%s | reference=%s",
        status,
        age_str,
        result.get("snapshot_as_of_date", "?"),
        result.get("reference_date", "?"),
    )
    for issue in result.get("issues", []):
        log.error("  ISSUE: %s", issue)
    feeds = result.get("feeds", {})
    present = count_present_feeds(feeds)
    if feeds:
        log.info("  Feeds: %d/%d present", present, len(feeds))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Check market_snapshot.json freshness and validity")
    parser.add_argument("--snapshot-path", default=str(DEFAULT_SNAPSHOT_PATH))
    parser.add_argument("--as-of-date", default="")
    parser.add_argument(
        "--max-stale-days",
        type=int,
        default=MAX_STALE_TRADING_DAYS,
        help="Maximum allowed trading-day age before FAIL (default: 2)",
    )
    args = parser.parse_args()

    ref = date.fromisoformat(args.as_of_date) if args.as_of_date else date.today()
    result = check_freshness(
        snapshot_path=args.snapshot_path,
        reference_date=ref,
        max_stale_trading_days=args.max_stale_days,
    )

    emit_diagnostic(result)

    if not result["ok"]:
        print("\nREGIME_INPUT_STALE_OR_INVALID — regime-dependent production " "should not proceed on this snapshot.")
        for issue in result["issues"]:
            print(f"  • {issue}")
        return 1

    print(
        f"\nSnapshot OK — {result['age_trading_days']} trading day(s) old "
        f"(as_of={result['snapshot_as_of_date']}, reference={result['reference_date']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
