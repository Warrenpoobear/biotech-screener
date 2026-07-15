#!/usr/bin/env python3
"""Post-open options surface delta monitor — morning briefing.

Compares today's live Tastytrade options diagnostics against yesterday's
promoted snapshot and flags names with significant overnight surface shifts:
  - ATM IV jumps (absolute and regime changes)
  - Risk reversal (RR_25d) flips or large moves
  - Put/call skew shifts
  - Term structure transitions (contango ↔ backwardation)

Read-only — does not change rankings, scoring, or execution.

Output:
  - Markdown briefing to stdout (pipe to file or review)
  - JSON artifact: data/snapshots/{date}/surface_delta.json
  - CSV artifact:  data/snapshots/{date}/surface_delta.csv

Usage:
    python tools/surface_delta_monitor.py
    python tools/surface_delta_monitor.py --as-of-date 2026-03-26
    python tools/surface_delta_monitor.py --prior-date 2026-03-25
    python tools/surface_delta_monitor.py --json-only
    python tools/surface_delta_monitor.py --dry-run   # fetch live, diff, print only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("surface_delta_monitor")

SCHEMA_VERSION = "surface_delta.v1"

# ---------------------------------------------------------------------------
# Thresholds — when to flag a name
# ---------------------------------------------------------------------------
# Thresholds scale by IV regime.  EXTREME-regime names (IV > 200%) have
# naturally high day-to-day jitter on all surface metrics; fixed thresholds
# would fire on nearly every name.  We use *relative* IV thresholds for
# EXTREME and *absolute* for NORMAL/ELEVATED.

# ATM IV change — NORMAL/ELEVATED: absolute (e.g. 0.10 = 10 vol points)
IV_JUMP_THRESHOLD = 0.10
IV_JUMP_LARGE = 0.20
# ATM IV change — EXTREME: relative to prior IV (e.g. 0.30 = 30% of prior)
IV_JUMP_REL_THRESHOLD = 0.30
IV_JUMP_REL_LARGE = 0.50

# Risk reversal: absolute change in RR_25d
# NORMAL/ELEVATED
RR_FLIP_THRESHOLD = 0.02  # sign change + magnitude > 2% is a flip
RR_MOVE_THRESHOLD = 0.05  # large move without sign change
# EXTREME: much wider bands (these RR values swing 20-80% routinely)
RR_FLIP_THRESHOLD_EXTREME = 0.20
RR_MOVE_THRESHOLD_EXTREME = 0.40

# Put/call skew: absolute change
SKEW_SHIFT_THRESHOLD = 0.10
SKEW_SHIFT_THRESHOLD_EXTREME = 0.40

# Term slope: transition across zero (contango ↔ backwardation)
TERM_SLOPE_FLIP_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _sf(val: Any, default: float = float("nan")) -> float:
    """Safe float parse."""
    if val is None or val == "":
        return default
    try:
        f = float(val)
        return f
    except (ValueError, TypeError):
        return default


def load_diagnostics_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load options_diagnostics.csv → {ticker: row_dict}."""
    if not path.exists():
        logger.warning("Diagnostics file not found: %s", path)
        return {}
    result: Dict[str, Dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ticker = (row.get("ticker") or "").strip().upper()
            if ticker:
                result[ticker] = row
    return result


def _is_promoted_snapshot(name: str) -> bool:
    """True for promoted snapshot dirs (YYYY-MM-DD), false for staging/pre dirs."""
    return len(name) == 10 and not name.startswith("_") and name != "state"


def find_prior_snapshot(
    snapshots_root: Path,
    current_date: str,
) -> Optional[Path]:
    """Find the most recent promoted snapshot directory before current_date."""
    candidates = sorted(
        d.name
        for d in snapshots_root.iterdir()
        if d.is_dir() and _is_promoted_snapshot(d.name) and d.name < current_date
    )
    if not candidates:
        return None
    return snapshots_root / candidates[-1]


def find_prior_snapshots(
    snapshots_root: Path,
    current_date: str,
    limit: int = 10,
) -> List[Path]:
    """Promoted snapshot directories before current_date, newest first.

    Used to walk back past crash-fragment days whose directories exist but
    hold no diagnostics (e.g. 2026-07-13/14 GATE_ALLOWLIST outage).
    """
    candidates = sorted(
        (
            d.name
            for d in snapshots_root.iterdir()
            if d.is_dir() and _is_promoted_snapshot(d.name) and d.name < current_date
        ),
        reverse=True,
    )
    return [snapshots_root / n for n in candidates[:limit]]


def find_latest_snapshot(snapshots_root: Path) -> Optional[Path]:
    """Find the latest promoted snapshot directory."""
    candidates = sorted(d.name for d in snapshots_root.iterdir() if d.is_dir() and _is_promoted_snapshot(d.name))
    if not candidates:
        return None
    return snapshots_root / candidates[-1]


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------


def classify_iv_regime(atm_iv: float) -> str:
    """Classify ATM IV into regime (mirrors options_diagnostics.py)."""
    import math

    if math.isnan(atm_iv):
        return ""
    if atm_iv >= 2.00:
        return "EXTREME"
    if atm_iv >= 0.60:
        return "ELEVATED"
    return "NORMAL"


def _pick_thresholds(p_iv: float, c_iv: float) -> Dict[str, float]:
    """Return regime-appropriate thresholds based on the higher of the two IVs.

    EXTREME-regime names (IV > 200%) get relative IV thresholds and much wider
    RR/skew bands to suppress routine jitter.
    """
    import math

    ref_iv = max(p_iv, c_iv) if not (math.isnan(p_iv) or math.isnan(c_iv)) else 0.0
    if ref_iv >= 2.00:
        return {
            "iv_jump": IV_JUMP_REL_THRESHOLD * ref_iv,
            "iv_jump_large": IV_JUMP_REL_LARGE * ref_iv,
            "rr_flip": RR_FLIP_THRESHOLD_EXTREME,
            "rr_move": RR_MOVE_THRESHOLD_EXTREME,
            "skew_shift": SKEW_SHIFT_THRESHOLD_EXTREME,
        }
    return {
        "iv_jump": IV_JUMP_THRESHOLD,
        "iv_jump_large": IV_JUMP_LARGE,
        "rr_flip": RR_FLIP_THRESHOLD,
        "rr_move": RR_MOVE_THRESHOLD,
        "skew_shift": SKEW_SHIFT_THRESHOLD,
    }


def compute_delta(
    ticker: str,
    prior: Dict[str, Any],
    current: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Compute surface delta for one ticker between two snapshots.

    Returns a delta dict if any threshold is breached, else None.
    Skips names where neither snapshot has opt_use_for_judgment == YES
    (illiquid chains produce noise, not signal).
    """
    import math

    # Both must have data
    if (
        prior.get("opt_liquidity_state", "absent") == "absent"
        or current.get("opt_liquidity_state", "absent") == "absent"
    ):
        return None

    # Gate: at least one snapshot must be judgment-grade
    if prior.get("opt_use_for_judgment") != "YES" and current.get("opt_use_for_judgment") != "YES":
        return None

    p_iv = _sf(prior.get("opt_atm_iv"))
    c_iv = _sf(current.get("opt_atm_iv"))
    p_rr = _sf(prior.get("opt_rr_25d"))
    c_rr = _sf(current.get("opt_rr_25d"))
    p_skew = _sf(prior.get("opt_put_call_skew"))
    c_skew = _sf(current.get("opt_put_call_skew"))
    p_slope = _sf(prior.get("opt_term_slope"))
    c_slope = _sf(current.get("opt_term_slope"))

    th = _pick_thresholds(p_iv, c_iv)

    flags: List[str] = []
    severity = "info"  # info < watch < alert

    delta = {
        "ticker": ticker,
        "catalyst_days": current.get("catalyst_days", ""),
        "catalyst_bucket": current.get("catalyst_bucket", ""),
    }

    # --- ATM IV jump ---
    if not math.isnan(p_iv) and not math.isnan(c_iv):
        iv_change = c_iv - p_iv
        iv_change_abs = abs(iv_change)
        delta["prior_atm_iv"] = round(p_iv, 4)
        delta["current_atm_iv"] = round(c_iv, 4)
        delta["atm_iv_change"] = round(iv_change, 4)

        if iv_change_abs >= th["iv_jump_large"]:
            direction = "up" if iv_change > 0 else "down"
            flags.append(f"iv_jump_large_{direction}")
            severity = "alert"
        elif iv_change_abs >= th["iv_jump"]:
            direction = "up" if iv_change > 0 else "down"
            flags.append(f"iv_jump_{direction}")
            severity = max(severity, "watch", key=["info", "watch", "alert"].index)

        # Regime transition (only NORMAL↔ELEVATED or *→EXTREME is meaningful)
        p_regime = classify_iv_regime(p_iv)
        c_regime = classify_iv_regime(c_iv)
        if p_regime and c_regime and p_regime != c_regime:
            flags.append(f"regime_{p_regime.lower()}_to_{c_regime.lower()}")
            severity = max(severity, "watch", key=["info", "watch", "alert"].index)
    else:
        delta["prior_atm_iv"] = ""
        delta["current_atm_iv"] = ""
        delta["atm_iv_change"] = ""

    # --- Risk reversal ---
    if not math.isnan(p_rr) and not math.isnan(c_rr):
        rr_change = c_rr - p_rr
        delta["prior_rr_25d"] = round(p_rr, 4)
        delta["current_rr_25d"] = round(c_rr, 4)
        delta["rr_25d_change"] = round(rr_change, 4)

        # Sign flip: was positive (call skew), now negative (put skew), or vice versa
        sign_flipped = (p_rr > 0 and c_rr < 0) or (p_rr < 0 and c_rr > 0)
        if sign_flipped and abs(rr_change) >= th["rr_flip"]:
            direction = "bearish" if c_rr < 0 else "bullish"
            flags.append(f"rr_flipped_{direction}")
            severity = "alert"
        elif abs(rr_change) >= th["rr_move"]:
            direction = "bearish" if rr_change < 0 else "bullish"
            flags.append(f"rr_move_{direction}")
            severity = max(severity, "watch", key=["info", "watch", "alert"].index)
    else:
        delta["prior_rr_25d"] = ""
        delta["current_rr_25d"] = ""
        delta["rr_25d_change"] = ""

    # --- Put/call skew ---
    if not math.isnan(p_skew) and not math.isnan(c_skew):
        skew_change = c_skew - p_skew
        delta["prior_skew"] = round(p_skew, 4)
        delta["current_skew"] = round(c_skew, 4)
        delta["skew_change"] = round(skew_change, 4)

        if abs(skew_change) >= th["skew_shift"]:
            direction = "puts_bid" if skew_change > 0 else "calls_bid"
            flags.append(f"skew_shift_{direction}")
            severity = max(severity, "watch", key=["info", "watch", "alert"].index)
    else:
        delta["prior_skew"] = ""
        delta["current_skew"] = ""
        delta["skew_change"] = ""

    # --- Term structure flip ---
    if not math.isnan(p_slope) and not math.isnan(c_slope):
        delta["prior_term_slope"] = round(p_slope, 4)
        delta["current_term_slope"] = round(c_slope, 4)

        was_backwardation = p_slope < TERM_SLOPE_FLIP_THRESHOLD
        now_backwardation = c_slope < TERM_SLOPE_FLIP_THRESHOLD
        if was_backwardation != now_backwardation:
            if now_backwardation:
                flags.append("term_entered_backwardation")
            else:
                flags.append("term_exited_backwardation")
            severity = max(severity, "watch", key=["info", "watch", "alert"].index)
    else:
        delta["prior_term_slope"] = ""
        delta["current_term_slope"] = ""

    if not flags:
        return None

    delta["flags"] = flags
    delta["severity"] = severity
    delta["n_flags"] = len(flags)
    return delta


# ---------------------------------------------------------------------------
# Briefing output
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"alert": 0, "watch": 1, "info": 2}


def format_briefing(
    deltas: List[Dict[str, Any]],
    prior_date: str,
    current_date: str,
    n_prior: int,
    n_current: int,
    n_common: int,
    live_mode: bool,
) -> str:
    """Format delta list into a markdown briefing."""
    lines: List[str] = []

    source_label = "live Tastytrade" if live_mode else f"snapshot {current_date}"
    lines.append(f"# Surface Delta Monitor — {current_date}")
    lines.append("")
    lines.append(
        f"Prior: {prior_date} ({n_prior} names) | Current: {source_label} ({n_current} names) | Overlap: {n_common}"
    )
    lines.append("")

    if not deltas:
        lines.append("**No significant surface shifts detected.**")
        return "\n".join(lines)

    alerts = [d for d in deltas if d["severity"] == "alert"]
    watches = [d for d in deltas if d["severity"] == "watch"]

    if alerts:
        lines.append(f"## ALERT — {len(alerts)} name{'s' if len(alerts) != 1 else ''}")
        lines.append("")
        for d in alerts:
            lines.append(_format_delta_line(d))
        lines.append("")

    if watches:
        lines.append(f"## WATCH — {len(watches)} name{'s' if len(watches) != 1 else ''}")
        lines.append("")
        for d in watches:
            lines.append(_format_delta_line(d))
        lines.append("")

    # Summary stats
    lines.append("---")
    lines.append(f"Total flagged: {len(deltas)} ({len(alerts)} alert, {len(watches)} watch)")

    return "\n".join(lines)


def _format_delta_line(d: Dict[str, Any]) -> str:
    """Format one delta entry as a markdown bullet."""
    ticker = d["ticker"]
    cat_days = d.get("catalyst_days", "")
    cat_bucket = d.get("catalyst_bucket", "")
    cat_info = f" [{cat_bucket} {cat_days}d]" if cat_days and cat_bucket else ""

    parts = [f"- **{ticker}**{cat_info}:"]

    for flag in d["flags"]:
        if flag.startswith("iv_jump_large_"):
            direction = flag.split("_")[-1]
            change = d.get("atm_iv_change", 0)
            prior = d.get("prior_atm_iv", "?")
            current = d.get("current_atm_iv", "?")
            parts.append(f"ATM IV {direction} {abs(change) * 100:.0f}pp ({prior:.0%} → {current:.0%})")
        elif flag.startswith("iv_jump_"):
            direction = flag.split("_")[-1]
            change = d.get("atm_iv_change", 0)
            parts.append(f"IV {direction} {abs(change) * 100:.0f}pp")
        elif flag.startswith("rr_flipped_"):
            direction = flag.split("_")[-1]
            prior = d.get("prior_rr_25d", "?")
            current = d.get("current_rr_25d", "?")
            parts.append(f"25d RR flipped {direction} ({prior:+.2%} → {current:+.2%})")
        elif flag.startswith("rr_move_"):
            direction = flag.split("_")[-1]
            change = d.get("rr_25d_change", 0)
            parts.append(f"RR shift {direction} ({change:+.2%})")
        elif flag.startswith("skew_shift_"):
            direction = flag.split("_")[-1].replace("_", " ")
            change = d.get("skew_change", 0)
            parts.append(f"skew shift ({direction}, {change:+.4f})")
        elif flag.startswith("regime_"):
            # regime_normal_to_elevated
            regime_str = flag.replace("regime_", "").replace("_to_", " → ")
            parts.append(f"IV regime {regime_str}")
        elif "backwardation" in flag:
            if "entered" in flag:
                parts.append("entered backwardation (event premium)")
            else:
                parts.append("exited backwardation")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Live fetch (reuses existing options_diagnostics infrastructure)
# ---------------------------------------------------------------------------


def fetch_live_diagnostics(
    prior_tickers: List[str],
    as_of_date: str,
) -> Dict[str, Dict[str, Any]]:
    """Fetch live Tastytrade diagnostics for tickers from prior snapshot.

    Reuses fetch_options_diagnostics from common/options_diagnostics.py.
    """
    from common.options_diagnostics import fetch_options_diagnostics

    logger.info("Fetching live diagnostics for %d tickers...", len(prior_tickers))
    return fetch_options_diagnostics(prior_tickers, as_of_date)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DELTA_CSV_COLUMNS = [
    "ticker",
    "severity",
    "n_flags",
    "catalyst_days",
    "catalyst_bucket",
    "prior_atm_iv",
    "current_atm_iv",
    "atm_iv_change",
    "prior_rr_25d",
    "current_rr_25d",
    "rr_25d_change",
    "prior_skew",
    "current_skew",
    "skew_change",
    "prior_term_slope",
    "current_term_slope",
    "flags",
]


def run(
    as_of_date: Optional[str] = None,
    prior_date: Optional[str] = None,
    live: bool = True,
    dry_run: bool = False,
    json_only: bool = False,
) -> Dict[str, Any]:
    """Run the surface delta monitor.

    Parameters
    ----------
    as_of_date : current date (defaults to today)
    prior_date : explicit prior snapshot date (defaults to auto-find)
    live : if True, fetch live Tastytrade data; if False, use today's snapshot
    dry_run : if True, print briefing but don't write artifacts
    json_only : if True, skip markdown output to stdout

    Returns
    -------
    Result dict with schema version, dates, counts, and deltas list.
    """
    snapshots_root = REPO_ROOT / "data" / "snapshots"

    if as_of_date is None:
        as_of_date = date.today().isoformat()

    # --- Find prior snapshot ---
    # Errors raise RuntimeError, never sys.exit: run() is called in-process by
    # run_daily_production's non-blocking step, and SystemExit would sail past
    # its `except Exception` and fail the whole pipeline (2026-07-15 outage).
    if prior_date:
        prior_dir = snapshots_root / prior_date
        if not prior_dir.exists():
            raise RuntimeError(f"Prior snapshot not found: {prior_dir}")
        prior_diag = load_diagnostics_csv(prior_dir / "options_diagnostics.csv")
        if not prior_diag:
            raise RuntimeError(f"No diagnostics data in prior snapshot {prior_date}")
    else:
        # Auto-find: most recent prior snapshot that actually HAS diagnostics,
        # so a crash-fragment day doesn't blind the monitor.
        prior_dir = None
        prior_diag: Dict[str, Dict[str, Any]] = {}
        for _candidate in find_prior_snapshots(snapshots_root, as_of_date):
            _diag = load_diagnostics_csv(_candidate / "options_diagnostics.csv")
            if _diag:
                prior_dir, prior_diag = _candidate, _diag
                break
        if prior_dir is None:
            raise RuntimeError(f"No prior snapshot with options diagnostics found before {as_of_date}")
        prior_date = prior_dir.name
    logger.info("Loaded prior diagnostics: %s (%d names)", prior_date, len(prior_diag))

    # --- Get current diagnostics ---
    if live:
        # Fetch live data for tickers that had data in prior snapshot
        live_tickers = sorted(t for t, d in prior_diag.items() if d.get("opt_liquidity_state", "absent") != "absent")
        current_diag = fetch_live_diagnostics(live_tickers, as_of_date)
        # Carry forward catalyst context from prior snapshot for briefing display
        for ticker, diag in current_diag.items():
            if ticker in prior_diag:
                if "catalyst_days" not in diag or diag.get("catalyst_days") == "":
                    diag["catalyst_days"] = prior_diag[ticker].get("catalyst_days", "")
                if "catalyst_bucket" not in diag or diag.get("catalyst_bucket") == "":
                    diag["catalyst_bucket"] = prior_diag[ticker].get("catalyst_bucket", "")
    else:
        current_dir = snapshots_root / as_of_date
        if not current_dir.exists():
            raise RuntimeError(f"Current snapshot not found: {current_dir}")
        current_diag = load_diagnostics_csv(current_dir / "options_diagnostics.csv")

    if not current_diag:
        raise RuntimeError("No current diagnostics available")
    logger.info("Current diagnostics: %d names", len(current_diag))

    # --- Compute deltas for overlapping tickers ---
    common_tickers = sorted(set(prior_diag) & set(current_diag))
    deltas: List[Dict[str, Any]] = []
    for ticker in common_tickers:
        d = compute_delta(ticker, prior_diag[ticker], current_diag[ticker])
        if d is not None:
            deltas.append(d)

    # Sort: alerts first, then watches, then by IV change magnitude
    deltas.sort(
        key=lambda d: (
            SEVERITY_ORDER.get(d["severity"], 9),
            -d.get("n_flags", 0),
            -abs(d.get("atm_iv_change", 0) or 0),
        )
    )

    logger.info(
        "Deltas: %d flagged / %d compared (alerts=%d, watch=%d)",
        len(deltas),
        len(common_tickers),
        sum(1 for d in deltas if d["severity"] == "alert"),
        sum(1 for d in deltas if d["severity"] == "watch"),
    )

    # --- Build result ---
    result = {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prior_date": prior_date,
        "current_date": as_of_date,
        "live_mode": live,
        "n_prior": len(prior_diag),
        "n_current": len(current_diag),
        "n_compared": len(common_tickers),
        "n_flagged": len(deltas),
        "n_alert": sum(1 for d in deltas if d["severity"] == "alert"),
        "n_watch": sum(1 for d in deltas if d["severity"] == "watch"),
        "thresholds": {
            "iv_jump": IV_JUMP_THRESHOLD,
            "iv_jump_large": IV_JUMP_LARGE,
            "rr_flip": RR_FLIP_THRESHOLD,
            "rr_move": RR_MOVE_THRESHOLD,
            "skew_shift": SKEW_SHIFT_THRESHOLD,
        },
        "deltas": deltas,
    }

    # --- Output ---
    briefing = format_briefing(
        deltas,
        prior_date,
        as_of_date,
        len(prior_diag),
        len(current_diag),
        len(common_tickers),
        live_mode=live,
    )

    if not json_only:
        print(briefing)

    if not dry_run:
        out_dir = snapshots_root / as_of_date
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = out_dir / "surface_delta.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("Wrote %s", json_path)

        # CSV
        csv_path = out_dir / "surface_delta.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DELTA_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for d in deltas:
                row = dict(d)
                row["flags"] = "|".join(d.get("flags", []))
                writer.writerow(row)
        logger.info("Wrote %s", csv_path)

        # Markdown
        md_path = out_dir / "surface_delta.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(briefing)
        logger.info("Wrote %s", md_path)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Post-open options surface delta monitor",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Current date (YYYY-MM-DD, default: today)",
    )
    parser.add_argument(
        "--prior-date",
        default=None,
        help="Explicit prior snapshot date (default: auto-find latest before as-of-date)",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Use today's snapshot instead of live Tastytrade fetch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print briefing only, don't write artifacts",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Write artifacts but skip markdown to stdout",
    )
    args = parser.parse_args()

    try:
        result = run(
            as_of_date=args.as_of_date,
            prior_date=args.prior_date,
            live=not args.no_live,
            dry_run=args.dry_run,
            json_only=args.json_only,
        )
    except RuntimeError as err:
        logger.error("%s", err)
        sys.exit(1)

    # Exit code: 0 if no alerts, 2 if watch-only, 1 if alerts
    if result["n_alert"] > 0:
        sys.exit(1)
    if result["n_watch"] > 0:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
