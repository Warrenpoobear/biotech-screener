"""
PIT gap forward-return assembly — Method A (primary) + Method B (sensitivity).

Method A: same-archive basis; horizons 1d/3d/5d/20d ONLY; no 60d computation.
Method B: single May 7 archive basis; horizons 1d/3d/5d/20d/60d;
          all rows labeled SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE.

Outputs written to artifacts/audit/ (gitignored).
No production files read or modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Governance constants
# ---------------------------------------------------------------------------

GAP_START = "2026-01-16"
GAP_END = "2026-05-07"
ATXS_EXCLUSION_AFTER = "2026-01-23"
METHOD_B_ARCHIVE = "2026-05-07"
SENSITIVITY_LABEL = "SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE"
HORIZONS_A = [1, 3, 5, 20]  # NO 60d for Method A
HORIZONS_B = [1, 3, 5, 20, 60]  # Sensitivity only
COVERAGE_THRESHOLD = 28
CONTINUITY_JUMP_THRESHOLD = 0.50

# Acceptance thresholds (log only — do not assert conclusions)
THRESHOLD_A_5D = 40
THRESHOLD_A_20D = 25
THRESHOLD_B_60D = 20


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def repo_root() -> Path:
    this = Path(__file__).resolve()
    # scripts/research/pit_gap_forward_returns.py → repo root is 2 levels up
    return this.parent.parent.parent


def snapshots_dir() -> Path:
    return repo_root() / "data" / "snapshots"


def archives_dir() -> Path:
    return repo_root() / "data" / "pit_archives"


def audit_dir() -> Path:
    return repo_root() / "artifacts" / "audit"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_rankings(snap_date: str) -> list[dict]:
    path = snapshots_dir() / snap_date / "rankings.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rank_str = row.get("actionable_rank", "")
            try:
                rank = int(rank_str)
            except (ValueError, TypeError):
                continue
            if 1 <= rank <= 30:
                rows.append(
                    {
                        "ticker": row["ticker"],
                        "actionable_rank": rank,
                        "target_weight_pct": row.get("target_weight_pct", ""),
                    }
                )
    return rows


def load_prices(arch_date: str) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Return ({ticker: {date_str: close}}, sorted_trading_dates)."""
    path = archives_dir() / arch_date / "price_history.csv"
    if not path.exists():
        log.warning("price_history.csv missing for archive %s", arch_date)
        return {}, []
    prices: dict[str, dict[str, float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = row.get("ticker", "")
            d = row.get("date", "")
            c_str = row.get("close", "")
            if not t or not d or not c_str:
                continue
            try:
                c = float(c_str)
            except (ValueError, TypeError):
                continue
            if t not in prices:
                prices[t] = {}
            prices[t][d] = c
    all_dates: set[str] = set()
    for ticker_dates in prices.values():
        all_dates.update(ticker_dates.keys())
    sorted_dates = sorted(all_dates)
    return prices, sorted_dates


# ---------------------------------------------------------------------------
# Manifest SHA256 check
# ---------------------------------------------------------------------------


def check_manifest(arch_date: str) -> str:
    """Return PASS, STALE_MANIFEST, or MISSING_MANIFEST."""
    manifest_path = archives_dir() / arch_date / "manifest.json"
    price_path = archives_dir() / arch_date / "price_history.csv"
    if not manifest_path.exists():
        return "MISSING_MANIFEST"
    if not price_path.exists():
        return "MISSING_MANIFEST"
    with open(manifest_path) as f:
        try:
            manifest = json.load(f)
        except json.JSONDecodeError:
            return "MISSING_MANIFEST"
    expected_hash = manifest.get("files", {}).get("price_history.csv", {}).get("sha256")
    if not expected_hash:
        return "MISSING_MANIFEST"
    h = hashlib.sha256()
    with open(price_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    actual_hash = h.hexdigest()
    if actual_hash == expected_hash:
        return "PASS"
    return "STALE_MANIFEST"


# ---------------------------------------------------------------------------
# Resolve archive (Method A only)
# ---------------------------------------------------------------------------


def resolve_archive(snap_date: str) -> tuple[Optional[str], bool]:
    """Return (arch_date, is_fallback). Returns (None, False) if no archive found."""
    direct = archives_dir() / snap_date
    if direct.exists() and (direct / "price_history.csv").exists():
        return snap_date, False
    available = sorted(
        d.name
        for d in archives_dir().iterdir()
        if d.is_dir() and d.name < snap_date and (d / "price_history.csv").exists()
    )
    if not available:
        log.warning("No archive found on or before %s — skipping", snap_date)
        return None, False
    return available[-1], True


# ---------------------------------------------------------------------------
# Core return computation (spec §5.7)
# ---------------------------------------------------------------------------


def resolve_anchor(
    ticker: str,
    snap_date: str,
    prices: dict[str, dict[str, float]],
    sorted_dates: list[str],
) -> tuple[Optional[float], Optional[str]]:
    ticker_prices = prices.get(ticker, {})
    if snap_date in ticker_prices:
        return ticker_prices[snap_date], snap_date
    candidates = [d for d in sorted_dates if d <= snap_date and d in ticker_prices]
    if candidates:
        best = candidates[-1]
        return ticker_prices[best], best
    return None, None


def compute_return(
    ticker: str,
    anchor_date: Optional[str],
    horizon: int,
    prices: dict[str, dict[str, float]],
    anchor_close: Optional[float],
    sorted_dates: list[str],
) -> Optional[float]:
    if anchor_date is None or anchor_close is None:
        return None
    try:
        idx = sorted_dates.index(anchor_date)
    except ValueError:
        return None
    fwd_idx = idx + horizon
    if fwd_idx >= len(sorted_dates):
        return None
    fwd_date = sorted_dates[fwd_idx]
    fwd_close = prices.get(ticker, {}).get(fwd_date)
    if fwd_close is None:
        return None
    return (fwd_close - anchor_close) / anchor_close


# ---------------------------------------------------------------------------
# Continuity flagging (§7.7)
# ---------------------------------------------------------------------------


def flag_continuity(arch_date: str, prices: dict[str, dict[str, float]], sorted_dates: list[str]) -> list[dict]:
    flags = []
    for ticker, tprices in prices.items():
        ticker_dates = sorted(d for d in sorted_dates if d in tprices)
        for i in range(len(ticker_dates) - 1):
            d0, d1 = ticker_dates[i], ticker_dates[i + 1]
            p0, p1 = tprices[d0], tprices[d1]
            if p0 > 0:
                move = abs(p1 / p0 - 1)
                if move > CONTINUITY_JUMP_THRESHOLD:
                    flags.append(
                        {
                            "arch_date": arch_date,
                            "ticker": ticker,
                            "date_from": d0,
                            "date_to": d1,
                            "price_from": p0,
                            "price_to": p1,
                            "magnitude": round(move, 4),
                        }
                    )
    return flags


# ---------------------------------------------------------------------------
# Gap snapshot discovery
# ---------------------------------------------------------------------------


def get_gap_snapshots() -> list[str]:
    snap_dir = snapshots_dir()
    dates = []
    for d in snap_dir.iterdir():
        name = d.name
        if not (len(name) == 10 and name[4] == "-" and name[7] == "-"):
            continue
        if not (GAP_START <= name <= GAP_END):
            continue
        if (d / "rankings.csv").exists():
            dates.append(name)
    return sorted(dates)


# ---------------------------------------------------------------------------
# Method A
# ---------------------------------------------------------------------------


def run_method_a(
    gap_dates: list[str],
    run_date: str,
) -> tuple[list[dict], dict]:
    """
    Returns (rows, validation_summary).
    No 60d horizon. No 60d conclusion.
    """
    rows = []
    archive_cache: dict[str, tuple[dict, list[str]]] = {}
    manifest_results: dict[str, str] = {}
    continuity_flags: list[dict] = []
    skipped_snapshots = []
    fallback_count = 0

    log.info("Method A: processing %d gap snapshots", len(gap_dates))

    for snap_date in gap_dates:
        arch_date, is_fallback = resolve_archive(snap_date)
        if arch_date is None:
            skipped_snapshots.append(snap_date)
            continue
        if is_fallback:
            fallback_count += 1
            log.info("  [Method A] %s → fallback archive %s", snap_date, arch_date)

        if arch_date not in archive_cache:
            manifest_result = check_manifest(arch_date)
            manifest_results[arch_date] = manifest_result
            if manifest_result != "PASS":
                log.warning(
                    "  [Method A] Manifest %s for archive %s (continuing per spec §7.8)",
                    manifest_result,
                    arch_date,
                )
            prices, sorted_dates = load_prices(arch_date)
            archive_cache[arch_date] = (prices, sorted_dates)
            # Continuity flags per archive (only flag once per archive)
            flags = flag_continuity(arch_date, prices, sorted_dates)
            continuity_flags.extend(flags)
        else:
            prices, sorted_dates = archive_cache[arch_date]

        top30 = load_rankings(snap_date)
        if not top30:
            log.warning("  [Method A] No top-30 rows for %s", snap_date)

        # XBI anchor
        xbi_anchor_close, xbi_anchor_date = resolve_anchor("XBI", snap_date, prices, sorted_dates)
        if xbi_anchor_close is None:
            log.warning("  [Method A] XBI anchor null for %s — all excess returns null", snap_date)
        xbi_returns = {}
        for h in HORIZONS_A:
            xbi_returns[h] = compute_return("XBI", xbi_anchor_date, h, prices, xbi_anchor_close, sorted_dates)

        # Per-ticker rows
        for stock in top30:
            ticker = stock["ticker"]
            atxs_excluded = ticker == "ATXS" and snap_date > ATXS_EXCLUSION_AFTER
            if atxs_excluded:
                row = {
                    "snap_date": snap_date,
                    "ticker": ticker,
                    "actionable_rank": stock["actionable_rank"],
                    "target_weight_pct": stock["target_weight_pct"],
                    "archive_date": arch_date,
                    "archive_fallback": is_fallback,
                    "anchor_date": None,
                    "anchor_close": None,
                    "atxs_excluded": True,
                }
                for h in HORIZONS_A:
                    row[f"actual_return_{h}d"] = None
                    row[f"xbi_return_{h}d"] = None
                    row[f"excess_return_{h}d"] = None
                row["forward_complete_5d"] = False
                row["forward_complete_20d"] = False
                rows.append(row)
                continue

            anchor_close, anchor_date = resolve_anchor(ticker, snap_date, prices, sorted_dates)
            actual_returns = {}
            excess_returns = {}
            for h in HORIZONS_A:
                actual_returns[h] = compute_return(ticker, anchor_date, h, prices, anchor_close, sorted_dates)
                xr = xbi_returns.get(h)
                ar = actual_returns[h]
                excess_returns[h] = (ar - xr) if (ar is not None and xr is not None) else None

            row = {
                "snap_date": snap_date,
                "ticker": ticker,
                "actionable_rank": stock["actionable_rank"],
                "target_weight_pct": stock["target_weight_pct"],
                "archive_date": arch_date,
                "archive_fallback": is_fallback,
                "anchor_date": anchor_date,
                "anchor_close": anchor_close,
                "atxs_excluded": False,
            }
            for h in HORIZONS_A:
                row[f"actual_return_{h}d"] = actual_returns[h]
                row[f"xbi_return_{h}d"] = xbi_returns.get(h)
                row[f"excess_return_{h}d"] = excess_returns[h]
            row["forward_complete_5d"] = actual_returns.get(5) is not None
            row["forward_complete_20d"] = actual_returns.get(20) is not None
            rows.append(row)

    # Build validation summary
    manifest_counts = {"PASS": 0, "STALE_MANIFEST": 0, "MISSING_MANIFEST": 0}
    for v in manifest_results.values():
        manifest_counts[v] += 1

    summary = {
        "method": "A",
        "gap_snapshots_found": len(gap_dates),
        "gap_snapshots_skipped": len(skipped_snapshots),
        "fallback_archives_used": fallback_count,
        "archives_loaded": len(archive_cache),
        "manifest_pass": manifest_counts["PASS"],
        "manifest_stale": manifest_counts["STALE_MANIFEST"],
        "manifest_missing": manifest_counts["MISSING_MANIFEST"],
        "continuity_flags": len(continuity_flags),
        "continuity_flag_details": continuity_flags,
        "skipped_snapshots": skipped_snapshots,
    }

    # Acceptance threshold logging (no assertions — log only)
    snaps_with_5d = sum(
        1
        for snap in gap_dates
        if any(r["snap_date"] == snap and r["anchor_close"] is not None and r["forward_complete_5d"] for r in rows)
    )
    snaps_with_20d = sum(
        1
        for snap in gap_dates
        if any(r["snap_date"] == snap and r["anchor_close"] is not None and r["forward_complete_20d"] for r in rows)
    )
    log.info(
        "[Method A] Acceptance: 5d qualified snapshots=%d (threshold=%d) | 20d=%d (threshold=%d)",
        snaps_with_5d,
        THRESHOLD_A_5D,
        snaps_with_20d,
        THRESHOLD_A_20D,
    )
    log.info("[Method A] NO 60d horizon — no 60d conclusion possible from Method A")

    summary["acceptance_5d_qualified_snapshots"] = snaps_with_5d
    summary["acceptance_20d_qualified_snapshots"] = snaps_with_20d
    summary["acceptance_5d_threshold"] = THRESHOLD_A_5D
    summary["acceptance_20d_threshold"] = THRESHOLD_A_20D
    summary["acceptance_5d_meets_threshold"] = snaps_with_5d >= THRESHOLD_A_5D
    summary["acceptance_20d_meets_threshold"] = snaps_with_20d >= THRESHOLD_A_20D
    summary["no_60d_conclusion"] = True  # Enforced invariant

    return rows, summary


# ---------------------------------------------------------------------------
# Method B
# ---------------------------------------------------------------------------


def run_method_b(
    gap_dates: list[str],
    run_date: str,
) -> tuple[list[dict], dict]:
    """
    Returns (rows, validation_summary).
    Single May 7 archive basis. All rows labeled SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE.
    """
    arch_date = METHOD_B_ARCHIVE
    arch_path = archives_dir() / arch_date / "price_history.csv"
    if not arch_path.exists():
        log.error("[Method B] PRECONDITION FAILED: %s not found", arch_path)
        return [], {"method": "B", "error": f"May 7 archive missing: {arch_path}"}

    manifest_result = check_manifest(arch_date)
    log.info("[Method B] Manifest check for %s: %s", arch_date, manifest_result)

    prices, sorted_dates = load_prices(arch_date)

    # Confirm last_date in archive is 2026-05-07
    last_date = sorted_dates[-1] if sorted_dates else "UNKNOWN"
    log.info("[Method B] Archive last_date: %s (expected %s)", last_date, arch_date)

    continuity_flags = flag_continuity(arch_date, prices, sorted_dates)

    rows = []
    log.info("Method B: processing %d gap snapshots (single archive basis)", len(gap_dates))

    for snap_date in gap_dates:
        top30 = load_rankings(snap_date)
        if not top30:
            log.warning("  [Method B] No top-30 rows for %s", snap_date)

        xbi_anchor_close, xbi_anchor_date = resolve_anchor("XBI", snap_date, prices, sorted_dates)
        if xbi_anchor_close is None:
            log.warning("  [Method B] XBI anchor null for %s", snap_date)
        xbi_returns = {}
        for h in HORIZONS_B:
            xbi_returns[h] = compute_return("XBI", xbi_anchor_date, h, prices, xbi_anchor_close, sorted_dates)

        for stock in top30:
            ticker = stock["ticker"]
            atxs_excluded = ticker == "ATXS" and snap_date > ATXS_EXCLUSION_AFTER
            if atxs_excluded:
                row = {
                    "snap_date": snap_date,
                    "ticker": ticker,
                    "actionable_rank": stock["actionable_rank"],
                    "target_weight_pct": stock["target_weight_pct"],
                    "archive_date": arch_date,
                    "archive_fallback": False,
                    "anchor_date": None,
                    "anchor_close": None,
                    "atxs_excluded": True,
                    "sensitivity_label": SENSITIVITY_LABEL,
                }
                for h in HORIZONS_B:
                    row[f"actual_return_{h}d"] = None
                    row[f"xbi_return_{h}d"] = None
                    row[f"excess_return_{h}d"] = None
                row["forward_complete_5d"] = False
                row["forward_complete_20d"] = False
                row["forward_complete_60d"] = False
                rows.append(row)
                continue

            anchor_close, anchor_date = resolve_anchor(ticker, snap_date, prices, sorted_dates)
            actual_returns = {}
            excess_returns = {}
            for h in HORIZONS_B:
                actual_returns[h] = compute_return(ticker, anchor_date, h, prices, anchor_close, sorted_dates)
                xr = xbi_returns.get(h)
                ar = actual_returns[h]
                excess_returns[h] = (ar - xr) if (ar is not None and xr is not None) else None

            row = {
                "snap_date": snap_date,
                "ticker": ticker,
                "actionable_rank": stock["actionable_rank"],
                "target_weight_pct": stock["target_weight_pct"],
                "archive_date": arch_date,
                "archive_fallback": False,
                "anchor_date": anchor_date,
                "anchor_close": anchor_close,
                "atxs_excluded": False,
                "sensitivity_label": SENSITIVITY_LABEL,
            }
            for h in HORIZONS_B:
                row[f"actual_return_{h}d"] = actual_returns[h]
                row[f"xbi_return_{h}d"] = xbi_returns.get(h)
                row[f"excess_return_{h}d"] = excess_returns[h]
            row["forward_complete_5d"] = actual_returns.get(5) is not None
            row["forward_complete_20d"] = actual_returns.get(20) is not None
            row["forward_complete_60d"] = actual_returns.get(60) is not None
            rows.append(row)

    # Validate all rows carry the sensitivity label
    rows_missing_label = [r for r in rows if r.get("sensitivity_label") != SENSITIVITY_LABEL]
    if rows_missing_label:
        log.error("[Method B] FAIL: %d rows missing sensitivity_label", len(rows_missing_label))
    else:
        log.info("[Method B] PASS: all %d rows carry sensitivity_label", len(rows))

    # Validate all rows have archive_date = "2026-05-07"
    rows_wrong_archive = [r for r in rows if r.get("archive_date") != arch_date]
    if rows_wrong_archive:
        log.error("[Method B] FAIL: %d rows have wrong archive_date", len(rows_wrong_archive))
    else:
        log.info("[Method B] PASS: all rows have archive_date=%s", arch_date)

    # Acceptance threshold for 60d
    snaps_with_60d = sum(
        1
        for snap in gap_dates
        if any(r["snap_date"] == snap and r["anchor_close"] is not None and r["forward_complete_60d"] for r in rows)
    )
    log.info(
        "[Method B] Acceptance: 60d qualified snapshots=%d (threshold=%d)",
        snaps_with_60d,
        THRESHOLD_B_60D,
    )

    summary = {
        "method": "B",
        "archive_date": arch_date,
        "archive_last_date": last_date,
        "manifest_result": manifest_result,
        "gap_snapshots_found": len(gap_dates),
        "total_rows": len(rows),
        "rows_with_sensitivity_label": len(rows) - len(rows_missing_label),
        "rows_missing_sensitivity_label": len(rows_missing_label),
        "rows_with_correct_archive": len(rows) - len(rows_wrong_archive),
        "rows_with_wrong_archive": len(rows_wrong_archive),
        "continuity_flags": len(continuity_flags),
        "continuity_flag_details": continuity_flags,
        "acceptance_60d_qualified_snapshots": snaps_with_60d,
        "acceptance_60d_threshold": THRESHOLD_B_60D,
        "acceptance_60d_meets_threshold": snaps_with_60d >= THRESHOLD_B_60D,
        "sensitivity_label_check": "PASS" if not rows_missing_label else "FAIL",
        "single_archive_basis_check": "PASS" if not rows_wrong_archive else "FAIL",
    }
    return rows, summary


# ---------------------------------------------------------------------------
# Validation checks (§7)
# ---------------------------------------------------------------------------


def run_validation_checks(
    method_a_rows: list[dict],
    method_b_rows: list[dict],
    gap_dates: list[str],
    summary_a: dict,
    summary_b: dict,
) -> dict:
    results = {}

    # §7.1 Archive date resolution
    results["v1_archive_resolution"] = {
        "gap_snapshots": len(gap_dates),
        "skipped": summary_a.get("gap_snapshots_skipped", 0),
        "fallback_count": summary_a.get("fallback_archives_used", 0),
        "status": "PASS" if summary_a.get("gap_snapshots_skipped", 0) == 0 else "WARN_MISSING_ARCHIVES",
    }

    # §7.2 Same-archive basis for Method A
    results["v2_same_archive_basis"] = {
        "status": "PASS",
        "note": "Enforced: each row's prices loaded from a single archive directory (arch_date field).",
    }

    # §7.3 May 7 archive basis for Method B
    last = summary_b.get("archive_last_date", "UNKNOWN")
    results["v3_may7_archive"] = {
        "archive_last_date": last,
        "status": "PASS" if last == METHOD_B_ARCHIVE else f"WARN_LAST_DATE_IS_{last}",
    }

    # §7.4 ATXS exclusion
    atxs_post_errors = []
    atxs_pre_errors = []
    for r in method_a_rows + method_b_rows:
        if r["ticker"] != "ATXS":
            continue
        if r["snap_date"] > ATXS_EXCLUSION_AFTER:
            if r["anchor_close"] is not None or not r["atxs_excluded"]:
                atxs_post_errors.append(r["snap_date"])
        else:
            if r["anchor_close"] is None and not r["atxs_excluded"]:
                atxs_pre_errors.append(r["snap_date"])
    results["v4_atxs_exclusion"] = {
        "post_exclusion_errors": atxs_post_errors,
        "pre_exclusion_null_errors": atxs_pre_errors,
        "status": "PASS" if not atxs_post_errors and not atxs_pre_errors else "FAIL",
    }

    # §7.5 Anchor coverage per snapshot
    low_coverage_snaps = []
    for snap_date in gap_dates:
        snap_rows_a = [r for r in method_a_rows if r["snap_date"] == snap_date and not r["atxs_excluded"]]
        non_null = sum(1 for r in snap_rows_a if r["anchor_close"] is not None)
        if non_null < COVERAGE_THRESHOLD:
            low_coverage_snaps.append({"snap_date": snap_date, "non_null_anchors": non_null})
    results["v5_anchor_coverage"] = {
        "low_coverage_snapshots": low_coverage_snaps,
        "status": "PASS" if not low_coverage_snaps else f"LOW_COVERAGE_{len(low_coverage_snaps)}_SNAPSHOTS",
    }

    # §7.6 XBI coverage
    xbi_null_snaps = []
    for snap_date in gap_dates:
        snap_rows_a = [r for r in method_a_rows if r["snap_date"] == snap_date and r["ticker"] == "XBI"]
        # XBI isn't in top-30; check via xbi_return columns instead
        snap_all = [r for r in method_a_rows if r["snap_date"] == snap_date]
        if snap_all:
            sample_xbi = snap_all[0].get("xbi_return_1d")
            if sample_xbi is None and snap_all[0].get("anchor_close") is not None:
                xbi_null_snaps.append(snap_date)
    results["v6_xbi_coverage"] = {
        "snapshots_with_null_xbi": xbi_null_snaps,
        "status": "PASS" if not xbi_null_snaps else f"WARN_XBI_NULL_{len(xbi_null_snaps)}_SNAPSHOTS",
    }

    # §7.7 Continuity flags
    all_flags = summary_a.get("continuity_flag_details", []) + summary_b.get("continuity_flag_details", [])
    results["v7_continuity"] = {
        "flag_count": len(all_flags),
        "flags": all_flags,
        "binary_event_review_note": (
            "These flags require manual review to confirm they represent real events "
            "(readouts, acquisitions) rather than price data errors."
        ),
        "status": "INFO",
    }

    # §7.8 Manifest SHA256
    results["v8_manifest"] = {
        "method_a_pass": summary_a.get("manifest_pass", 0),
        "method_a_stale": summary_a.get("manifest_stale", 0),
        "method_a_missing": summary_a.get("manifest_missing", 0),
        "method_b_result": summary_b.get("manifest_result", "UNKNOWN"),
        "status": "INFO",
        "note": "STALE_MANIFEST expected for archives rebuilt on 2026-04-10 (per PR #383).",
    }

    # §7.9 No production file modification
    results["v9_no_production_modification"] = {
        "status": "PASS",
        "note": (
            "Enforced by design: script imports only stdlib + csv/json/hashlib/pathlib. "
            "No ranker_engine, selector_engine, decision_engine, run_screen, final_score, "
            "sizing, run_phase2_snapshot_delta imported. No data/snapshots writes."
        ),
    }

    return results


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_method_a_csv(rows: list[dict], run_date: str) -> Path:
    cols = [
        "snap_date",
        "ticker",
        "actionable_rank",
        "target_weight_pct",
        "archive_date",
        "archive_fallback",
        "anchor_date",
        "anchor_close",
        "atxs_excluded",
        "actual_return_1d",
        "actual_return_3d",
        "actual_return_5d",
        "actual_return_20d",
        "xbi_return_1d",
        "xbi_return_3d",
        "xbi_return_5d",
        "xbi_return_20d",
        "excess_return_1d",
        "excess_return_3d",
        "excess_return_5d",
        "excess_return_20d",
        "forward_complete_5d",
        "forward_complete_20d",
    ]
    path = audit_dir() / f"gap_panel_method_a_{run_date}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("Method A output written: %s (%d rows)", path, len(rows))
    return path


def write_method_b_csv(rows: list[dict], run_date: str) -> Path:
    cols = [
        "snap_date",
        "ticker",
        "actionable_rank",
        "target_weight_pct",
        "archive_date",
        "archive_fallback",
        "anchor_date",
        "anchor_close",
        "atxs_excluded",
        "actual_return_1d",
        "actual_return_3d",
        "actual_return_5d",
        "actual_return_20d",
        "actual_return_60d",
        "xbi_return_1d",
        "xbi_return_3d",
        "xbi_return_5d",
        "xbi_return_20d",
        "xbi_return_60d",
        "excess_return_1d",
        "excess_return_3d",
        "excess_return_5d",
        "excess_return_20d",
        "excess_return_60d",
        "forward_complete_5d",
        "forward_complete_20d",
        "forward_complete_60d",
        "sensitivity_label",
    ]
    path = audit_dir() / f"gap_panel_method_b_sensitivity_{run_date}.csv"
    with open(path, "w", newline="") as f:
        f.write("# SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE\n")
        f.write("# Method B: single May 7 archive basis. Not primary results. Do not override Method A.\n")
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info("Method B output written: %s (%d rows)", path, len(rows))
    return path


def write_validation_report(
    validation: dict,
    summary_a: dict,
    summary_b: dict,
    run_date: str,
    path_a: Optional[Path],
    path_b: Optional[Path],
) -> Path:
    path = audit_dir() / f"gap_assembly_validation_{run_date}.md"

    def _status_icon(s: str) -> str:
        if s == "PASS":
            return "PASS"
        if s.startswith("WARN") or s == "INFO":
            return "WARN"
        if s.startswith("FAIL"):
            return "FAIL"
        return s

    a5d_status = "MEETS" if summary_a.get("acceptance_5d_meets_threshold") else "BELOW"
    a20d_status = "MEETS" if summary_a.get("acceptance_20d_meets_threshold") else "BELOW"
    b60d_status = "MEETS" if summary_b.get("acceptance_60d_meets_threshold") else "BELOW"
    path_a_name = path_a.name if path_a else "NOT_WRITTEN"
    path_b_name = path_b.name if path_b else "NOT_WRITTEN"

    lines = [
        "# PIT Gap Forward Return Assembly — Validation Report",
        "",
        f"**Run date:** {run_date}  ",
        f"**Gap period:** {GAP_START} to {GAP_END}  ",
        f"**Method A output:** `{path_a_name}`  ",
        f"**Method B output:** `{path_b_name}`  ",
        "",
        "---",
        "",
        "## Method A Summary",
        "",
        f"- Gap snapshots found: {summary_a.get('gap_snapshots_found', 'N/A')}",
        f"- Skipped (no archive): {summary_a.get('gap_snapshots_skipped', 'N/A')}",
        f"- Fallback archives used: {summary_a.get('fallback_archives_used', 'N/A')}",
        f"- Manifests PASS/STALE/MISSING: {summary_a.get('manifest_pass', 0)}/{summary_a.get('manifest_stale', 0)}/{summary_a.get('manifest_missing', 0)}",
        f"- **5d qualified snapshots:** {summary_a.get('acceptance_5d_qualified_snapshots', 0)} (threshold {THRESHOLD_A_5D}): {a5d_status} threshold",
        f"- **20d qualified snapshots:** {summary_a.get('acceptance_20d_qualified_snapshots', 0)} (threshold {THRESHOLD_A_20D}): {a20d_status} threshold",
        "- **60d:** NOT COMPUTED — Method A makes no 60d conclusion under any circumstance",
        "",
        "## Method B Summary",
        "",
        f"- Archive date: {summary_b.get('archive_date', 'N/A')}",
        f"- Archive last_date: {summary_b.get('archive_last_date', 'N/A')}",
        f"- Manifest: {summary_b.get('manifest_result', 'N/A')}",
        f"- Total rows: {summary_b.get('total_rows', 0)}",
        f"- Rows with SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE label: {summary_b.get('rows_with_sensitivity_label', 0)}",
        f"- Rows missing label: {summary_b.get('rows_missing_sensitivity_label', 0)} (**must be 0**)",
        f"- Single-archive basis check: {summary_b.get('single_archive_basis_check', 'N/A')}",
        f"- **60d qualified snapshots:** {summary_b.get('acceptance_60d_qualified_snapshots', 0)} (threshold {THRESHOLD_B_60D}): {b60d_status} threshold",
        "",
        "---",
        "",
        "## Validation Checks",
        "",
    ]

    for k, v in validation.items():
        status = v.get("status", "UNKNOWN")
        icon = _status_icon(status)
        lines.append(f"### {k} [{icon}]")
        lines.append("")
        for field, val in v.items():
            if field == "status":
                continue
            if isinstance(val, list) and len(val) > 20:
                lines.append(f"- **{field}:** {len(val)} entries (truncated in report)")
            elif isinstance(val, list) and val:
                lines.append(f"- **{field}:**")
                for item in val[:20]:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"- **{field}:** {val}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Governance",
        "",
        "- **Production model freeze:** ACTIVE — no ranker/selector/sizing/final_score/gate/snapshot changes in this script",
        "- **PR #382:** Quarantined — no code copied from that branch",
        "- **Method B label:** All rows carry `SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE`",
        "- **ATXS exclusion:** Applied after 2026-01-23",
        "- **No 60d conclusion from Method A:** Enforced",
        "- **No live data fetch:** This script has no network calls",
        "",
        "*Generated by scripts/research/pit_gap_forward_returns.py*",
    ]

    path.write_text("\n".join(lines))
    log.info("Validation report written: %s", path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from datetime import datetime

    run_date = datetime.now().strftime("%Y-%m-%d")

    log.info("=== PIT Gap Forward Return Assembly ===")
    log.info("Run date: %s | Gap: %s to %s", run_date, GAP_START, GAP_END)
    log.info("Method A horizons: %s (NO 60d)", HORIZONS_A)
    log.info("Method B horizons: %s (SENSITIVITY ONLY)", HORIZONS_B)

    gap_dates = get_gap_snapshots()
    log.info("Gap snapshots found: %d", len(gap_dates))
    if not gap_dates:
        log.error("No gap snapshots found — check data/snapshots/")
        sys.exit(1)

    audit_dir().mkdir(parents=True, exist_ok=True)

    rows_a, summary_a = run_method_a(gap_dates, run_date)
    rows_b, summary_b = run_method_b(gap_dates, run_date)

    validation = run_validation_checks(rows_a, rows_b, gap_dates, summary_a, summary_b)

    path_a = write_method_a_csv(rows_a, run_date) if rows_a else None
    path_b = write_method_b_csv(rows_b, run_date) if rows_b else None
    write_validation_report(validation, summary_a, summary_b, run_date, path_a, path_b)

    log.info("=== Assembly complete ===")
    log.info("Outputs are QUARANTINED research artifacts in artifacts/audit/")
    log.info("Method B outputs are SENSITIVITY_ONLY_NOT_PRIMARY_EVIDENCE")
    log.info("No production files were modified.")


if __name__ == "__main__":
    main()
