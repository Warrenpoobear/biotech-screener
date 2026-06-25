#!/usr/bin/env python3
"""Data Auditor — integrity checks for biotech screener pipeline inputs.

Runnable standalone:
    python3 agents/data_auditor/run_audit.py --as-of-date 2026-04-02
    python3 agents/data_auditor/run_audit.py --daily-only
    python3 agents/data_auditor/run_audit.py --weekly-only

Exit codes: 0=PASS, 1=FAIL, 2=WARN
"""

import argparse
import csv
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data"
PROD_DIR = REPO_ROOT / "production_data"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
PIT_ARCHIVE_DIR = DATA_DIR / "pit_archives"
PIT_FIN_DIR = PROD_DIR / "pit_financials"
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "data_auditor"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path):
    """Load JSON, return None if missing."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _load_rankings(as_of_date_str):
    """Load rankings.csv for the given date, return list of dicts."""
    path = SNAPSHOT_DIR / as_of_date_str / "rankings.csv"
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _top_n_tickers(rankings, n=30):
    """Return top-N tickers by actionable_rank (ascending, 1 = best)."""
    if not rankings:
        return []
    ranked = []
    for row in rankings:
        try:
            r = int(row.get("actionable_rank", 9999))
            ranked.append((r, row["ticker"]))
        except (ValueError, KeyError):
            continue
    ranked.sort()
    return [t for _, t in ranked[:n]]


def _parse_date(s):
    """Parse YYYY-MM-DD string to date."""
    return datetime.strptime(s, "%Y-%m-%d").date()


def _status_merge(*statuses):
    """FAIL > WARN > ERROR > PASS."""
    priority = {"FAIL": 3, "WARN": 2, "ERROR": 1, "PASS": 0}
    best = "PASS"
    for s in statuses:
        if priority.get(s, 0) > priority.get(best, 0):
            best = s
    return best


# ---------------------------------------------------------------------------
# Daily checks
# ---------------------------------------------------------------------------


def check_archive_verification(as_of_date_str):
    """Check 1: PIT archive manifest exists for today."""
    result = {"status": "PASS", "detail": ""}
    manifest = PIT_ARCHIVE_DIR / as_of_date_str / "manifest.json"
    if manifest.exists():
        result["detail"] = f"Archive manifest found for {as_of_date_str}"
        return result

    # Check if archive dir exists but no manifest
    archive_dir = PIT_ARCHIVE_DIR / as_of_date_str
    if archive_dir.exists():
        result["status"] = "WARN"
        result["detail"] = f"Archive dir exists for {as_of_date_str} but manifest.json missing"
        return result

    # Check for consecutive missing days
    yesterday = (_parse_date(as_of_date_str) - timedelta(days=1)).isoformat()
    _ = PIT_ARCHIVE_DIR / yesterday / "manifest.json"
    yesterday_dir = PIT_ARCHIVE_DIR / yesterday

    if not archive_dir.exists() and not yesterday_dir.exists():
        result["status"] = "FAIL"
        result["detail"] = f"Archive missing for {as_of_date_str} AND {yesterday} (2+ consecutive)"
    else:
        result["status"] = "WARN"
        result["detail"] = f"Archive manifest missing for {as_of_date_str}"
    return result


def check_universe_ipo_consistency(as_of_date_str):
    """Check 2: survivorship violations and IPO coverage."""
    result = {"status": "PASS", "violations": 0, "missing_ipo": 0, "detail": ""}

    ipo_data = _load_json(PROD_DIR / "ipo_dates.json")
    rankings = _load_rankings(as_of_date_str)

    if ipo_data is None:
        result["status"] = "ERROR"
        result["detail"] = "Cannot load ipo_dates.json"
        return result
    if rankings is None:
        result["status"] = "ERROR"
        result["detail"] = f"Cannot load rankings.csv for {as_of_date_str}"
        return result

    ipo_tickers = ipo_data.get("tickers", {})
    as_of = _parse_date(as_of_date_str)

    # Check ranked tickers for survivorship violations
    violations = []
    for row in rankings:
        ticker = row.get("ticker", "")
        if not ticker:
            continue
        ipo_entry = ipo_tickers.get(ticker, {})
        first_price = ipo_entry.get("first_price_date", "")
        if first_price:
            try:
                if _parse_date(first_price) > as_of:
                    violations.append(ticker)
            except ValueError:
                pass

    # Check universe tickers missing from ipo_dates
    # Skip pseudo-tickers (e.g. _XBI_BENCHMARK_) that are not real equities
    universe = _load_json(PROD_DIR / "universe.json")
    missing_ipo = []
    if universe:
        for entry in universe:
            ticker = entry.get("ticker", "") if isinstance(entry, dict) else str(entry)
            if ticker and ticker not in ipo_tickers and not ticker.startswith("_"):
                missing_ipo.append(ticker)

    result["violations"] = len(violations)
    result["missing_ipo"] = len(missing_ipo)

    parts = []
    if violations:
        result["status"] = "FAIL"
        parts.append(f"{len(violations)} survivorship violation(s): {violations[:10]}")
    if missing_ipo:
        result["status"] = _status_merge(result["status"], "WARN")
        parts.append(f"{len(missing_ipo)} universe ticker(s) missing from ipo_dates.json: {missing_ipo[:10]}")
    if not parts:
        parts.append("No survivorship violations; all universe tickers in ipo_dates.json")

    result["detail"] = "; ".join(parts)
    return result


def check_pit_financials_freshness(as_of_date_str):
    """Check 3: PIT financials exist and are fresh for top-30."""
    result = {"status": "PASS", "stale_tickers": [], "missing_tickers": [], "detail": ""}

    rankings = _load_rankings(as_of_date_str)
    if rankings is None:
        result["status"] = "ERROR"
        result["detail"] = f"Cannot load rankings.csv for {as_of_date_str}"
        return result

    top30 = _top_n_tickers(rankings, 30)
    as_of = _parse_date(as_of_date_str)
    stale_cutoff = as_of - timedelta(days=120)

    for ticker in top30:
        pit_path = PIT_FIN_DIR / f"{ticker}.json"
        if not pit_path.exists():
            result["missing_tickers"].append(ticker)
            continue

        pit_data = _load_json(pit_path)
        if pit_data is None:
            result["missing_tickers"].append(ticker)
            continue

        # Find the most recent filed date across all fact categories
        facts = pit_data.get("facts", {})
        latest_filed = None
        for _category, entries in facts.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                filed_str = entry.get("filed", "")
                if filed_str:
                    try:
                        filed_date = _parse_date(filed_str)
                        if latest_filed is None or filed_date > latest_filed:
                            latest_filed = filed_date
                    except ValueError:
                        pass

        if latest_filed is None:
            result["stale_tickers"].append(ticker)
        elif latest_filed < stale_cutoff:
            result["stale_tickers"].append(ticker)

    if result["missing_tickers"]:
        result["status"] = "FAIL"
    if result["stale_tickers"]:
        result["status"] = _status_merge(result["status"], "WARN")

    parts = []
    if result["missing_tickers"]:
        parts.append(f"{len(result['missing_tickers'])} top-30 missing PIT financials: {result['missing_tickers']}")
    if result["stale_tickers"]:
        parts.append(f"{len(result['stale_tickers'])} top-30 stale (>120d): {result['stale_tickers']}")
    if not parts:
        parts.append(f"All {len(top30)} top-30 names have fresh PIT financials")
    result["detail"] = "; ".join(parts)
    return result


def _latest_fact_entry(entries):
    """Pick most recent entry by (filed_date, end_date). Returns (end_date, val) or None."""
    latest = None
    for entry in entries or []:
        filed_str = entry.get("filed", "")
        end_str = entry.get("end", "")
        if filed_str and end_str:
            try:
                key = (_parse_date(filed_str), _parse_date(end_str))
                if latest is None or key > latest[0]:
                    latest = (key, _parse_date(end_str), entry.get("val"))
            except ValueError:
                pass
    if latest is None or latest[2] is None:
        return None
    return (latest[1], latest[2])


def _fact_entry_at(entries, end_date):
    """Find the entry with given end_date (prefer most recently filed). Returns val or None."""
    best = None
    for entry in entries or []:
        filed_str = entry.get("filed", "")
        end_str = entry.get("end", "")
        if filed_str and end_str:
            try:
                if _parse_date(end_str) == end_date:
                    filed_date = _parse_date(filed_str)
                    if best is None or filed_date > best[0]:
                        best = (filed_date, entry.get("val"))
            except ValueError:
                pass
    return best[1] if best and best[1] is not None else None


def check_financial_consistency(as_of_date_str):
    """Check 4: Compare cash in financial_records vs PIT financials for top-30.

    - Prefers CashAndSecurities over Cash (aligns with PIT: cash + short_term_investments)
    - Skips foreign issuers where Cash_currency != USD (audit has no FX table)
    - FAIL when static fallback (production_data/financial_records.json) is used for
      any top-30 ticker: financial_score is the dominant negative ranker signal
      (NW-t=-3.41) so stale cash/runway figures directly affect ranking quality.
      Use pit_archives/<date>/financial_records.json when available (normal path).
    """
    result = {
        "status": "PASS",
        "divergences": [],
        "skipped_non_usd": [],
        "fallback_tickers_top30": [],
        "used_static_fallback": False,
        "detail": "",
    }

    rankings = _load_rankings(as_of_date_str)
    if rankings is None:
        result["status"] = "ERROR"
        result["detail"] = f"Cannot load rankings.csv for {as_of_date_str}"
        return result

    pit_archive_path = PIT_ARCHIVE_DIR / as_of_date_str / "financial_records.json"
    if pit_archive_path.exists():
        fin_records_path = pit_archive_path
    else:
        fin_records_path = PROD_DIR / "financial_records.json"
        result["used_static_fallback"] = True
    fin_records_raw = _load_json(fin_records_path)

    if fin_records_raw is None:
        result["status"] = "ERROR"
        result["detail"] = "Cannot load financial_records.json"
        return result

    # Build ticker -> {cash, cash_and_securities, currency} lookup
    def _extract(entry):
        cash = entry.get("Cash")
        cs = entry.get("CashAndSecurities")
        # Currency default is USD when unspecified (domestic issuers omit the tag)
        currency = entry.get("CashAndSecurities_currency") or entry.get("Cash_currency") or "USD"
        try:
            cash_f = float(cash) if cash is not None else None
        except (ValueError, TypeError):
            cash_f = None
        try:
            cs_f = float(cs) if cs is not None else None
        except (ValueError, TypeError):
            cs_f = None
        return {"cash": cash_f, "cash_and_securities": cs_f, "currency": currency}

    fr_records = {}
    if isinstance(fin_records_raw, list):
        for entry in fin_records_raw:
            ticker = entry.get("ticker", "")
            if ticker:
                fr_records[ticker] = _extract(entry)
    elif isinstance(fin_records_raw, dict):
        for ticker, entry in fin_records_raw.items():
            if isinstance(entry, dict):
                fr_records[ticker] = _extract(entry)

    top30 = _top_n_tickers(rankings, 30)

    for ticker in top30:
        rec = fr_records.get(ticker)
        if rec is None:
            continue

        if rec["currency"] != "USD":
            result["skipped_non_usd"].append({"ticker": ticker, "currency": rec["currency"]})
            continue

        # Prefer CashAndSecurities; on PIT side aggregate cash + short_term_investments
        # (same end period) so the comparison is semantically aligned.
        use_securities = rec["cash_and_securities"] is not None
        fr_val = rec["cash_and_securities"] if use_securities else rec["cash"]
        if fr_val is None or fr_val == 0:
            continue

        pit_data = _load_json(PIT_FIN_DIR / f"{ticker}.json")
        if pit_data is None:
            continue

        facts = pit_data.get("facts", {})
        cash_latest = _latest_fact_entry(facts.get("cash", []))
        if cash_latest is None:
            continue

        end_date, cash_val = cash_latest
        pit_val = float(cash_val)
        pit_label = "cash"

        if use_securities:
            sti_val = _fact_entry_at(facts.get("short_term_investments", []), end_date)
            if sti_val is not None:
                pit_val += float(sti_val)
                pit_label = "cash+short_term_investments"
            # If no matching STI period, compare cash-only to fr cash (not cash_and_securities)
            # to avoid flagging a semantic mismatch as a divergence.
            elif rec["cash"] is not None and rec["cash"] > 0:
                fr_val = rec["cash"]

        if pit_val == 0:
            continue

        pct_diff = abs(fr_val - pit_val) / max(abs(fr_val), abs(pit_val))
        # PIT vs current financial_records divergence is expected when
        # quarterly filings arrive at different times. Only flag extreme
        # cases (>50%) that may indicate data corruption rather than lag.
        if pct_diff > 0.50:
            result["divergences"].append(
                {
                    "ticker": ticker,
                    "financial_records_value": fr_val,
                    "pit_value": pit_val,
                    "pit_basis": pit_label,
                    "pct_diff": round(pct_diff * 100, 1),
                }
            )

    # When the static fallback was used, record which top-30 tickers were
    # affected so the report surfaces the exact ranking impact.
    if result["used_static_fallback"]:
        for ticker in top30:
            if fr_records.get(ticker) is not None:
                result["fallback_tickers_top30"].append(ticker)

    parts = []
    if result["fallback_tickers_top30"]:
        result["status"] = "FAIL"
        n = len(result["fallback_tickers_top30"])
        sample = result["fallback_tickers_top30"][:5]
        suffix = "..." if n > 5 else ""
        parts.append(
            f"STATIC FALLBACK active: {n} top-30 tickers using stale production_data/financial_records.json"
            f" ({sample}{suffix}) — financial_score rankings may reflect stale cash/runway"
        )
    if result["divergences"]:
        result["status"] = _status_merge(result["status"], "WARN")
        parts.append(
            f"{len(result['divergences'])} divergence(s) > 50%: " f"{[d['ticker'] for d in result['divergences']]}"
        )
    if result["skipped_non_usd"]:
        parts.append(
            f"{len(result['skipped_non_usd'])} skipped (non-USD): "
            f"{[s['ticker'] for s in result['skipped_non_usd']]}"
        )
    if not parts:
        parts.append("No cash divergences > 50% in top-30")
    result["detail"] = "; ".join(parts)
    return result


def check_price_data_gaps(as_of_date_str):
    """Check 5: Top-30 tickers have recent price data."""
    result = {"status": "PASS", "missing_tickers": [], "detail": ""}

    rankings = _load_rankings(as_of_date_str)
    if rankings is None:
        result["status"] = "ERROR"
        result["detail"] = f"Cannot load rankings.csv for {as_of_date_str}"
        return result

    price_path = PROD_DIR / "price_history.csv"
    if not price_path.exists():
        result["status"] = "ERROR"
        result["detail"] = "price_history.csv not found"
        return result

    top30 = _top_n_tickers(rankings, 30)
    as_of = _parse_date(as_of_date_str)

    # Find the most recent trading day (skip weekends)
    recent_day = as_of
    if recent_day.weekday() == 5:  # Saturday
        recent_day = recent_day - timedelta(days=1)
    elif recent_day.weekday() == 6:  # Sunday
        recent_day = recent_day - timedelta(days=2)

    # Build set of (ticker, date) pairs from price_history.csv
    # Only scan for the recent_day to avoid loading entire file into memory
    recent_str = recent_day.isoformat()
    tickers_with_price = set()

    with open(price_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("date", "") == recent_str:
                ticker = row.get("ticker", "")
                if ticker in top30:
                    tickers_with_price.add(ticker)

    missing = [t for t in top30 if t not in tickers_with_price]
    result["missing_tickers"] = missing

    if missing:
        result["status"] = "WARN"
        result["detail"] = f"{len(missing)} top-30 ticker(s) missing price for {recent_str}: {missing[:10]}"
    else:
        result["detail"] = f"All {len(top30)} top-30 tickers have price data for {recent_str}"
    return result


# ---------------------------------------------------------------------------
# Weekly checks
# ---------------------------------------------------------------------------


def check_pit_validation_sweep(as_of_date_str):
    """Check 6: Survivorship audit on past 7 snapshots."""
    result = {"status": "PASS", "violation_counts": {}, "detail": ""}

    ipo_data = _load_json(PROD_DIR / "ipo_dates.json")
    if ipo_data is None:
        result["status"] = "ERROR"
        result["detail"] = "Cannot load ipo_dates.json"
        return result

    ipo_tickers = ipo_data.get("tickers", {})
    as_of = _parse_date(as_of_date_str)
    total_violations = 0

    for days_back in range(7):
        check_date = as_of - timedelta(days=days_back)
        check_str = check_date.isoformat()
        rankings = _load_rankings(check_str)
        if rankings is None:
            continue

        violations = 0
        for row in rankings:
            ticker = row.get("ticker", "")
            ipo_entry = ipo_tickers.get(ticker, {})
            first_price = ipo_entry.get("first_price_date", "")
            if first_price:
                try:
                    if _parse_date(first_price) > check_date:
                        violations += 1
                except ValueError:
                    pass

        result["violation_counts"][check_str] = violations
        total_violations += violations

    if total_violations > 0:
        result["status"] = "WARN"
        result["detail"] = (
            f"{total_violations} total survivorship violation(s) across 7 snapshots: {result['violation_counts']}"
        )
    else:
        result["detail"] = "No survivorship violations in past 7 snapshots"
    return result


def check_edgar_coverage(as_of_date_str):
    """Check 7: PIT financials coverage vs universe."""
    result = {"status": "PASS", "universe_count": 0, "pit_count": 0, "coverage_pct": 0.0, "detail": ""}

    universe = _load_json(PROD_DIR / "universe.json")
    if universe is None:
        result["status"] = "ERROR"
        result["detail"] = "Cannot load universe.json"
        return result

    universe_tickers = set()
    for entry in universe:
        ticker = entry.get("ticker", "") if isinstance(entry, dict) else str(entry)
        if ticker:
            universe_tickers.add(ticker)

    # Count tickers with PIT financials
    pit_tickers = set()
    if PIT_FIN_DIR.exists():
        for f in PIT_FIN_DIR.iterdir():
            if f.suffix == ".json":
                pit_tickers.add(f.stem)

    covered = universe_tickers & pit_tickers
    result["universe_count"] = len(universe_tickers)
    result["pit_count"] = len(covered)
    result["coverage_pct"] = round(len(covered) / len(universe_tickers) * 100, 1) if universe_tickers else 0

    if result["coverage_pct"] < 95.0:
        result["status"] = "WARN"
        result["detail"] = (
            f"EDGAR coverage {result['coverage_pct']}% ({len(covered)}/{len(universe_tickers)}) — below 95% threshold"
        )
    else:
        result["detail"] = f"EDGAR coverage {result['coverage_pct']}% ({len(covered)}/{len(universe_tickers)})"
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Known optional engine modules in run_screen.py (HAS_* flags at import time).
# Add new engines here when they're introduced; removing an entry is intentional
# deprecation (not a regression).
_OPTIONAL_ENGINES = [
    ("HAS_ENHANCEMENTS", ["indication_mapper", "pos_engine", "regime_engine", "short_interest_engine"]),
    ("HAS_MACRO_COLLECTOR", ["macro_data_collector"]),
    ("HAS_ACCURACY_ENHANCEMENTS", ["accuracy_enhancements_adapter"]),
    ("HAS_DILUTION_RISK", ["dilution_risk_engine"]),
    ("HAS_TIMELINE_SLIPPAGE", ["timeline_slippage_engine"]),
    ("HAS_FDA_DESIGNATIONS", ["fda_designation_engine"]),
    ("HAS_PIPELINE_DIVERSITY", ["pipeline_diversity_engine"]),
    ("HAS_COMPETITIVE_INTENSITY", ["competitive_intensity_engine"]),
    ("HAS_PARTNERSHIP_ENGINE", ["partnership_engine"]),
    ("HAS_CASH_BURN_ENGINE", ["cash_burn_engine"]),
    ("HAS_PHASE_MOMENTUM_ENGINE", ["phase_momentum_engine"]),
    ("HAS_MORNINGSTAR", ["morningstar_signal_engine"]),
    ("HAS_RISK_GATES", ["risk_gates"]),
    ("HAS_LIQUIDITY_SCORING", ["liquidity_scoring_engine"]),
    ("HAS_TICKER_VALIDATION", ["ticker_validation"]),
]


def check_optional_engines(as_of_date_str):
    """Check: optional engine presence vs prior day.

    Probes each known optional engine module via importlib. Writes a
    optional_engines.json sidecar to data/snapshots/<date>/ for trending.
    FAILs when any engine that was importable yesterday is no longer importable
    today (true->false regression = silent feature disappearance).
    """
    import importlib

    result = {
        "status": "PASS",
        "engines": {},
        "regressions": [],
        "new_failures": [],
        "detail": "",
    }

    # Probe each engine group
    for flag, modules in _OPTIONAL_ENGINES:
        group_ok = False
        for mod in modules:
            try:
                importlib.import_module(mod)
                group_ok = True
                break
            except ImportError:
                pass
        result["engines"][flag] = group_ok

    # Write today's sidecar
    today_sidecar = SNAPSHOT_DIR / as_of_date_str / "optional_engines.json"
    try:
        today_sidecar.parent.mkdir(parents=True, exist_ok=True)
        with open(today_sidecar, "w", encoding="utf-8") as f:
            json.dump(
                {"as_of_date": as_of_date_str, "engines": result["engines"]},
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")
    except OSError as e:
        result["status"] = _status_merge(result["status"], "WARN")
        result["detail"] = f"Could not write optional_engines.json sidecar: {e}"
        return result

    # Compare to prior day
    prior_date = (_parse_date(as_of_date_str) - timedelta(days=1)).isoformat()
    prior_sidecar = SNAPSHOT_DIR / prior_date / "optional_engines.json"
    if prior_sidecar.exists():
        try:
            prior = json.loads(prior_sidecar.read_text(encoding="utf-8"))
            prior_engines = prior.get("engines", {})
            for flag, ok_today in result["engines"].items():
                ok_prior = prior_engines.get(flag)
                if ok_prior is True and ok_today is False:
                    result["regressions"].append(flag)
        except (json.JSONDecodeError, OSError):
            pass

    if result["regressions"]:
        result["status"] = "FAIL"
        result["detail"] = (
            f"Optional engine regression(s): {result['regressions']} " f"— was importable yesterday, missing today"
        )
    else:
        ok_count = sum(1 for v in result["engines"].values() if v)
        total = len(result["engines"])
        result["detail"] = f"{ok_count}/{total} optional engine groups importable" + (
            f"; {len(result['new_failures'])} new failures" if result["new_failures"] else ""
        )
    return result


def run_audit(as_of_date_str, daily_only=False, weekly_only=False):
    """Run all applicable checks and produce report."""
    checks = {}
    run_daily = not weekly_only
    run_weekly = not daily_only

    if run_daily:
        print(f"[data_auditor] Running daily checks for {as_of_date_str}...")
        checks["archive_verification"] = check_archive_verification(as_of_date_str)
        checks["universe_ipo_consistency"] = check_universe_ipo_consistency(as_of_date_str)
        checks["pit_financials_freshness"] = check_pit_financials_freshness(as_of_date_str)
        checks["financial_consistency"] = check_financial_consistency(as_of_date_str)
        checks["price_data_gaps"] = check_price_data_gaps(as_of_date_str)
        checks["optional_engines"] = check_optional_engines(as_of_date_str)

    if run_weekly:
        print(f"[data_auditor] Running weekly checks for {as_of_date_str}...")
        checks["pit_validation_sweep"] = check_pit_validation_sweep(as_of_date_str)
        checks["edgar_coverage"] = check_edgar_coverage(as_of_date_str)

    # Compute overall verdict
    all_statuses = [c["status"] for c in checks.values()]
    verdict = _status_merge(*all_statuses) if all_statuses else "PASS"

    # Count by status
    pass_count = sum(1 for s in all_statuses if s == "PASS")
    warn_count = sum(1 for s in all_statuses if s == "WARN")
    fail_count = sum(1 for s in all_statuses if s == "FAIL")
    error_count = sum(1 for s in all_statuses if s == "ERROR")

    parts = []
    if fail_count:
        parts.append(f"{fail_count} FAIL")
    if warn_count:
        parts.append(f"{warn_count} WARN")
    if error_count:
        parts.append(f"{error_count} ERROR")
    if pass_count:
        parts.append(f"{pass_count} PASS")
    summary = f"{len(all_statuses)} checks: {', '.join(parts)}"

    report = {
        "schema": "data_integrity_report.v1",
        "as_of_date": as_of_date_str,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "verdict": verdict,
        "checks": checks,
        "summary": summary,
    }

    # Write artifact
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACT_DIR / f"integrity_report_{as_of_date_str}.json"
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # Print human-readable summary
    print()
    print(f"{'=' * 60}")
    print(f"  Data Integrity Report — {as_of_date_str}")
    print(f"  Verdict: {verdict}")
    print(f"  {summary}")
    print(f"{'=' * 60}")
    for name, check in checks.items():
        status = check["status"]
        marker = {"PASS": "OK", "WARN": "!!", "FAIL": "XX", "ERROR": "??"}.get(status, "??")
        print(f"  [{marker}] {name}: {check.get('detail', '')}")
    print(f"{'=' * 60}")
    print(f"  Report: {artifact_path}")
    print()

    return report


def main():
    parser = argparse.ArgumentParser(description="Data Auditor — pipeline input integrity checks")
    parser.add_argument(
        "--as-of-date", default=date.today().isoformat(), help="Date to audit (YYYY-MM-DD, default: today)"
    )
    parser.add_argument("--daily-only", action="store_true", help="Run daily checks only")
    parser.add_argument("--weekly-only", action="store_true", help="Run weekly checks only")
    args = parser.parse_args()
    started = time.perf_counter()

    if args.daily_only and args.weekly_only:
        print("ERROR: --daily-only and --weekly-only are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    report = run_audit(args.as_of_date, daily_only=args.daily_only, weekly_only=args.weekly_only)

    verdict = report["verdict"]
    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        exec_id = log_agent_run(
            "data_auditor",
            f"Integrity audit for {args.as_of_date}",
            inputs={"as_of_date": args.as_of_date},
            outputs={"verdict": verdict, "summary": report.get("summary")},
            success=verdict not in ("FAIL", "ERROR"),
            error=verdict if verdict in ("FAIL", "ERROR") else None,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if exec_id:
            attach_outcome_verdict(
                exec_id,
                was_correct=verdict == "PASS",
                evidence=f"verdict={verdict}",
            )
    except Exception:
        pass

    if verdict == "FAIL":
        sys.exit(1)
    elif verdict in ("WARN", "ERROR"):
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
