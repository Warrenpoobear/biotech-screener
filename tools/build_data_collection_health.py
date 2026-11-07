#!/usr/bin/env python3
"""
Spec 026: Data Collection Health Orchestrator

Post-screen QA layer that reads existing snapshot artifacts and produces
a unified collection-health summary (JSON + markdown).

Usage (standalone):
    python tools/build_data_collection_health.py \
        --snapshot-dir data/snapshots/2026-03-17

Programmatic (from run_screen.py):
    from tools.build_data_collection_health import run_from_screen
    run_from_screen(snapshot_dir, data_dir, as_of_date)
"""

import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.types import is_retired_status  # noqa: E402  (needs PROJECT_ROOT on sys.path)

logger = logging.getLogger("data_collection_health")

SCHEMA_VERSION = "data_collection_health.v1"

DEFAULT_THRESHOLDS_PATH = PROJECT_ROOT / "production_data" / "data_collection_health_thresholds.json"

DEFAULT_THRESHOLDS = {
    "market_data_min_coverage_pct": 0.95,
    "ctgov_min_trial_count": 15000,
    "ctgov_min_tickers_covered": 250,
    "ctgov_max_malformed_count": 0,
    "ctgov_max_pcd_after_cd_count": 0,
    "ctgov_diff_events_warn_if_zero": True,
    "sec8k_warn_if_zero_without_skip": True,
    "options_top60_min_chain_coverage_pct": 0.60,
    "options_universe_min_coverage_pct": 0.50,
    "inputs_manifest_required_missing_max": 0,
    "catalyst_min_total_events": 500,
    "catalyst_min_tickers_with_events": 200,
}


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.debug("Failed to load %s: %s", path, e)
        return None


def load_thresholds(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or DEFAULT_THRESHOLDS_PATH
    loaded = _load_json(p)
    if loaded:
        merged = dict(DEFAULT_THRESHOLDS)
        merged.update(loaded)
        return merged
    return dict(DEFAULT_THRESHOLDS)


# ---------------------------------------------------------------------------
# Source health checks
# ---------------------------------------------------------------------------


def _check_cache_health(snap_dir: Path) -> Dict[str, Any]:
    ch = _load_json(snap_dir / "cache_health.json")
    if ch is None:
        return {"status": "WARN", "reason": "cache_health.json missing", "present": False}
    return {
        "status": (
            "PASS" if ch.get("overall_status") == "ok" else ("FAIL" if ch.get("overall_status") == "bad" else "WARN")
        ),
        "present": True,
        "overall_status": ch.get("overall_status"),
        "sec8k_status": ch.get("sec8k", {}).get("status"),
        "sec8k_count": ch.get("sec8k", {}).get("count"),
        "sec8k_reason": ch.get("sec8k", {}).get("reason", ""),
        "ctgov_status": ch.get("ctgov", {}).get("status"),
        "ctgov_count": ch.get("ctgov", {}).get("count"),
        "degraded_run": ch.get("degraded_run", False),
    }


def _check_catalyst_source_mix(snap_dir: Path, thresholds: Dict) -> Dict[str, Any]:
    sm = _load_json(snap_dir / "catalyst_source_mix.json")
    if sm is None:
        return {"status": "WARN", "reason": "catalyst_source_mix.json missing", "present": False}

    total_events = sm.get("total_events", 0)
    tickers_with_events = sm.get("unique_tickers_with_events", 0)
    by_source = sm.get("by_source", {})
    pre_dedup = sm.get("pre_dedup_by_source", {})

    flags: List[str] = []
    status = "PASS"

    # Diff-based events check
    ctgov_diff = pre_dedup.get("CTGOV", 0)
    ctgov_calendar = pre_dedup.get("CTGOV_CALENDAR", 0)
    if ctgov_diff == 0 and thresholds.get("ctgov_diff_events_warn_if_zero"):
        if ctgov_calendar > 0 and total_events >= thresholds.get("catalyst_min_total_events", 0):
            # Calendar coverage is healthy — diff gap is expected on unchanged days
            flags.append(
                "diff_based_catalyst_events=0 " "(no CTGov deltas vs prior snapshot; calendar coverage still present)"
            )
            if status != "FAIL":
                status = "WARN"
        else:
            # Both diff AND calendar are absent — real blind spot
            flags.append("diff_based_catalyst_events=0 (trial_records may be stale)")
            status = "FAIL"

    # SEC 8-K check
    sec8k_count = pre_dedup.get("SEC_8K_FILING", 0)
    if sec8k_count == 0 and thresholds.get("sec8k_warn_if_zero_without_skip"):
        flags.append("SEC_8K_FILING=0 (cache missing or empty)")
        if status != "FAIL":
            status = "WARN"

    # Total event floor
    if total_events < thresholds.get("catalyst_min_total_events", 0):
        flags.append(f"total_events={total_events} below floor {thresholds['catalyst_min_total_events']}")
        status = "FAIL"

    # Ticker coverage floor
    if tickers_with_events < thresholds.get("catalyst_min_tickers_with_events", 0):
        flags.append(
            f"tickers_with_events={tickers_with_events} below floor {thresholds['catalyst_min_tickers_with_events']}"
        )
        if status != "FAIL":
            status = "WARN"

    return {
        "status": status,
        "present": True,
        "total_events": total_events,
        "tickers_with_events": tickers_with_events,
        "by_source": by_source,
        "pre_dedup_by_source": pre_dedup,
        "ctgov_diff_events": ctgov_diff,
        "sec8k_events": sec8k_count,
        "flags": flags,
    }


def _check_coverage_quality(snap_dir: Path, thresholds: Dict, as_of_date: str = "") -> Dict[str, Any]:
    cq = _load_json(snap_dir / "coverage_quality.json")
    if cq is None:
        return {"status": "WARN", "reason": "coverage_quality.json missing", "present": False}

    flags: List[str] = []
    status = "PASS"

    cat = cq.get("catalyst_coverage", {})
    comp = cq.get("component_coverage", {})
    opts_fresh = cq.get("options_data_freshness", {})

    options_pct = (comp.get("options_pct", 0) or 0) / 100.0

    if options_pct < thresholds.get("options_universe_min_coverage_pct", 0):
        flags.append(
            f"options_coverage={options_pct:.1%} below floor {thresholds['options_universe_min_coverage_pct']:.0%}"
        )
        if status != "FAIL":
            status = "WARN"

    # Options data staleness: check if freshness data is from the expected date
    opts_collect_date = opts_fresh.get("collect_date", "")
    if opts_collect_date and as_of_date and opts_collect_date < as_of_date:
        flags.append(f"options_data_stale: collected={opts_collect_date}, expected={as_of_date}")
        if status != "FAIL":
            status = "WARN"

    return {
        "status": status,
        "present": True,
        "catalyst_specific_days_pct": cat.get("specific_days_pct"),
        "catalyst_family_pct": cq.get("catalyst_family_coverage", {}).get("coverage_pct"),
        "options_coverage_pct": comp.get("options_pct"),
        "sponsor_coverage_pct": comp.get("sponsor_pct"),
        "drawdown_coverage_pct": comp.get("drawdown_pct"),
        "options_all_fresh": opts_fresh.get("all_fresh"),
        "flags": flags,
    }


def _check_market_data(snap_dir: Path, data_dir: Path, as_of_date: str, thresholds: Dict) -> Dict[str, Any]:
    flags: List[str] = []
    status = "PASS"

    # Check price_history freshness
    # On weekends/holidays the exact as_of_date has no data, so we fall
    # back to the latest available trading day for coverage measurement.
    price_path = data_dir / "price_history.csv"
    latest_date = None
    ticker_count = 0
    used_fallback = False
    if price_path.exists():
        exact_tickers: set = set()
        fallback_tickers: set = set()
        with open(price_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                d = row.get("date", "")
                t = row.get("ticker", "")
                if d <= as_of_date:
                    if d == as_of_date:
                        exact_tickers.add(t)
                    if latest_date is None or d > latest_date:
                        latest_date = d
                        fallback_tickers = {t}
                    elif d == latest_date:
                        fallback_tickers.add(t)
        # Prefer exact date; fall back to latest trading day
        if exact_tickers:
            price_tickers = exact_tickers
        else:
            price_tickers = fallback_tickers
            used_fallback = bool(fallback_tickers)
    else:
        price_tickers = set()
        flags.append("price_history.csv missing")
        status = "FAIL"

    # Load universe tickers for coverage denominator. Retired statuses and
    # synthetic tickers (_XBI_BENCHMARK_) are excluded, matching the
    # production gate run_daily_production.check_market_data_coverage.
    universe_path = data_dir / "universe.json"
    universe_tickers: set = set()
    if universe_path.exists():
        with open(universe_path) as f:
            uni = json.load(f)
        entries = uni if isinstance(uni, list) else uni.get("tickers", [])
        for entry in entries:
            if isinstance(entry, dict):
                if is_retired_status(entry.get("status")):
                    continue
                t = entry.get("ticker", "")
            elif isinstance(entry, str):
                t = entry
            else:
                continue
            if t and not t.startswith("_"):
                universe_tickers.add(t)

    # Coverage = universe tickers that have price data (intersection),
    # so it never exceeds 100%.
    universe_count = len(universe_tickers)
    ticker_count = len(price_tickers & universe_tickers) if universe_tickers else len(price_tickers)
    coverage_pct = ticker_count / universe_count if universe_count > 0 else 0
    if coverage_pct < thresholds.get("market_data_min_coverage_pct", 0):
        flags.append(f"price_coverage={coverage_pct:.1%} below floor {thresholds['market_data_min_coverage_pct']:.0%}")
        status = "FAIL"

    # Label weekend/holiday fallback transparently
    measured_date = as_of_date
    if used_fallback and latest_date:
        measured_date = latest_date
        flags.append(f"price_coverage measured on {latest_date} (as_of_date {as_of_date} is non-trading)")

    return {
        "status": status,
        "latest_price_date": latest_date,
        "measured_price_date": measured_date,
        "as_of_date_ticker_count": ticker_count,
        "universe_ticker_count": universe_count,
        "price_coverage_pct": round(coverage_pct * 100, 1),
        "flags": flags,
    }


def _check_options(snap_dir: Path, thresholds: Dict) -> Dict[str, Any]:
    od = _load_json(snap_dir / "options_diagnostics_summary.json")
    if od is None:
        return {"status": "WARN", "reason": "options_diagnostics_summary.json missing", "present": False}

    flags: List[str] = []
    status = "PASS"

    cov = od.get("coverage", {})
    n_universe = cov.get("n_universe", 0)
    n_with_data = cov.get("n_with_options_data", 0)
    coverage_pct = cov.get("coverage_pct", 0) / 100.0

    # Chain coverage for top-60
    chains_dir = snap_dir / "chains"
    top60_with_chain = 0
    if chains_dir.exists():
        chain_tickers = {p.stem for p in chains_dir.glob("*.json")}
        # Read rankings to find top 60
        rankings_path = snap_dir / "rankings.csv"
        if rankings_path.exists():
            with open(rankings_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                top60 = []
                for r in reader:
                    arank = r.get("actionable_rank", "").strip()
                    if arank:
                        try:
                            if int(arank) <= 60:
                                top60.append(r["ticker"])
                        except ValueError:
                            pass
            top60_with_chain = sum(1 for t in top60 if t in chain_tickers)
            top60_chain_pct = top60_with_chain / len(top60) if top60 else 0
            if top60_chain_pct < thresholds.get("options_top60_min_chain_coverage_pct", 0):
                flags.append(
                    f"top60_chain_coverage={top60_chain_pct:.1%} below floor "
                    f"{thresholds['options_top60_min_chain_coverage_pct']:.0%}"
                )
                if status != "FAIL":
                    status = "WARN"

    # Long-call report coverage
    lc = _load_json(snap_dir / "long_call_candidates.json")
    lc_tradeable = lc.get("n_tradeable", 0) if lc else None
    lc_no_trade = lc.get("n_no_trade", 0) if lc else None

    return {
        "status": status,
        "present": True,
        "universe_count": n_universe,
        "with_options_data": n_with_data,
        "options_coverage_pct": round(coverage_pct * 100, 1),
        "top60_with_chain": top60_with_chain,
        "long_call_tradeable": lc_tradeable,
        "long_call_no_trade": lc_no_trade,
        "flags": flags,
    }


def _check_ctgov(snap_dir: Path, data_dir: Path, as_of_date: str, thresholds: Dict) -> Dict[str, Any]:
    flags: List[str] = []
    status = "PASS"

    # Check CTGov PIT cache
    cache_path = PROJECT_ROOT / "cache" / "ctgov" / f"trial_records_{as_of_date}.json"
    cache_present = cache_path.exists()
    trial_count = 0
    tickers_covered = 0

    if cache_present:
        trials = _load_json(cache_path)
        if trials and isinstance(trials, list):
            trial_count = len(trials)
            tickers_covered = len({t.get("ticker") for t in trials if t.get("ticker")})

    if not cache_present:
        flags.append(f"CTGov PIT cache missing for {as_of_date}")
        status = "WARN"
    elif trial_count < thresholds.get("ctgov_min_trial_count", 0):
        flags.append(f"trial_count={trial_count} below floor {thresholds['ctgov_min_trial_count']}")
        status = "FAIL"

    if tickers_covered < thresholds.get("ctgov_min_tickers_covered", 0) and cache_present:
        flags.append(f"tickers_covered={tickers_covered} below floor {thresholds['ctgov_min_tickers_covered']}")
        if status != "FAIL":
            status = "WARN"

    return {
        "status": status,
        "cache_present": cache_present,
        "trial_count": trial_count,
        "tickers_covered": tickers_covered,
        "flags": flags,
    }


def _check_sec(snap_dir: Path, as_of_date: str) -> Dict[str, Any]:
    # Check SEC 8-K cache file
    sec_dir = PROJECT_ROOT / "cache" / "sec" / "8k_catalysts"
    cache_present = False
    filing_count = 0
    if sec_dir.exists():
        matches = list(sec_dir.glob(f"8k_catalysts_{as_of_date}_*.json"))
        if matches:
            cache_present = True
            sec_data = _load_json(matches[0])
            if sec_data and isinstance(sec_data, list):
                filing_count = len(sec_data)

    return {
        "status": "PASS" if cache_present else "WARN",
        "cache_present": cache_present,
        "filing_count": filing_count,
    }


def _check_fda(as_of_date: str) -> Dict[str, Any]:
    fda_dir = PROJECT_ROOT / "cache" / "fda"
    adcom_present = False
    reg_present = False
    if fda_dir.exists():
        adcom_present = bool(list(fda_dir.glob(f"adcom_calendar_{as_of_date}*.json")))
        reg_present = bool(list(fda_dir.glob(f"fda_regulatory_{as_of_date}*.json")))

    # INFO (not WARN) when FDA caches are absent — they are opt-in features,
    # not a degraded state that requires action.
    return {
        "status": "PASS" if (adcom_present or reg_present) else "INFO",
        "adcom_cache_present": adcom_present,
        "regulatory_cache_present": reg_present,
    }


def _check_inputs_manifest(snap_dir: Path, thresholds: Dict) -> Dict[str, Any]:
    manifest = _load_json(snap_dir / "inputs_manifest.json")
    if manifest is None:
        # INFO (not WARN) — inputs manifest is opt-in via --inputs-manifest write.
        return {"status": "INFO", "present": False, "reason": "inputs_manifest.json not found"}

    deps = manifest.get("dependencies", [])
    required_missing = sum(1 for d in deps if d.get("required") and not d.get("present", True))

    status = "PASS"
    if required_missing > thresholds.get("inputs_manifest_required_missing_max", 0):
        status = "FAIL"

    return {
        "status": status,
        "present": True,
        "dependency_count": len(deps),
        "required_missing_count": required_missing,
        "warning_count": sum(1 for d in deps if d.get("warnings")),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def build_health(
    snapshot_dir: Path,
    data_dir: Path,
    as_of_date: str,
    thresholds: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build the unified data collection health summary."""
    if thresholds is None:
        thresholds = load_thresholds()

    sources = {
        "cache_health": _check_cache_health(snapshot_dir),
        "catalyst_source_mix": _check_catalyst_source_mix(snapshot_dir, thresholds),
        "coverage_quality": _check_coverage_quality(snapshot_dir, thresholds, as_of_date),
        "market_data": _check_market_data(snapshot_dir, data_dir, as_of_date, thresholds),
        "ctgov": _check_ctgov(snapshot_dir, data_dir, as_of_date, thresholds),
        "sec": _check_sec(snapshot_dir, as_of_date),
        "fda": _check_fda(as_of_date),
        "options": _check_options(snapshot_dir, thresholds),
        "inputs_manifest": _check_inputs_manifest(snapshot_dir, thresholds),
    }

    # Aggregate flags
    all_flags: List[str] = []
    for src_name, src_result in sources.items():
        for flag in src_result.get("flags", []):
            all_flags.append(f"[{src_name}] {flag}")

    # Derive overall status (INFO does not escalate — it signals expected/opt-in state)
    statuses = [s.get("status", "PASS") for s in sources.values()]
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "WARN" in statuses:
        overall = "WARN"
    else:
        overall = "PASS"

    return {
        "schema_version": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": overall,
        "sources": sources,
        "thresholds": thresholds,
        "flags": all_flags,
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_json_report(health: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(health, f, indent=2, default=str)
    logger.info(f"Wrote {path}")


def write_markdown_report(health: Dict, path: Path) -> None:
    lines = [
        f"# Data Collection Health — {health['as_of_date']}",
        "",
        f"**Overall: {health['status']}**",
        f"**Schema**: `{health['schema_version']}`",
        "",
    ]

    # Flags summary
    if health["flags"]:
        lines.append("## Flags")
        lines.append("")
        for flag in health["flags"]:
            lines.append(f"- {flag}")
        lines.append("")

    # Source coverage table
    lines.append("## Source Coverage")
    lines.append("")
    lines.append("| Source | Status | Key Metrics |")
    lines.append("|--------|--------|-------------|")

    src = health["sources"]

    # Cache health
    ch = src.get("cache_health", {})
    lines.append(
        f"| Cache Health | {ch.get('status', '?')} | sec8k={ch.get('sec8k_status', '?')}, ctgov={ch.get('ctgov_status', '?')}, degraded={ch.get('degraded_run', '?')} |"
    )

    # Catalyst source mix
    csm = src.get("catalyst_source_mix", {})
    lines.append(
        f"| Catalyst Mix | {csm.get('status', '?')} | events={csm.get('total_events', '?')}, tickers={csm.get('tickers_with_events', '?')}, diff={csm.get('ctgov_diff_events', '?')}, sec8k={csm.get('sec8k_events', '?')} |"
    )

    # Market data
    md = src.get("market_data", {})
    md_detail = f"latest={md.get('latest_price_date', '?')}, coverage={md.get('price_coverage_pct', '?')}%"
    if md.get("measured_price_date") and md["measured_price_date"] != health.get("as_of_date"):
        md_detail += f" (measured on {md['measured_price_date']})"
    lines.append(f"| Market Data | {md.get('status', '?')} | {md_detail} |")

    # CTGov
    ct = src.get("ctgov", {})
    lines.append(
        f"| CTGov | {ct.get('status', '?')} | cache={'yes' if ct.get('cache_present') else 'no'}, trials={ct.get('trial_count', '?')}, tickers={ct.get('tickers_covered', '?')} |"
    )

    # SEC
    sc = src.get("sec", {})
    lines.append(
        f"| SEC | {sc.get('status', '?')} | cache={'yes' if sc.get('cache_present') else 'no'}, filings={sc.get('filing_count', '?')} |"
    )

    # FDA
    fd = src.get("fda", {})
    lines.append(
        f"| FDA | {fd.get('status', '?')} | adcom={'yes' if fd.get('adcom_cache_present') else 'no'}, regulatory={'yes' if fd.get('regulatory_cache_present') else 'no'} |"
    )

    # Coverage quality
    cq = src.get("coverage_quality", {})
    lines.append(
        f"| Coverage Quality | {cq.get('status', '?')} | catalyst={cq.get('catalyst_specific_days_pct', '?')}%, options={cq.get('options_coverage_pct', '?')}%, sponsor={cq.get('sponsor_coverage_pct', '?')}% |"
    )

    # Options
    op = src.get("options", {})
    lines.append(
        f"| Options | {op.get('status', '?')} | universe={op.get('options_coverage_pct', '?')}%, top60_chains={op.get('top60_with_chain', '?')}, LC_tradeable={op.get('long_call_tradeable', '?')} |"
    )

    # Inputs manifest
    im = src.get("inputs_manifest", {})
    lines.append(
        f"| Inputs Manifest | {im.get('status', '?')} | present={'yes' if im.get('present') else 'no'}, deps={im.get('dependency_count', '?')}, required_missing={im.get('required_missing_count', '?')} |"
    )

    lines.append("")

    # Source detail: catalyst pre/post dedup
    if csm.get("present"):
        lines.append("## Catalyst Source Detail")
        lines.append("")
        lines.append("| Source | Pre-Dedup | Post-Dedup |")
        lines.append("|--------|-----------|------------|")
        pre = csm.get("pre_dedup_by_source", {})
        post = csm.get("by_source", {})
        all_sources = sorted(set(list(pre.keys()) + list(post.keys())))
        for s in all_sources:
            lines.append(f"| {s} | {pre.get(s, 0)} | {post.get(s, 0)} |")
        lines.append("")

    # Actions
    actions = []
    if csm.get("ctgov_diff_events", 0) == 0:
        actions.append("Refresh trial_records.json — diff-based catalyst detection is blind")
    if csm.get("sec8k_events", 0) == 0:
        actions.append("Warm SEC 8-K cache: `warm_caches.py --sources sec_8k`")
    if ch.get("degraded_run"):
        actions.append("Cache health degraded — check cache_refresh sidecar for details")
    if not ct.get("cache_present"):
        actions.append(f"Warm CTGov PIT cache: `warm_caches.py --sources ctgov --as-of-date {health['as_of_date']}`")
    if not im.get("present"):
        actions.append("Enable inputs manifest: run with `--inputs-manifest write`")

    if actions:
        lines.append("## Suggested Actions")
        lines.append("")
        for a in actions:
            lines.append(f"- {a}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Entry point for run_screen.py
# ---------------------------------------------------------------------------


def run_from_screen(
    snapshot_dir: Path,
    data_dir: Path,
    as_of_date: str,
) -> Optional[str]:
    """Auto-report entry point called from run_screen.py after snapshot.

    Returns overall status string, or None on failure.
    """
    try:
        health = build_health(snapshot_dir, data_dir, as_of_date)
        write_json_report(health, snapshot_dir / "data_collection_health.json")
        write_markdown_report(health, snapshot_dir / "data_collection_health.md")

        n_flags = len(health["flags"])
        logger.info(
            "[COLLECTION_HEALTH] %s (%d flags) — %s",
            health["status"],
            n_flags,
            snapshot_dir / "data_collection_health.md",
        )
        return health["status"]
    except Exception as exc:
        logger.warning("[COLLECTION_HEALTH] Report generation failed: %s — skipping", exc)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Spec 026: Data Collection Health Report")
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "production_data")
    parser.add_argument("--as-of-date", type=str, default=None)
    parser.add_argument("--thresholds", type=Path, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    as_of_date = args.as_of_date or args.snapshot_dir.name
    thresholds = load_thresholds(args.thresholds)

    health = build_health(args.snapshot_dir, args.data_dir, as_of_date, thresholds)
    write_json_report(health, args.snapshot_dir / "data_collection_health.json")
    write_markdown_report(health, args.snapshot_dir / "data_collection_health.md")

    print(f"\n{'='*60}")
    print(f"DATA COLLECTION HEALTH — {as_of_date}")
    print(f"{'='*60}")
    print(f"Status: {health['status']}")
    if health["flags"]:
        print(f"Flags ({len(health['flags'])}):")
        for f in health["flags"]:
            print(f"  - {f}")
    else:
        print("No flags.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
