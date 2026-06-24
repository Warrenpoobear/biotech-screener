#!/usr/bin/env python3
"""Post-run operations digest — single-screen actionable summary.

Reads today's snapshot + shadow portfolio artifacts, diffs against the
prior snapshot, and produces a concise digest highlighting only items
that need human attention.

Designed as the core skill for an ops agent: run after production,
summarize new issues, refuse to modify active rulesets.

Output:
    artifacts/ops_digest/{date}_digest.json   — structured digest
    artifacts/ops_digest/{date}_digest.md     — human-readable summary

Usage:
    python tools/build_ops_digest.py --as-of-date 2026-03-23
    python tools/build_ops_digest.py  # auto-detect latest snapshot

Programmatic:
    from tools.build_ops_digest import build_ops_digest
    digest = build_ops_digest("2026-03-23")
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("ops_digest")

SCHEMA_VERSION = "ops_digest.v1"
SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshots"
SHADOW_DIR = REPO_ROOT / "artifacts" / "live_shadow"
READINESS_DIR = REPO_ROOT / "artifacts" / "readiness"
OUT_DIR = REPO_ROOT / "artifacts" / "ops_digest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _find_prior_snapshot(snapshots_dir: Path, current_date: str) -> Optional[str]:
    """Find the most recent snapshot before current_date."""
    candidates = []
    for d in snapshots_dir.iterdir():
        name = d.name
        if d.is_dir() and len(name) == 10 and name < current_date and "__" not in name:
            try:
                # Validate date format
                datetime.strptime(name, "%Y-%m-%d")
                candidates.append(name)
            except ValueError:
                continue
    return max(candidates) if candidates else None


def _find_latest_snapshot(snapshots_dir: Path) -> Optional[str]:
    """Find the most recent snapshot."""
    candidates = []
    for d in snapshots_dir.iterdir():
        name = d.name
        if d.is_dir() and len(name) == 10 and "__" not in name:
            try:
                datetime.strptime(name, "%Y-%m-%d")
                candidates.append(name)
            except ValueError:
                continue
    return max(candidates) if candidates else None


def _severity(status: str) -> int:
    return {"FAIL": 3, "WARN": 2, "INFO": 1, "PASS": 0}.get(status, 0)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_pipeline_status(snap_dir: Path, as_of_date: str) -> Dict[str, Any]:
    """Check run_manifest.json for pipeline completion status.

    Without this, a crashed pipeline could leave stale health files from the
    prior day, causing the digest to report CLEAR when production actually failed.
    """
    # Check snapshot-dir manifest first, then fallback output/ manifest
    manifest = _load_json(snap_dir / "run_manifest.json")
    if not manifest:
        manifest = _load_json(REPO_ROOT / "output" / "run_manifest.json")

    if not manifest:
        return {
            "available": False,
            "detail": "run_manifest.json not found — pipeline may not have run",
        }

    manifest_date = manifest.get("as_of_date", "")
    if manifest_date != as_of_date:
        return {
            "available": True,
            "stale": True,
            "overall_status": "UNKNOWN",
            "manifest_date": manifest_date,
            "detail": f"Manifest is for {manifest_date}, not {as_of_date} — possible stale data",
        }

    overall = manifest.get("overall_status", "UNKNOWN")
    gate_results = manifest.get("gate_results", {})
    n_pass = sum(1 for v in gate_results.values() if v == "PASS")
    n_fail = sum(1 for v in gate_results.values() if v == "FAIL")
    n_warn = sum(1 for v in gate_results.values() if v == "WARN")

    return {
        "available": True,
        "stale": False,
        "overall_status": overall,
        "gates_pass": n_pass,
        "gates_fail": n_fail,
        "gates_warn": n_warn,
        "generated_at": manifest.get("generated_at", ""),
    }


def _build_health_section(snap_dir: Path) -> Dict[str, Any]:
    """Phase-2 health + collection health + exposure checks."""
    phase2 = _load_json(snap_dir / "phase2_health.json") or {}
    collection = _load_json(snap_dir / "data_collection_health.json") or {}
    exposure = _load_json(snap_dir / "health_exposure_metrics.json") or {}

    # Collect all non-PASS items
    alerts: List[Dict[str, str]] = []

    # Phase2 health
    p2_status = phase2.get("status", "?")
    if p2_status != "OK":
        for reason in phase2.get("reasons", []):
            alerts.append({"source": "phase2_health", "level": "WARN", "detail": reason})

    # Collection health flags
    for flag in collection.get("flags", []):
        level = "WARN"
        if "FAIL" in flag.upper():
            level = "FAIL"
        elif "measured on" in flag.lower():
            level = "INFO"
        alerts.append({"source": "collection_health", "level": level, "detail": flag})

    # Exposure checks
    for name, check in (exposure.get("checks") or {}).items():
        if check.get("status", "PASS") != "PASS":
            alerts.append(
                {
                    "source": "exposure",
                    "level": check["status"],
                    "detail": f"{name}: {check.get('value', '?')} (threshold: {check.get('warn_threshold', '?')})",
                }
            )

    return {
        "phase2_status": p2_status,
        "collection_status": collection.get("status", "?"),
        "exposure_alerts": len([a for a in alerts if a["level"] in ("WARN", "FAIL")]),
        "alerts": sorted(alerts, key=lambda a: -_severity(a["level"])),
    }


def _build_coverage_section(snap_dir: Path) -> Dict[str, Any]:
    """Coverage quality snapshot."""
    cq = _load_json(snap_dir / "coverage_quality.json") or {}
    elig = _load_json(snap_dir / "eligibility_summary.json") or {}

    cat_cov = cq.get("catalyst_coverage", {})
    comp_cov = cq.get("component_coverage", {})
    reg_sec = cq.get("regulatory_secondary", {})

    return {
        "n_total": elig.get("n_total", 0),
        "n_eligible": elig.get("n_eligible", 0),
        "n_ineligible": elig.get("n_ineligible", 0),
        "catalyst_pct": cat_cov.get("specific_days_pct", 0),
        "sponsor_pct": comp_cov.get("sponsor_pct", 0),
        "options_pct": comp_cov.get("options_pct", 0),
        "regulatory_secondary_pct": reg_sec.get("coverage_pct", 0),
    }


def _build_delta_section(snap_dir: Path, prior_date: Optional[str]) -> Dict[str, Any]:
    """Turnover and tier changes vs prior."""
    delta = _load_json(snap_dir / "phase2_run_delta_details.json")
    if not delta:
        return {"available": False, "prior_date": prior_date}

    pt = delta.get("portfolio_turnover") or {}
    return {
        "available": True,
        "prior_date": (delta.get("prior") or {}).get("date", prior_date),
        "entrants": pt.get("entrants", []),
        "exits": pt.get("exits", []),
        "name_turnover_pct": pt.get("name_turnover_pct", 0),
        "weight_l1_delta": pt.get("weight_l1_delta", 0),
        "tier_current": delta.get("tier_distribution", {}).get("current", {}),
        "tier_prior": delta.get("tier_distribution", {}).get("prior", {}),
        "top_catalysts": delta.get("top_catalysts", []),
    }


def _build_performance_section() -> Dict[str, Any]:
    """Shadow portfolio performance summary."""
    metrics = _load_json(SHADOW_DIR / "portfolio_metrics.json")
    if not metrics:
        return {"available": False}

    return {
        "available": True,
        "n_periods": metrics.get("n_periods", 0),
        "cumulative_return_pct": metrics.get("cumulative_return_pct", 0),
        "cumulative_excess_pct": metrics.get("cumulative_excess_pct", 0),
        "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
        "sharpe_ratio": metrics.get("sharpe_ratio", 0),
        "win_rate": metrics.get("win_rate", 0),
        "total_pnl_usd": metrics.get("total_pnl_usd", 0),
        "best_period": metrics.get("best_period", {}),
        "worst_period": metrics.get("worst_period", {}),
        "sleeve_attribution": metrics.get("sleeve_attribution", {}),
    }


def _build_readiness_section(as_of_date: str) -> Dict[str, Any]:
    """Latest readiness scorecard."""
    sc = _load_json(READINESS_DIR / f"scorecard_{as_of_date}.json")
    if not sc:
        # Try to find the latest scorecard
        candidates = sorted(READINESS_DIR.glob("scorecard_*.json"))
        if candidates:
            sc = _load_json(candidates[-1])
    if not sc:
        return {"available": False}

    checks_summary = {}
    warn_checks = []
    fail_checks = []
    for check in sc.get("checks", []):
        name = check.get("name", "?")
        status = check.get("status", "?")
        checks_summary[name] = status
        if status == "WARN":
            warn_checks.append(f"{name}: {check.get('detail', check.get('value', ''))}")
        elif status == "FAIL":
            fail_checks.append(f"{name}: {check.get('detail', check.get('value', ''))}")

    return {
        "available": True,
        "verdict": sc.get("verdict", "?"),
        "checks": checks_summary,
        "warn_checks": warn_checks,
        "fail_checks": fail_checks,
        "n_perf_rows": sc.get("context", {}).get("n_performance_rows", 0),
    }


def _build_om11_section(as_of_date: str) -> Dict[str, Any]:
    """Options Monitor v1.1 verdict summary for ops digest."""
    v11_path = REPO_ROOT / "artifacts" / "options_verdict" / f"{as_of_date}_verdict_v11.json"
    data = _load_json(v11_path)
    if not data:
        return {"available": False}

    verdicts = data.get("verdicts", [])
    high = [v for v in verdicts if v.get("om11_monitor_verdict") == "HIGH"]
    watch = [v for v in verdicts if v.get("om11_monitor_verdict") == "WATCH"]
    new_v = [v for v in verdicts if v.get("state") == "NEW"]
    resolved = [v for v in verdicts if v.get("state") == "RESOLVED"]

    # Top trade biases
    trade_biases = {}
    for v in verdicts:
        bias = v.get("om11_trade_bias", "NO_ACTION")
        if bias != "NO_ACTION":
            trade_biases[bias] = trade_biases.get(bias, 0) + 1

    return {
        "available": True,
        "n_active": data.get("n_active", 0),
        "n_high": len(high),
        "n_watch": len(watch),
        "n_new": len(new_v),
        "n_resolved": len(resolved),
        "high_tickers": [v["ticker"] for v in high[:5]],
        "new_tickers": [v["ticker"] for v in new_v[:5]],
        "trade_biases": trade_biases,
    }


def _build_surface_delta_section(snap_dir: Path) -> Dict[str, Any]:
    """Optional post-open surface delta briefing (linked sidecar).

    Reads surface_delta.json produced by surface_delta_monitor.py.
    Only available when that tool has run for this snapshot date.
    """
    sd = _load_json(snap_dir / "surface_delta.json")
    if not sd:
        return {"available": False}

    # Summarize: just counts and top alert names
    deltas = sd.get("deltas", [])
    alert_names = [d["ticker"] for d in deltas if d.get("severity") == "alert"]
    watch_names = [d["ticker"] for d in deltas if d.get("severity") == "watch"]

    return {
        "available": True,
        "prior_date": sd.get("prior_date", "?"),
        "live_mode": sd.get("live_mode", False),
        "n_compared": sd.get("n_compared", 0),
        "n_alert": sd.get("n_alert", 0),
        "n_watch": sd.get("n_watch", 0),
        "alert_names": alert_names[:10],  # cap for digest brevity
        "watch_names": watch_names[:10],
    }


def _build_nearest_catalysts(snap_dir: Path, n: int = 10) -> List[Dict[str, Any]]:
    """Extract nearest catalysts from delta details."""
    delta = _load_json(snap_dir / "phase2_run_delta_details.json")
    if not delta:
        return []
    tc = delta.get("top_catalysts", {})
    if isinstance(tc, dict):
        tc = tc.get("current", [])
    return tc[:n]


# ---------------------------------------------------------------------------
# Asymmetry outliers — implied_vs_realized mispricing flags
# ---------------------------------------------------------------------------

_UNDERPRICED_THRESHOLD = 0.7  # implied < 70% of historical realized
_OVERPRICED_THRESHOLD = 2.0  # implied > 2x historical realized


def _build_asymmetry_outliers(snap_dir: Path) -> Dict[str, Any]:
    """Flag top-30 names where implied move diverges from historical realized.

    Uses event_premium_decomp.json (if available) or the asymmetry score output.
    Returns underpriced (market not pricing enough) and overpriced lists.
    """
    epd_path = snap_dir / "event_premium_decomp.json"
    if not epd_path.exists():
        return {"available": False}

    epd = _load_json(epd_path)
    if not epd or not epd.get("names"):
        return {"available": False}

    # Also load asymmetry score if available
    as_of = snap_dir.name
    asym_path = REPO_ROOT / "output" / "ranker_eval" / f"asymmetry_score_{as_of}.json"
    asym_lookup: Dict[str, Dict] = {}
    if asym_path.exists():
        asym_data = _load_json(asym_path)
        if asym_data:
            for entry in asym_data.get("names", []):
                asym_lookup[entry.get("ticker", "")] = entry

    underpriced: List[Dict[str, Any]] = []
    overpriced: List[Dict[str, Any]] = []

    for name in epd.get("names", []):
        ticker = name.get("ticker", "")
        ivr = name.get("epd_implied_vs_realized_ratio")
        if ivr is None:
            continue

        regime = name.get("epd_surface_regime", "")
        quality = name.get("epd_quality", "")
        asym = asym_lookup.get(ticker, {})
        asym_score = asym.get("asymmetry_score")
        asym_rank = asym.get("asymmetry_rank")
        liq = asym.get("opt_liquidity_state", "")

        entry = {
            "ticker": ticker,
            "implied_vs_realized": round(ivr, 2),
            "surface_regime": regime,
            "epd_quality": quality,
            "asymmetry_score": asym_score,
            "asymmetry_rank": asym_rank,
            "liquidity": liq,
        }

        if ivr < _UNDERPRICED_THRESHOLD:
            entry["flag"] = "UNDERPRICED"
            underpriced.append(entry)
        elif ivr > _OVERPRICED_THRESHOLD:
            entry["flag"] = "OVERPRICED"
            overpriced.append(entry)

    # Sort: underpriced by ratio ascending (cheapest first), overpriced by ratio descending
    underpriced.sort(key=lambda x: x["implied_vs_realized"])
    overpriced.sort(key=lambda x: -x["implied_vs_realized"])

    return {
        "available": True,
        "n_underpriced": len(underpriced),
        "n_overpriced": len(overpriced),
        "underpriced": underpriced,
        "overpriced": overpriced[:5],  # cap overpriced to top 5 (most are overpriced)
    }


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def _build_review_packet_summary() -> Dict[str, Any]:
    """Load latest review packet and extract summary for ops digest."""
    review_dir = REPO_ROOT / "artifacts" / "review"
    if not review_dir.exists():
        return {"available": False}
    files = sorted(review_dir.glob("*_review_packet.json"), reverse=True)
    if not files:
        return {"available": False}
    try:
        rp = json.loads(files[0].read_text())
        timing = rp.get("timing", {})
        cal = rp.get("calibration", {})
        confusion = rp.get("confusion", {})
        rq = rp.get("review_queue", {})
        priorities = rq.get("top_priorities", [])[:3]
        return {
            "available": True,
            "packet_date": rp.get("snapshot_date"),
            "n_warnings": timing.get("n_warnings", 0),
            "calibration_status": rp.get("calibration_status"),
            "brier": cal.get("overall_brier"),
            "n_resolved": cal.get("n_resolved", 0),
            "herald_accuracy": confusion.get("accuracy"),
            "n_labeled": confusion.get("n_labeled", 0),
            "top_priorities": [p.get("ticker", "?") for p in priorities],
        }
    except Exception:
        return {"available": False}


def build_ops_digest(
    as_of_date: str,
    *,
    snapshots_dir: Path = SNAPSHOT_DIR,
    shadow_dir: Path = SHADOW_DIR,
    readiness_dir: Path = READINESS_DIR,
) -> Dict[str, Any]:
    """Build the unified operations digest.

    Returns structured dict with all sections.
    """
    snap_dir = snapshots_dir / as_of_date
    if not snap_dir.exists():
        return {"error": f"No snapshot for {as_of_date}"}

    prior_date = _find_prior_snapshot(snapshots_dir, as_of_date)

    # Pipeline completion status — check run_manifest.json
    pipeline_status = _build_pipeline_status(snap_dir, as_of_date)

    health = _build_health_section(snap_dir)
    coverage = _build_coverage_section(snap_dir)
    delta = _build_delta_section(snap_dir, prior_date)
    performance = _build_performance_section()
    readiness = _build_readiness_section(as_of_date)
    catalysts = _build_nearest_catalysts(snap_dir)
    surface_delta = _build_surface_delta_section(snap_dir)

    # Options Monitor v1.1 verdict summary
    om11_summary = _build_om11_section(as_of_date)

    # Compute overall attention level
    n_fails = len([a for a in health["alerts"] if a["level"] == "FAIL"])
    n_warns = len([a for a in health["alerts"] if a["level"] == "WARN"])
    readiness_verdict = readiness.get("verdict", "?") if readiness.get("available") else "?"
    pipeline_failed = pipeline_status.get("overall_status") == "FAIL"
    pipeline_missing = not pipeline_status.get("available", False)

    if pipeline_failed or n_fails > 0 or readiness_verdict == "HOLD":
        attention = "ACTION_REQUIRED"
    elif pipeline_missing or n_warns > 0 or readiness_verdict == "REVIEW":
        attention = "REVIEW"
    else:
        attention = "CLEAR"

    # Collect action items with root-cause codes and new/carried-over flags
    action_items: List[Dict[str, str]] = []

    # Pipeline status action items
    if pipeline_failed:
        action_items.append(
            {
                "code": "pipeline:FAIL",
                "level": "FAIL",
                "source": "pipeline",
                "detail": f"Pipeline status={pipeline_status.get('overall_status')} "
                f"(fail={pipeline_status.get('gates_fail', 0)}, "
                f"warn={pipeline_status.get('gates_warn', 0)})",
                "new": True,
            }
        )
    elif pipeline_missing:
        action_items.append(
            {
                "code": "pipeline:MISSING",
                "level": "WARN",
                "source": "pipeline",
                "detail": pipeline_status.get("detail", "run_manifest.json not found"),
                "new": True,
            }
        )
    elif pipeline_status.get("stale"):
        action_items.append(
            {
                "code": "pipeline:STALE",
                "level": "WARN",
                "source": "pipeline",
                "detail": pipeline_status.get("detail", "Manifest date mismatch"),
                "new": True,
            }
        )
    prior_digest = _load_json(OUT_DIR / f"{prior_date}_digest.json") if prior_date else None
    prior_codes = set()
    if prior_digest:
        for item in prior_digest.get("action_items", []):
            if isinstance(item, dict):
                prior_codes.add(item.get("code", ""))
            elif isinstance(item, str):
                prior_codes.add(item)

    for alert in health["alerts"]:
        if alert["level"] in ("FAIL", "WARN"):
            code = f"{alert['source']}:{alert['level']}"
            is_new = code not in prior_codes and alert["detail"] not in prior_codes
            action_items.append(
                {
                    "code": code,
                    "level": alert["level"],
                    "source": alert["source"],
                    "detail": alert["detail"],
                    "new": is_new,
                }
            )
    for item in readiness.get("fail_checks", []):
        code = f"readiness:FAIL:{item.split(':')[0] if ':' in item else 'unknown'}"
        action_items.append(
            {
                "code": code,
                "level": "FAIL",
                "source": "readiness",
                "detail": item,
                "new": code not in prior_codes,
            }
        )
    for item in readiness.get("warn_checks", []):
        code = f"readiness:WARN:{item.split(':')[0] if ':' in item else 'unknown'}"
        action_items.append(
            {
                "code": code,
                "level": "WARN",
                "source": "readiness",
                "detail": item,
                "new": code not in prior_codes,
            }
        )

    # Load ruleset info from phase2_health (has the ID) + metadata
    p2_metrics = (health.get("_raw_phase2") or {}).get("metrics", {})
    if not p2_metrics:
        p2_raw = _load_json(snap_dir / "phase2_health.json") or {}
        p2_metrics = p2_raw.get("metrics", {})
    ruleset_id = p2_metrics.get("ruleset_id", "?")
    meta = _load_json(snap_dir / "metadata.json") or {}
    ruleset_version = meta.get("decision_engine_version", meta.get("engine_version", "?"))

    # Receipt provenance — find active receipt for ruleset context
    receipt_provenance: Dict[str, Any] = {"available": False}
    receipts_dir = REPO_ROOT / "artifacts" / "promotions"
    if receipts_dir.exists() and ruleset_id != "?":
        for rp in sorted(receipts_dir.glob("promotion_*.json"), reverse=True):
            try:
                rd = json.loads(rp.read_text(encoding="utf-8"))
                if rd.get("new_active_id") == ruleset_id:
                    receipt_provenance = {
                        "available": True,
                        "receipt_file": rp.name,
                        "promoted_at": rd.get("created_at_utc", ""),
                        "old_active_id": rd.get("old_active_id", ""),
                        "forced": rd.get("forced", False),
                        "gate_verdict": (rd.get("gate") or {}).get("verdict", "?"),
                    }
                    break
            except (json.JSONDecodeError, OSError):
                continue

    # Decision diff — top rank movers from drift report
    decision_diff: Dict[str, Any] = {"available": False}
    drift = _load_json(snap_dir / "drift_report.json")
    stability_diagnostics: Dict[str, Any] = {"available": False}
    if drift:
        dm = drift.get("metrics") or {}
        decision_diff = {
            "available": True,
            "prior_date": drift.get("prior_date"),
            "top20_entrants": dm.get("top20_entrants", []),
            "top20_exits": dm.get("top20_exits", []),
            "spearman_rho": dm.get("rank_spearman_rho"),
            "top20_overlap_pct": dm.get("top20_overlap_pct"),
            "top60_overlap_pct": dm.get("top60_overlap_pct"),
            "tier_migrations": dm.get("tier_migration_count"),
            "eligibility_changes": dm.get("eligibility_change_count"),
            "mean_rank_delta_top60": dm.get("mean_abs_rank_delta_top60"),
        }
        # Stability plumbing diagnostics (drift_report v1.2.0+) — separate the
        # "did market shift?" signal from the "did our plumbing break?" signal.
        sel_stats = (dm.get("score_delta_stats") or {}).get("selector_score") or {}
        stability_diagnostics = {
            "available": "action_change_pct" in dm,
            "action_change_pct": dm.get("action_change_pct"),
            "action_transition_matrix": dm.get("action_transition_matrix"),
            "mean_abs_selector_score_delta": sel_stats.get("mean_abs_delta"),
            "p95_abs_selector_score_delta": sel_stats.get("p95_abs_delta"),
            "max_coverage_drop_feature": dm.get("max_coverage_drop_feature"),
            "max_coverage_drop_pp": dm.get("max_coverage_drop_pp"),
            "coverage_drops_top5": dm.get("coverage_drops_top5") or [],
            "near_miss_share_pct": dm.get("near_miss_share_pct"),
            "near_miss": dm.get("near_miss") or {},
            # Plumbing-vs-regime heuristic: low overlap + coverage drop = plumbing
            "plumbing_suspect": bool(
                dm.get("top20_overlap_pct") is not None
                and dm.get("top20_overlap_pct") < 70.0
                and (dm.get("max_coverage_drop_pp") or 0.0) >= 10.0
            ),
        }

    # Asymmetry outliers — implied_vs_realized flags for discretionary review
    asymmetry_outliers = _build_asymmetry_outliers(snap_dir)

    # Review packet summary (timing + event quality)
    review_packet_summary = _build_review_packet_summary()

    return {
        "schema": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "prior_date": prior_date,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "attention": attention,
        "action_items": action_items,
        "ruleset": {"id": ruleset_id, "version": ruleset_version},
        "receipt_provenance": receipt_provenance,
        "pipeline_status": pipeline_status,
        "health": health,
        "coverage": coverage,
        "delta": delta,
        "decision_diff": decision_diff,
        "stability_diagnostics": stability_diagnostics,
        "performance": performance,
        "readiness": readiness,
        "nearest_catalysts": catalysts,
        "surface_delta": surface_delta,
        "options_monitor_v11": om11_summary,
        "asymmetry_outliers": asymmetry_outliers,
        "review_packet": review_packet_summary,
    }


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def format_digest_md(d: Dict[str, Any]) -> str:
    """Render the digest as a concise markdown briefing."""
    lines: List[str] = []

    attention = d.get("attention", "?")
    icon = {"CLEAR": "CLEAR", "REVIEW": "REVIEW", "ACTION_REQUIRED": "ACTION REQUIRED"}
    lines.append(f"# Ops Digest — {d['as_of_date']}")
    lines.append("")
    lines.append(f"**Status: {icon.get(attention, attention)}**  ")
    rs = d.get("ruleset", {})
    rp = d.get("receipt_provenance", {})
    lines.append(f"Ruleset: `{rs.get('id', '?')}` ({rs.get('version', '?')})  ")
    if rp.get("available"):
        lines.append(
            f"Promoted: {rp.get('promoted_at', '?')} (from `{rp.get('old_active_id', '?')}`, gate={rp.get('gate_verdict', '?')}, forced={rp.get('forced', '?')})  "
        )
    lines.append(f"Prior: {d.get('prior_date', 'none')}  ")
    lines.append(f"Generated: {d.get('generated_at', '')}")
    lines.append("")

    # Action items with new/carried-over flags
    action_items = d.get("action_items", [])
    if action_items:
        lines.append("## Action Items")
        lines.append("")
        for item in action_items:
            if isinstance(item, dict):
                flag = "NEW" if item.get("new") else "carried"
                lines.append(f"- [{flag}] [{item.get('source', '?')}] {item.get('detail', '?')}")
            else:
                lines.append(f"- {item}")
        lines.append("")
    else:
        lines.append("## No action items")
        lines.append("")

    # Health
    h = d.get("health", {})
    lines.append("## Health")
    lines.append("")
    lines.append("| Gate | Status |")
    lines.append("|------|--------|")
    lines.append(f"| Phase-2 | {h.get('phase2_status', '?')} |")
    lines.append(f"| Collection | {h.get('collection_status', '?')} |")
    lines.append(f"| Exposure alerts | {h.get('exposure_alerts', 0)} |")
    lines.append("")

    # Coverage
    c = d.get("coverage", {})
    lines.append("## Coverage")
    lines.append("")
    lines.append(
        "| Metric | Value |\n|--------|-------|\n"
        f"| Universe | {c.get('n_eligible', 0)}/{c.get('n_total', 0)} eligible |\n"
        f"| Catalyst | {c.get('catalyst_pct', 0):.1f}% |\n"
        f"| Sponsor | {c.get('sponsor_pct', 0):.1f}% |\n"
        f"| Options | {c.get('options_pct', 0):.1f}% |\n"
        f"| Regulatory (secondary) | {c.get('regulatory_secondary_pct', 0):.1f}% |"
    )
    lines.append("")

    # Delta
    delta = d.get("delta", {})
    if delta.get("available"):
        lines.append("## Delta vs Prior")
        lines.append("")
        entrants = delta.get("entrants", [])
        exits = delta.get("exits", [])
        if entrants or exits:
            lines.append(f"- Entrants ({len(entrants)}): {', '.join(entrants) if entrants else 'none'}")
            lines.append(f"- Exits ({len(exits)}): {', '.join(exits) if exits else 'none'}")
            lines.append(f"- Turnover: {delta.get('name_turnover_pct', 0):.1f}%")
        else:
            lines.append("No portfolio changes.")
        lines.append("")

    # Decision diff
    dd = d.get("decision_diff", {})
    if dd.get("available"):
        lines.append("## Decision Diff")
        lines.append("")
        lines.append(f"Spearman rho: {dd.get('spearman_rho', '?')}  ")
        lines.append(f"Top-20 overlap: {dd.get('top20_overlap_pct', '?')}%  ")
        lines.append(f"Top-60 overlap: {dd.get('top60_overlap_pct', '?')}%  ")
        lines.append(f"Tier migrations: {dd.get('tier_migrations', '?')}  ")
        lines.append(f"Eligibility changes: {dd.get('eligibility_changes', '?')}")
        lines.append("")
        t20_in = dd.get("top20_entrants", [])
        t20_out = dd.get("top20_exits", [])
        if t20_in or t20_out:
            lines.append(f"Top-20 entrants: {', '.join(t20_in) if t20_in else 'none'}  ")
            lines.append(f"Top-20 exits: {', '.join(t20_out) if t20_out else 'none'}")
            lines.append("")
        movers = dd.get("biggest_rank_movers", [])
        if movers:
            lines.append("Biggest rank movers:")
            for m in movers[:5]:
                if isinstance(m, dict):
                    lines.append(
                        f"  {m.get('ticker', '?')}: {m.get('prior_rank', '?')} → {m.get('current_rank', '?')} (Δ{m.get('shift', '?')})"
                    )
                else:
                    lines.append(f"  {m}")
            lines.append("")

    # Stability plumbing diagnostics — answers "plumbing or regime?"
    sd = d.get("stability_diagnostics", {})
    if sd.get("available"):
        lines.append("## Stability Plumbing")
        lines.append("")
        if sd.get("plumbing_suspect"):
            lines.append(
                "PLUMBING SUSPECT: top-20 overlap collapsed AND feature coverage dropped. "
                "Investigate inputs before attributing to market regime."
            )
            lines.append("")
        lines.append(f"Action change: {sd.get('action_change_pct', '?')}% of common tickers changed tier  ")
        lines.append(f"Mean |selector_score delta|: {sd.get('mean_abs_selector_score_delta', '?')}  ")
        lines.append(f"P95 |selector_score delta|: {sd.get('p95_abs_selector_score_delta', '?')}  ")
        lines.append(
            f"Near-miss @ K=30 (25bps): {sd.get('near_miss_share_pct', '?')}% of top-30 within 25bps of cutoff"
        )
        lines.append("")
        drops = sd.get("coverage_drops_top5") or []
        if drops:
            lines.append("Top feature coverage drops (prior → current):")
            for drop in drops:
                lines.append(
                    f"  {drop.get('feature', '?')}: {drop.get('prior_pct', '?')}% → {drop.get('cur_pct', '?')}% ({drop.get('delta_pp', '?'):+}pp)"
                )
            lines.append("")

    # Performance
    perf = d.get("performance", {})
    if perf.get("available"):
        lines.append("## Shadow Performance")
        lines.append("")
        lines.append(
            "| Metric | Value |\n|--------|-------|\n"
            f"| Cumulative | {perf.get('cumulative_return_pct', 0):+.2f}% |\n"
            f"| Excess vs XBI | {perf.get('cumulative_excess_pct', 0):+.2f}% (EW-vs-cap-wt gap included) |\n"
            f"| Max DD | {perf.get('max_drawdown_pct', 0):.2f}% |\n"
            f"| Sharpe | {perf.get('sharpe_ratio', 0):.3f} |\n"
            f"| Win rate | {perf.get('win_rate', 0):.0%} |\n"
            f"| PnL | ${perf.get('total_pnl_usd', 0):,.0f} |\n"
            f"| Periods | {perf.get('n_periods', 0)} |"
        )
        lines.append("")

        sleeve = perf.get("sleeve_attribution", {})
        if sleeve:
            lines.append("Sleeve attribution:")
            for bucket, pnl in sleeve.items():
                lines.append(f"  {bucket}: ${pnl:+,.0f}")
            lines.append("")

    # Readiness
    rd = d.get("readiness", {})
    if rd.get("available"):
        lines.append("## Readiness")
        lines.append("")
        lines.append(f"**Verdict: {rd.get('verdict', '?')}** ({rd.get('n_perf_rows', 0)} perf rows)")
        lines.append("")
        checks = rd.get("checks", {})
        if checks:
            lines.append("| Check | Status |")
            lines.append("|-------|--------|")
            for name, status in checks.items():
                lines.append(f"| {name} | {status} |")
            lines.append("")

    # Review Packet Summary (timing + event quality)
    rps = d.get("review_packet", {})
    if rps.get("available"):
        lines.append("## Review Packet")
        lines.append("")
        lines.append(f"Timing: {rps.get('n_warnings', 0)} warnings, cal={rps.get('calibration_status', '—')}")
        if rps.get("brier") is not None:
            lines.append(f"  Brier={rps['brier']}, {rps.get('n_resolved', 0)} resolved")
        if rps.get("herald_accuracy") is not None:
            lines.append(f"  Herald accuracy={rps['herald_accuracy']} ({rps.get('n_labeled', 0)} labeled)")
        if rps.get("top_priorities"):
            lines.append("  Priority: " + ", ".join(rps["top_priorities"]))
        lines.append("")

    # Nearest catalysts
    cats = d.get("nearest_catalysts", [])
    if cats:
        lines.append("## Nearest Catalysts")
        lines.append("")
        lines.append("| Ticker | Days | Tier |")
        lines.append("|--------|------|------|")
        for cat in cats[:10]:
            t = cat.get("ticker", "?")
            days = cat.get("catalyst_days", cat.get("days", "?"))
            tier = cat.get("tier_dev", cat.get("tier", "?"))
            lines.append(f"| {t} | {days} | {tier} |")
        lines.append("")

    # Surface delta (post-open sidecar — only present when surface_delta_monitor ran)
    sd = d.get("surface_delta", {})
    if sd.get("available"):
        source = "live" if sd.get("live_mode") else "snapshot"
        lines.append("## Surface Delta (post-open)")
        lines.append("")
        lines.append(
            f"vs {sd.get('prior_date', '?')} ({source}) | "
            f"{sd.get('n_compared', 0)} compared | "
            f"**{sd.get('n_alert', 0)} alert** / {sd.get('n_watch', 0)} watch"
        )
        alert_names = sd.get("alert_names", [])
        if alert_names:
            lines.append("")
            lines.append(f"Alert names: {', '.join(alert_names)}")
        watch_names = sd.get("watch_names", [])
        if watch_names:
            lines.append(f"Watch names: {', '.join(watch_names)}")
        lines.append("")

    # Asymmetry outliers
    ao = d.get("asymmetry_outliers", {})
    if ao.get("available"):
        lines.append("## Asymmetry Outliers")
        lines.append("")
        under = ao.get("underpriced", [])
        over = ao.get("overpriced", [])
        if under:
            lines.append(f"**Underpriced ({len(under)})** — market not pricing enough vs historical:")
            lines.append("")
            lines.append("| Ticker | Impl/Real | Regime | Liq | Asym Rank |")
            lines.append("|--------|-----------|--------|-----|-----------|")
            for u in under:
                ar = u.get("asymmetry_rank", "—")
                lines.append(
                    f"| {u['ticker']} | {u['implied_vs_realized']:.2f} | "
                    f"{u.get('surface_regime', '?')[:20]} | {u.get('liquidity', '?')} | {ar} |"
                )
            lines.append("")
        if over:
            lines.append(f"**Overpriced ({len(over)})** — market pricing larger move than historical:")
            lines.append("")
            lines.append("| Ticker | Impl/Real | Regime | Liq |")
            lines.append("|--------|-----------|--------|-----|")
            for o in over[:5]:
                lines.append(
                    f"| {o['ticker']} | {o['implied_vs_realized']:.1f}x | "
                    f"{o.get('surface_regime', '?')[:20]} | {o.get('liquidity', '?')} |"
                )
            lines.append("")
        if not under and not over:
            lines.append("No outliers (all within 0.7x–2.0x implied/realized range).")
            lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def write_ops_digest(
    digest: Dict[str, Any],
    out_dir: Path = OUT_DIR,
) -> Dict[str, str]:
    """Write JSON + MD digest files. Returns paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    as_of = digest["as_of_date"]

    json_path = out_dir / f"{as_of}_digest.json"
    md_path = out_dir / f"{as_of}_digest.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(digest, f, indent=2, default=str)

    md_content = format_digest_md(digest)
    md_path.write_text(md_content, encoding="utf-8")

    return {"json_path": str(json_path), "md_path": str(md_path)}


# ---------------------------------------------------------------------------
# Programmatic entry point (for daily pipeline)
# ---------------------------------------------------------------------------


def run_ops_digest(as_of_date: str) -> Dict[str, Any]:
    """Build and write the ops digest. Returns digest + paths."""
    digest = build_ops_digest(as_of_date)
    if "error" in digest:
        return digest
    paths = write_ops_digest(digest)
    digest["_paths"] = paths
    return digest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build post-run operations digest")
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Snapshot date (YYYY-MM-DD). Default: latest snapshot.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=SNAPSHOT_DIR,
    )
    parser.add_argument("--stdout", action="store_true", help="Print markdown to stdout")
    args = parser.parse_args()
    started = time.perf_counter()

    as_of = args.as_of_date or _find_latest_snapshot(args.snapshot_dir)
    if not as_of:
        print("No snapshots found.", file=sys.stderr)
        try:
            from tools.agent_skill_telemetry import log_agent_run

            log_agent_run(
                "build_ops_digest",
                "Ops digest build",
                success=False,
                error="no snapshots",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception:
            pass
        sys.exit(1)

    digest = build_ops_digest(as_of, snapshots_dir=args.snapshot_dir)
    if "error" in digest:
        print(f"Error: {digest['error']}", file=sys.stderr)
        try:
            from tools.agent_skill_telemetry import log_agent_run

            log_agent_run(
                "build_ops_digest",
                f"Ops digest for {as_of}",
                inputs={"as_of_date": as_of},
                outputs={"error": digest["error"]},
                success=False,
                error=digest["error"],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception:
            pass
        sys.exit(1)

    paths = write_ops_digest(digest)
    print(f"Digest: {paths['md_path']}")

    try:
        from tools.agent_skill_telemetry import log_agent_run

        log_agent_run(
            "build_ops_digest",
            f"Ops digest for {as_of}",
            inputs={"as_of_date": as_of},
            outputs={"attention": digest.get("attention")},
            success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception:
        pass

    if args.stdout:
        print()
        print(format_digest_md(digest))


if __name__ == "__main__":
    main()
