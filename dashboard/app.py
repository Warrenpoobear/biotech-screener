"""Read-only operator dashboard — Policy Control Tower.

Reads existing JSON/MD artifacts and serves a web UI. Does NOT modify
any production state, positions, rankings, or execution.

Usage:
    python dashboard/app.py
    python dashboard/app.py --port 8080
    uvicorn dashboard.app:app --reload --port 8050
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Event type score → human-readable label (Spec 056)
EVENT_TYPE_LABELS = {3: "PDUFA", 2: "Data Readout", 1: "Clinical Milestone", 0: "Low/None"}

app = FastAPI(title="Biotech Screener — Policy Control Tower", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# --- Data loaders ---


def _load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _load_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _available_snapshot_dates() -> List[str]:
    snap_dir = REPO_ROOT / "data" / "snapshots"
    import re

    date_pat = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    return sorted(
        (d.name for d in snap_dir.iterdir() if d.is_dir() and date_pat.match(d.name)),
        reverse=True,
    )


def _load_positions(date: str) -> List[Dict]:
    path = REPO_ROOT / "artifacts" / "live_shadow" / "positions" / f"{date}.json"
    data = _load_json(path)
    if not data:
        return []
    return data.get("positions", [])


def _load_rankings(date: str) -> Dict[str, Dict]:
    rpath = REPO_ROOT / "data" / "snapshots" / date / "rankings.csv"
    if not rpath.exists():
        return {}
    with open(rpath, encoding="utf-8") as f:
        return {r["ticker"]: r for r in csv.DictReader(f)}


def _load_earnings_lookup() -> Dict[str, str]:
    """Load most recent earnings raw file into a ticker -> date lookup."""
    edir = REPO_ROOT / "artifacts" / "earnings_sync"
    if not edir.is_dir():
        return {}
    raw_files = sorted(edir.glob("earnings_raw_*.json"), reverse=True)
    if not raw_files:
        return {}
    data = _load_json(raw_files[0])
    if not data:
        return {}
    lookup: Dict[str, str] = {}
    for row in data.get("rows", []):
        t = row.get("symbol", "")
        d = row.get("earnings_date", "")
        if t and d:
            if t not in lookup or d < lookup[t]:
                lookup[t] = d
    return lookup


def _load_policy_comparison(date: str) -> Optional[Dict]:
    path = REPO_ROOT / "artifacts" / "policy_shadow" / "tier_weighted" / f"{date}_comparison.json"
    return _load_json(path)


def _load_policy_history() -> List[Dict]:
    path = REPO_ROOT / "artifacts" / "policy_shadow" / "tier_weighted" / "history.jsonl"
    rows = _load_jsonl(path)
    # Deduplicate by date
    seen = set()
    deduped = []
    for r in rows:
        d = r.get("date")
        if d and d not in seen:
            seen.add(d)
            deduped.append(r)
    return sorted(deduped, key=lambda r: r["date"])


def _load_bioshort_watch() -> Optional[Dict]:
    watch_dir = REPO_ROOT / "artifacts" / "bioshort_watch"
    if not watch_dir.exists():
        return None
    files = sorted(watch_dir.glob("*_watch.json"), reverse=True)
    return _load_json(files[0]) if files else None


_FRESHNESS_WARN_DAYS = 7
_FRESHNESS_ERROR_DAYS = 14


def _bioshort_freshness_meta() -> dict:
    """Compute freshness metadata for the four bioshort endpoints."""
    verdict_path = REPO_ROOT / "output" / "hedge_report" / "BIOSHORT_VERDICT.json"
    data = _load_json(verdict_path)
    as_of_date_str = (data or {}).get("as_of_date")
    if not as_of_date_str:
        return {
            "report_as_of_date": None,
            "report_age_days": None,
            "freshness_status": "FRESHNESS_UNKNOWN",
            "freshness_warn_days": _FRESHNESS_WARN_DAYS,
            "freshness_error_days": _FRESHNESS_ERROR_DAYS,
        }
    try:
        from datetime import date

        report_date = date.fromisoformat(as_of_date_str)
        age_days = (date.today() - report_date).days
    except (ValueError, TypeError):
        return {
            "report_as_of_date": as_of_date_str,
            "report_age_days": None,
            "freshness_status": "FRESHNESS_UNKNOWN",
            "freshness_warn_days": _FRESHNESS_WARN_DAYS,
            "freshness_error_days": _FRESHNESS_ERROR_DAYS,
        }
    if age_days <= _FRESHNESS_WARN_DAYS:
        status = "FRESH"
    elif age_days <= _FRESHNESS_ERROR_DAYS:
        status = "STALE_WARNING"
    else:
        status = "STALE_ERROR"
    return {
        "report_as_of_date": as_of_date_str,
        "report_age_days": age_days,
        "freshness_status": status,
        "freshness_warn_days": _FRESHNESS_WARN_DAYS,
        "freshness_error_days": _FRESHNESS_ERROR_DAYS,
    }


def _load_ops_digest(date: str) -> Optional[str]:
    path = REPO_ROOT / "artifacts" / "ops_digest" / f"{date}_digest.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _load_attribution(date: str) -> Optional[Dict]:
    path = REPO_ROOT / "artifacts" / "live_shadow" / "attribution" / date / "ATTRIBUTION_PACKET.json"
    return _load_json(path)


def _load_phase2_health(date: str) -> Optional[Dict]:
    path = REPO_ROOT / "data" / "snapshots" / date / "phase2_health.json"
    return _load_json(path)


def _load_timing_hazard(date: str) -> Optional[Dict]:
    path = REPO_ROOT / "artifacts" / "timing_hazard" / f"timing_hazard_{date}.json"
    return _load_json(path)


def _load_production_monitor(date: str) -> Optional[Dict]:
    path = REPO_ROOT / "artifacts" / "production_monitor" / f"{date}_monitor.json"
    return _load_json(path)


def _load_factor_drift(date: str) -> Optional[Dict]:
    path = REPO_ROOT / "artifacts" / "factor_drift" / f"{date}_factor_drift.json"
    return _load_json(path)


def _load_timing_hazard_review() -> Optional[Dict]:
    path = REPO_ROOT / "output" / "timing_hazard_review" / "timing_hazard_review.json"
    return _load_json(path)


def _load_shadow_performance() -> List[Dict]:
    path = REPO_ROOT / "artifacts" / "live_shadow" / "performance.csv"
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in csv.reader(f):
            if len(line) >= 10 and line[0] == "live_shadow_perf.v1":
                try:
                    rows.append(
                        {
                            "date": line[1],
                            "pnl": float(line[3]) if line[3] else 0,
                            "pnl_pct": float(line[4]) if line[4] else 0,
                            "xbi_pct": float(line[5]) if line[5] else 0,
                            "excess": float(line[6]) if line[6] else 0,
                            "positions": int(line[7]) if line[7] else 0,
                            "turnover": float(line[8]) if line[8] else 0,
                        }
                    )
                except (ValueError, IndexError):
                    pass
    return rows


# --- Routes ---


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, date: str = ""):
    dates = _available_snapshot_dates()
    if not date and dates:
        date = dates[0]

    # Load all data for this date
    positions = _load_positions(date)
    rankings = _load_rankings(date)
    earnings_lookup = _load_earnings_lookup()
    policy = _load_policy_comparison(date)
    policy_history = _load_policy_history()
    bioshort = _load_bioshort_watch()
    phase2 = _load_phase2_health(date)
    attribution = _load_attribution(date)
    perf = _load_shadow_performance()
    timing_hazard = _load_timing_hazard(date)
    production_monitor = _load_production_monitor(date)
    factor_drift = _load_factor_drift(date)

    # Enrich positions with rankings data
    enriched_positions = []
    for p in positions:
        t = p.get("ticker", "")
        r = rankings.get(t, {})
        # Event type score from rankings (Spec 056 overlay)
        ets_raw = r.get("event_type_score", "")
        try:
            ets_val = int(float(ets_raw)) if ets_raw != "" else None
        except (ValueError, TypeError):
            ets_val = None

        enriched_positions.append(
            {
                "ticker": t,
                "weight": p.get("weight_pct", 0),
                "bucket": p.get("bucket", ""),
                "tier": r.get("tier_dev", "?"),
                "rank": r.get("actionable_rank", ""),
                "mom": r.get("mom_state", ""),
                "risk": r.get("risk_flags", ""),
                "catalyst_days": r.get("catalyst_days", ""),
                "catalyst_family": r.get("catalyst_family", ""),
                "event_type_score": ets_val,
                "event_type_label": EVENT_TYPE_LABELS.get(ets_val, "—") if ets_val is not None else "—",
                "next_earnings_date": r.get("next_earnings_date", "") or earnings_lookup.get(t, ""),
            }
        )
    enriched_positions.sort(key=lambda p: ({"A": 0, "B": 1, "C": 2, "D": 3}.get(p["tier"], 4), -p["weight"]))

    # Catalyst quality summary (event type distribution)
    catalyst_quality = Counter()
    for p in enriched_positions:
        label = p.get("event_type_label", "—")
        catalyst_quality[label] += 1

    # Tier summary
    tier_summary = Counter(p["tier"] for p in enriched_positions)
    tier_weights = defaultdict(float)
    for p in enriched_positions:
        tier_weights[p["tier"]] += p["weight"]

    # Risky holds: C/D tier at >= 2% OR headwind + drawdown
    risky_holds = [
        p
        for p in enriched_positions
        if (p["tier"] in ("C", "D") and p["weight"] >= 2.0)
        or (p["mom"] == "headwind" and "deep_drawdown" in p.get("risk", ""))
    ]

    # Policy cumulative
    cum_current = 0.0
    cum_tiered = 0.0
    cum_exit = 0.0
    policy_path = []
    for row in policy_history:
        pc = row.get("pnl_current") or 0
        pt = row.get("pnl_tiered") or 0
        pe = row.get("pnl_exit") or 0
        cum_current += pc
        cum_tiered += pt
        cum_exit += pe
        if abs(pc) > 0.001:
            policy_path.append(
                {
                    "date": row.get("date", ""),
                    "current": round(cum_current, 2),
                    "tiered": round(cum_tiered, 2),
                    "exit": round(cum_exit, 2),
                }
            )

    # Attribution top/bottom
    attr_top = []
    attr_bottom = []
    if attribution:
        tc = attribution.get("top_contributors", {})
        attr_top = tc.get("top", [])[:5]
        attr_bottom = tc.get("bottom", [])[:5]

    # Shadow perf summary
    perf_trading = [p for p in perf if abs(p["pnl"]) > 0.01]
    cum_pnl = sum(p["pnl"] for p in perf)
    cum_pnl_pct = sum(p["pnl_pct"] for p in perf)

    # Alerts
    alerts = []
    if bioshort:
        for a in bioshort.get("alerts", []):
            alerts.append({"source": "bioshort", "level": bioshort.get("alert_level", "?"), "text": a})
    if policy:
        for ticker in policy.get("excluded_by_exit", []):
            alerts.append({"source": "policy", "level": "MEDIUM", "text": f"Exit overlay excluded: {ticker}"})
    if phase2 and phase2.get("status") == "FAIL":
        for reason in phase2.get("reasons", []):
            alerts.append({"source": "health", "level": "WARN", "text": f"Phase2: {reason}"})
    for p in risky_holds:
        alerts.append(
            {
                "source": "policy",
                "level": "LOW",
                "text": f"Risky hold: {p['ticker']} tier={p['tier']} wt={p['weight']:.1f}% mom={p['mom']} risk={p['risk']}",
            }
        )

    # Timing hazard overlay
    timing_warnings = []
    timing_review = _load_timing_hazard_review()
    timing_summary = {"n_catalysts": 0, "n_warnings": 0, "mean_on_time": None, "confidence_dist": {}}
    if timing_review:
        cal = timing_review.get("calibration", {})
        timing_summary["brier"] = cal.get("brier_score")
        timing_summary["ece"] = cal.get("ece")
        timing_summary["cal_verdict"] = cal.get("verdict", "")
        timing_summary["overconfidence"] = cal.get("overconfidence")
    timing_by_ticker: Dict[str, Dict] = {}
    if timing_hazard and "catalysts" in timing_hazard:
        timing_summary.update(
            {
                "n_catalysts": timing_hazard.get("n_catalysts", 0),
                "n_warnings": timing_hazard.get("n_warnings", 0),
                "mean_on_time": timing_hazard.get("mean_on_time_prob"),
                "confidence_dist": timing_hazard.get("confidence_dist", {}),
            }
        )
        for cat in timing_hazard["catalysts"]:
            t = cat.get("ticker", "")
            if t:
                timing_by_ticker[t] = cat
            if cat.get("execution_warning_flag"):
                timing_warnings.append(cat)
        # Feed timing warnings into alerts
        if timing_warnings:
            alerts.append(
                {
                    "source": "timing",
                    "level": "MEDIUM" if len(timing_warnings) >= 3 else "LOW",
                    "text": f"{len(timing_warnings)} catalyst(s) with execution warnings",
                }
            )
        for tw in timing_warnings:
            reasons = ", ".join(tw.get("warning_reasons", []))
            alerts.append(
                {
                    "source": "timing",
                    "level": "LOW",
                    "text": f"{tw.get('ticker', '?')} P(on_time)={tw.get('on_time_prob', 0):.0%} [{reasons}]",
                }
            )

    # Production monitor alerts
    prod_health = {}
    if production_monitor and "alerts" in production_monitor:
        prod_health = {
            "attention": production_monitor.get("attention", "?"),
            "hhi": production_monitor.get("hhi"),
            "jaccard": (production_monitor.get("overlap") or {}).get("jaccard"),
            "rank_corr": production_monitor.get("rank_correlation"),
            "ranker_divergent": (production_monitor.get("ranker_drift") or {}).get("n_divergent"),
            "catalyst_quality": production_monitor.get("catalyst_quality", {}),
        }
        for pa in production_monitor["alerts"]:
            alerts.append(
                {
                    "source": "production",
                    "level": pa.get("level", "WARN"),
                    "text": f"[{pa.get('code', '?')}] {pa.get('detail', '')}",
                }
            )

    # Factor drift alerts
    factor_drift_health = {}
    if factor_drift and "alerts" in factor_drift:
        factor_drift_health = {
            "attention": factor_drift.get("attention", "?"),
            "jaccard": factor_drift.get("jaccard_prev"),
            "hhi": factor_drift.get("hhi"),
            "universe_size": factor_drift.get("universe_size"),
        }
        for fa in factor_drift["alerts"]:
            alerts.append(
                {
                    "source": "factor_drift",
                    "level": fa.get("level", "YELLOW"),
                    "text": f"[{fa.get('code', '?')}] {fa.get('detail', '')}",
                }
            )

    # Enrich positions with timing confidence
    for p in enriched_positions:
        th = timing_by_ticker.get(p["ticker"], {})
        p["timing_confidence"] = th.get("timing_confidence_bucket", "")
        p["on_time_prob"] = th.get("on_time_prob")

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "date": date,
            "dates": dates[:30],
            "positions": enriched_positions,
            "n_positions": len(enriched_positions),
            "tier_summary": dict(tier_summary),
            "tier_weights": {k: round(v, 1) for k, v in tier_weights.items()},
            "risky_holds": risky_holds,
            "policy": policy,
            "policy_path": policy_path,
            "policy_gap_tiered": round(cum_tiered - cum_current, 2),
            "policy_gap_exit": round(cum_exit - cum_current, 2),
            "bioshort": bioshort,
            "phase2": phase2,
            "alerts": alerts,
            "attr_top": attr_top,
            "attr_bottom": attr_bottom,
            "cum_pnl": round(cum_pnl, 0),
            "cum_pnl_pct": round(cum_pnl_pct, 2),
            "perf_trading": perf_trading[-10:],
            "timing_summary": timing_summary,
            "timing_warnings": timing_warnings,
            "catalyst_quality": dict(catalyst_quality),
            "prod_health": prod_health,
            "factor_drift_health": factor_drift_health,
            "now": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        },
    )


@app.get("/api/positions/{date}")
async def api_positions(date: str):
    return _load_positions(date)


@app.get("/api/policy/{date}")
async def api_policy(date: str):
    return _load_policy_comparison(date) or {"error": "not found"}


@app.get("/api/policy-history")
async def api_policy_history():
    return _load_policy_history()


@app.get("/api/bioshort")
async def api_bioshort():
    return _load_bioshort_watch() or {"error": "not found"}


# --- New endpoints for React dashboard (v2) ---


@app.get("/api/dates")
async def api_dates():
    """List available snapshot dates, newest first."""
    return _available_snapshot_dates()


@app.get("/api/rankings/{date}")
async def api_rankings(date: str, top_n: int = 200):
    """Full rankings table for a given date. Returns list of row dicts."""
    rpath = REPO_ROOT / "data" / "snapshots" / date / "rankings.csv"
    if not rpath.exists():
        return {"error": f"No rankings for {date}"}
    with open(rpath, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Sort by actionable_rank, filter to ranked names
    ranked = []
    for r in rows:
        ar = r.get("actionable_rank", "").strip()
        if ar:
            try:
                r["_rank_int"] = int(ar)
                ranked.append(r)
            except ValueError:
                pass
    ranked.sort(key=lambda r: r["_rank_int"])
    for r in ranked:
        r.pop("_rank_int", None)
    return ranked[:top_n]


@app.get("/api/decision_portfolio/{date}")
async def api_decision_portfolio(date: str):
    """DEM decision_portfolio.csv for a given date."""
    # Check output snapshots first (produced by daily production)
    for base in [REPO_ROOT / "output" / "snapshots", REPO_ROOT / "data" / "snapshots"]:
        path = base / date / "decision_portfolio.csv"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            ranked = [r for r in rows if r.get("actionable_rank", "").strip()]
            ranked.sort(key=lambda r: int(r["actionable_rank"]))
            return ranked
    return {"error": f"No decision_portfolio for {date}"}


@app.get("/api/ticker/{ticker}")
async def api_ticker_detail(ticker: str, date: str = ""):
    """Merged ticker detail: ranking row + position + options + CRT."""
    if not date:
        dates = _available_snapshot_dates()
        date = dates[0] if dates else ""
    if not date:
        return {"error": "No snapshots available"}

    # Rankings row
    rankings = _load_rankings(date)
    ranking_row = rankings.get(ticker.upper(), {})

    # Shadow position
    positions = _load_positions(date)
    position = next((p for p in positions if p.get("ticker") == ticker.upper()), {})

    # Options diagnostics from rankings row
    options = {
        "atm_iv_change_5d": ranking_row.get("atm_iv_change_5d", ""),
        "opt_rr_25d": ranking_row.get("opt_rr_25d", ""),
        "actual_implied_move_pctile": ranking_row.get("actual_implied_move_pctile", ""),
        "iv_percentile_30d": ranking_row.get("iv_percentile_30d", ""),
    }

    # CRT resolutions
    crt = []
    crt_dir = REPO_ROOT / "data" / "snapshots" / "resolutions"
    if crt_dir.exists():
        import glob

        for f in glob.glob(str(crt_dir / "**" / f"{ticker.upper()}_*.json"), recursive=True):
            try:
                with open(f, encoding="utf-8") as fh:
                    rec = json.load(fh)
                if rec.get("outcome") and rec.get("outcome") != "INFORMATIONAL":
                    crt.append(rec)
            except (json.JSONDecodeError, OSError):
                pass

    # Spec 059: Risk overlay for this ticker
    risk_overlay = {}
    risk_data = _load_json(REPO_ROOT / "data" / "snapshots" / date / "catalyst_risk_overlay.json")
    if risk_data:
        for rm in risk_data.get("risk_matrix", []):
            if rm.get("ticker") == ticker.upper():
                risk_overlay["risk_matrix"] = rm
                break
        for al in risk_data.get("escalated_alerts", []):
            if al.get("ticker") == ticker.upper():
                risk_overlay["escalated_alert"] = al
                break

    # Spec 059: Surface anomaly for this ticker
    anomaly_data = _load_json(REPO_ROOT / "data" / "snapshots" / date / "surface_anomalies.json")
    if anomaly_data:
        for an in anomaly_data.get("anomalies", []):
            if an.get("ticker") == ticker.upper():
                risk_overlay["surface_anomaly"] = an
                break

    return {
        "ticker": ticker.upper(),
        "date": date,
        "ranking": ranking_row,
        "position": position,
        "options": options,
        "crt_resolutions": crt,
        "risk_overlay": risk_overlay,
    }


@app.get("/api/options_diagnostics/{date}")
async def api_options_diagnostics(date: str, top_n: int = 60):
    """Options fields extracted from rankings for the top-N names."""
    rankings = _load_rankings(date)
    if not rankings:
        return {"error": f"No rankings for {date}"}

    opts_fields = [
        "atm_iv_change_5d",
        "opt_rr_25d",
        "actual_implied_move_pctile",
        "iv_percentile_30d",
        "options_quality_composite",
    ]

    rows = []
    for ticker, r in rankings.items():
        ar = r.get("actionable_rank", "").strip()
        if not ar:
            continue
        try:
            rank = int(ar)
        except ValueError:
            continue
        if rank > top_n:
            continue
        row = {"ticker": ticker, "actionable_rank": rank}
        for field in opts_fields:
            row[field] = r.get(field, "")
        rows.append(row)

    rows.sort(key=lambda r: r["actionable_rank"])
    return rows


@app.get("/api/crt/resolutions")
async def api_crt_resolutions():
    """All CRT resolution records."""
    crt_dir = REPO_ROOT / "data" / "snapshots" / "resolutions"
    records = []
    if not crt_dir.exists():
        return records
    for month_dir in sorted(crt_dir.iterdir()):
        if not month_dir.is_dir() or not month_dir.name[:4].isdigit():
            continue
        for f in sorted(month_dir.glob("*.json")):
            try:
                with open(f, encoding="utf-8") as fh:
                    rec = json.load(fh)
                if rec.get("outcome") and rec.get("outcome") != "INFORMATIONAL":
                    records.append(rec)
            except (json.JSONDecodeError, OSError):
                pass
    return records


@app.get("/api/crt/calibration")
async def api_crt_calibration():
    """Latest CRT calibration summary."""
    path = REPO_ROOT / "data" / "snapshots" / "resolutions" / "calibration_summary.json"
    return _load_json(path) or {"error": "No calibration summary"}


@app.get("/api/tier_bucket_heatmap/{date}")
async def api_tier_bucket_heatmap(date: str):
    """Tier x bucket counts for heatmap."""
    rankings = _load_rankings(date)
    if not rankings:
        return {"error": f"No rankings for {date}"}
    from collections import Counter

    grid = Counter()
    for r in rankings.values():
        tier = r.get("tier_any", "")
        bucket = r.get("catalyst_bucket", "")
        if tier and bucket:
            grid[(tier, bucket)] += 1
    buckets = sorted(set(b for _, b in grid.keys()))
    tiers = ["A", "B", "C", "D"]
    return {
        "tiers": tiers,
        "buckets": buckets,
        "counts": {f"{t}|{b}": grid.get((t, b), 0) for t in tiers for b in buckets},
    }


@app.get("/api/shadow_performance")
async def api_shadow_performance():
    """Shadow portfolio performance timeseries."""
    return _load_shadow_performance()


@app.get("/api/bioshort/verdict")
async def api_bioshort_verdict():
    """Latest bioshort verdict."""
    path = REPO_ROOT / "output" / "hedge_report" / "BIOSHORT_VERDICT.json"
    data = _load_json(path) or {"error": "No bioshort verdict"}
    data["_freshness"] = _bioshort_freshness_meta()
    return data


@app.get("/api/bioshort/report")
async def api_bioshort_report():
    """Latest bioshort hedge report detail."""
    report_dir = REPO_ROOT / "output" / "hedge_report"
    if not report_dir.exists():
        return {"error": "No hedge reports", "_freshness": _bioshort_freshness_meta()}
    files = sorted(report_dir.glob("hedge_report_*.json"), reverse=True)
    if not files:
        return {"error": "No hedge report files", "_freshness": _bioshort_freshness_meta()}
    data = _load_json(files[0]) or {"error": "Failed to load report"}
    data["_freshness"] = _bioshort_freshness_meta()
    return data


@app.get("/api/bioshort/watch")
async def api_bioshort_watch():
    """Latest bioshort watch alerts."""
    data = _load_bioshort_watch() or {"error": "No bioshort watch"}
    data["_freshness"] = _bioshort_freshness_meta()
    return data


@app.get("/api/bioshort/archive")
async def api_bioshort_archive():
    """List of archived bioshort reports."""
    report_dir = REPO_ROOT / "output" / "hedge_report"
    if not report_dir.exists():
        return {"reports": [], "_freshness": _bioshort_freshness_meta()}
    files = sorted(report_dir.glob("hedge_report_*.json"), reverse=True)
    archive = []
    for f in files[:20]:
        data = _load_json(f)
        if data:
            archive.append(
                {
                    "date": data.get("as_of_date", f.stem.split("_")[-1]),
                    "file": f.name,
                }
            )
    return {"reports": archive, "_freshness": _bioshort_freshness_meta()}


@app.get("/api/options_quality/{date}")
async def api_options_quality(date: str):
    """Options quality manifest for a given date."""
    path = REPO_ROOT / "data" / "snapshots" / date / "options_quality_manifest.json"
    return _load_json(path) or {"error": f"No options quality manifest for {date}"}


@app.get("/api/construction_v2/performance")
async def api_construction_v2_performance():
    """Construction v2 shadow performance timeseries."""
    perf_path = REPO_ROOT / "artifacts" / "construction_v2" / "performance.csv"
    if not perf_path.exists():
        return {"error": "No construction v2 data"}
    rows = []
    with open(perf_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    # Add cumulative columns
    cum_ew30 = 0.0
    cum_regime = 0.0
    cum_xbi = 0.0
    for r in rows:
        cum_ew30 += float(r.get("ew30_pnl_pct", 0))
        cum_regime += float(r.get("regime_pnl_pct", 0))
        cum_xbi += float(r.get("xbi_pct", 0))
        r["cum_ew30"] = round(cum_ew30, 4)
        r["cum_regime"] = round(cum_regime, 4)
        r["cum_xbi"] = round(cum_xbi, 4)
        r["cum_ew30_excess"] = round(cum_ew30 - cum_xbi, 4)
        r["cum_regime_excess"] = round(cum_regime - cum_xbi, 4)
    return rows


@app.get("/api/event_premium_decomp/{date}")
async def api_event_premium_decomp(date: str):
    """Event premium decomposition for top-30 names."""
    path = REPO_ROOT / "data" / "snapshots" / date / "event_premium_decomp.json"
    return _load_json(path) or {"error": f"No event premium decomp for {date}"}


@app.get("/api/construction_v2/positions/{date}")
async def api_construction_v2_positions(date: str):
    """Construction v2 positions for a given date."""
    pos_path = REPO_ROOT / "artifacts" / "construction_v2" / "positions" / f"{date}.json"
    return _load_json(pos_path) or {"error": f"No v2 positions for {date}"}


@app.get("/api/aact/{ticker}")
async def api_aact_trials(ticker: str, limit: int = 50):
    """Trial records linked to a ticker from AACT warehouse."""
    # Find latest AACT snapshot
    aact_dir = REPO_ROOT / "data" / "aact" / "snapshots"
    if not aact_dir.exists():
        return {
            "ticker": ticker.upper(),
            "n_trials": 0,
            "status": "no_data",
            "message": "No AACT data. Run tools/fetch_aact_snapshot.py first.",
        }
    snapshot_dirs = sorted(
        (d for d in aact_dir.iterdir() if d.is_dir() and d.name[:4].isdigit()),
        reverse=True,
    )
    if not snapshot_dirs:
        return {"ticker": ticker.upper(), "n_trials": 0, "status": "no_data"}

    master_path = snapshot_dirs[0] / "trial_master.json"
    if not master_path.exists():
        return {"ticker": ticker.upper(), "n_trials": 0, "status": "no_data"}

    data = _load_json(master_path)
    if not data:
        return {"ticker": ticker.upper(), "n_trials": 0, "status": "load_error"}

    ticker_upper = ticker.upper()
    matched = [t for t in data.get("trials", []) if (t.get("mapped_ticker") or "").upper() == ticker_upper]

    # Sort: active/recruiting first, then by phase desc
    phase_order = {"Phase 3": 0, "Phase 2/3": 1, "Phase 2": 2, "Phase 1/2": 3, "Phase 1": 4}
    active_statuses = {"Recruiting", "Active, not recruiting", "Enrolling by invitation", "Not yet recruiting"}
    matched.sort(
        key=lambda t: (
            0 if t.get("overall_status") in active_statuses else 1,
            phase_order.get(t.get("phase", ""), 9),
        )
    )

    # Summary stats
    status_counts: Dict[str, int] = {}
    phase_counts: Dict[str, int] = {}
    for t in matched:
        s = t.get("overall_status", "Unknown")
        p = t.get("phase", "Unknown")
        status_counts[s] = status_counts.get(s, 0) + 1
        phase_counts[p] = phase_counts.get(p, 0) + 1

    return {
        "ticker": ticker_upper,
        "snapshot_date": data.get("snapshot_date"),
        "n_trials": len(matched),
        "status_distribution": dict(sorted(status_counts.items(), key=lambda x: -x[1])),
        "phase_distribution": dict(sorted(phase_counts.items(), key=lambda x: -x[1])),
        "trials": [
            {
                "nct_id": t.get("nct_id"),
                "brief_title": t.get("brief_title"),
                "phase": t.get("phase"),
                "overall_status": t.get("overall_status"),
                "enrollment": t.get("enrollment"),
                "primary_completion_date": t.get("primary_completion_date"),
                "start_date": t.get("start_date"),
                "has_results": t.get("has_results", False),
                "condition_names": t.get("condition_names", [])[:3],
                "intervention_names": t.get("intervention_names", [])[:3],
                "mapping_confidence": t.get("mapping_confidence"),
            }
            for t in matched[:limit]
        ],
    }


@app.get("/api/aact/health")
async def api_aact_health():
    """Latest AACT ingest health report."""
    aact_dir = REPO_ROOT / "data" / "aact" / "snapshots"
    if not aact_dir.exists():
        return {"error": "No AACT data"}
    snapshot_dirs = sorted(
        (d for d in aact_dir.iterdir() if d.is_dir() and d.name[:4].isdigit()),
        reverse=True,
    )
    if not snapshot_dirs:
        return {"error": "No AACT snapshots"}
    health_path = snapshot_dirs[0] / "aact_health.json"
    return _load_json(health_path) or {"error": "No health report"}


@app.get("/api/purple_book/{ticker}")
async def api_purple_book(ticker: str, date: str = ""):
    """Biologic competition context from FDA Purple Book."""
    if not date:
        dates = _available_snapshot_dates()
        date = dates[0] if dates else ""

    pb_path = REPO_ROOT / "production_data" / "purple_book.json"
    pb_data = _load_json(pb_path)
    if not pb_data or not pb_data.get("products"):
        return {
            "ticker": ticker.upper(),
            "is_biologic_company": False,
            "status": "no_data",
            "message": "No Purple Book data loaded. Run scripts/ingest_purple_book.py first.",
        }

    from common.purple_book_features import get_biologic_competition

    return get_biologic_competition(ticker=ticker.upper(), as_of_date=date, pb_data=pb_data)


@app.get("/api/deal_comps/{ticker}")
async def api_deal_comps(ticker: str, date: str = ""):
    """Deal comp context from DealForma for a single ticker."""
    if not date:
        dates = _available_snapshot_dates()
        date = dates[0] if dates else ""

    comps_path = REPO_ROOT / "production_data" / "dealforma_comps.json"
    comps_data = _load_json(comps_path)
    if not comps_data or not comps_data.get("deals"):
        return {
            "ticker": ticker.upper(),
            "n_comps": 0,
            "status": "no_data",
            "message": "No DealForma data loaded. Run scripts/ingest_dealforma.py first.",
        }

    # Get ticker's TA and stage from rankings
    rankings = _load_rankings(date)
    row = rankings.get(ticker.upper(), {})
    ta = row.get("therapeutic_area", "")
    stage = row.get("lead_program_phase", "")
    modality = row.get("modality", "")

    from common.dealforma_features import get_deal_comps

    return get_deal_comps(
        ticker=ticker.upper(),
        therapeutic_area=ta or None,
        stage=stage or None,
        modality=modality or None,
        as_of_date=date,
        comps_data=comps_data,
    )


@app.get("/api/herald/health")
async def api_herald_health():
    """Latest Herald health artifact."""
    health_dir = REPO_ROOT / "data" / "press_releases"
    if not health_dir.exists():
        return {"error": "No Herald data"}
    files = sorted(health_dir.glob("health_*.json"), reverse=True)
    if not files:
        return {"error": "No health reports"}
    return _load_json(files[0]) or {"error": "Failed to load health"}


@app.get("/api/herald/releases/{date}")
async def api_herald_releases(date: str, limit: int = 50):
    """Company press releases for a given date."""
    path = REPO_ROOT / "data" / "press_releases" / f"releases_{date}.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records[:limit]


@app.get("/api/herald/classified/{date}")
async def api_herald_classified(date: str, limit: int = 50):
    """Classified press releases for a given date."""
    path = REPO_ROOT / "data" / "press_releases" / "classified" / f"classified_{date}.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records[:limit]


# --- Coinvest shadow ---


@app.get("/api/coinvest_shadow/history")
async def api_coinvest_shadow_history():
    """Coinvest anchor shadow history (CSV ledger)."""
    path = REPO_ROOT / "artifacts" / "coinvest_shadow" / "history.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


@app.get("/api/coinvest_shadow/latest")
async def api_coinvest_shadow_latest():
    """Latest coinvest shadow snapshot."""
    shadow_dir = REPO_ROOT / "artifacts" / "coinvest_shadow"
    if not shadow_dir.exists():
        return {"error": "No coinvest shadow data"}
    files = sorted(shadow_dir.glob("20*.json"), reverse=True)
    if not files:
        return {"error": "No coinvest shadow snapshots"}
    return _load_json(files[0]) or {"error": "Failed to load snapshot"}


# --- Post-promotion monitor ---


@app.get("/api/post_promotion_monitor")
async def api_post_promotion_monitor():
    """Post-promotion monitor history (all daily snapshots)."""
    pm_dir = REPO_ROOT / "artifacts" / "post_promotion_monitor"
    if not pm_dir.exists():
        return []
    rows = []
    for f in sorted(pm_dir.glob("*_monitor.json")):
        data = _load_json(f)
        if data:
            rows.append(data)
    return rows


@app.get("/api/post_promotion_monitor/latest")
async def api_post_promotion_monitor_latest():
    """Latest post-promotion monitor snapshot."""
    pm_dir = REPO_ROOT / "artifacts" / "post_promotion_monitor"
    if not pm_dir.exists():
        return {"error": "No post-promotion monitor data"}
    files = sorted(pm_dir.glob("*_monitor.json"), reverse=True)
    if not files:
        return {"error": "No monitor snapshots"}
    return _load_json(files[0]) or {"error": "Failed to load"}


# --- Regime shadow ---


@app.get("/api/regime_shadow/history")
async def api_regime_shadow_history():
    """Regime shadow history (all daily snapshots)."""
    rs_dir = REPO_ROOT / "artifacts" / "regime_shadow"
    if not rs_dir.exists():
        return []
    rows = []
    for f in sorted(rs_dir.glob("20*.json")):
        data = _load_json(f)
        if data:
            rows.append(
                {
                    "date": data.get("as_of_date", f.stem),
                    "simple_regime": (data.get("simple_classifier") or {}).get("regime"),
                    "rich_regime": (data.get("rich_classifier") or {}).get("regime"),
                    "rich_confidence": (data.get("rich_classifier") or {}).get("confidence"),
                    "agreement": data.get("agreement"),
                    "recommendation": (data.get("switching_policy") or {}).get("recommendation"),
                }
            )
    return rows


@app.get("/api/production_monitor/latest")
async def api_production_monitor_latest():
    """Latest production health monitor."""
    pm_dir = REPO_ROOT / "artifacts" / "production_monitor"
    if not pm_dir.exists():
        return {"error": "No production monitor data"}
    files = sorted(pm_dir.glob("*_monitor.json"), reverse=True)
    if not files:
        return {"error": "No production monitor snapshots"}
    return _load_json(files[0]) or {"error": "Failed to load"}


@app.get("/api/regime_shadow/latest")
async def api_regime_shadow_latest():
    """Latest regime shadow snapshot."""
    rs_dir = REPO_ROOT / "artifacts" / "regime_shadow"
    if not rs_dir.exists():
        return {"error": "No regime shadow data"}
    files = sorted(rs_dir.glob("20*.json"), reverse=True)
    if not files:
        return {"error": "No regime shadow snapshots"}
    return _load_json(files[0]) or {"error": "Failed to load"}


# ── Timing Hazard & Event Quality Endpoints ──────────────────────────


@app.get("/api/timing_hazard/latest")
async def api_timing_hazard_latest():
    """Latest timing hazard overlay."""
    th_dir = REPO_ROOT / "artifacts" / "timing_hazard"
    if not th_dir.exists():
        return {"error": "No timing hazard data"}
    files = sorted(th_dir.glob("timing_hazard_*.json"), reverse=True)
    if not files:
        return {"error": "No timing hazard snapshots"}
    return _load_json(files[0]) or {"error": "Failed to load"}


@app.get("/api/timing_hazard/review")
async def api_timing_hazard_review():
    """Latest timing hazard calibration review results."""
    return _load_timing_hazard_review() or {"error": "No review data — run timing_hazard_review.py"}


@app.get("/api/timing_hazard/calibration")
async def api_timing_hazard_calibration():
    """Timing hazard calibration ledger (JSONL → list)."""
    path = REPO_ROOT / "artifacts" / "timing_hazard" / "calibration_ledger.jsonl"
    if not path.exists():
        return []
    return _load_jsonl(path)


@app.get("/api/timing_hazard/calibration_dashboard")
async def api_timing_hazard_calibration_dashboard():
    """Extended calibration dashboard with per-horizon curves and source breakdown."""
    path = REPO_ROOT / "artifacts" / "timing_hazard" / "calibration_dashboard.json"
    if not path.exists():
        return {"error": "No calibration dashboard data"}
    return _load_json(path) or {"error": "Failed to load"}


@app.get("/api/event_quality_shadow/latest")
async def api_event_quality_shadow_latest():
    """Latest event quality shadow sizing comparison."""
    eq_dir = REPO_ROOT / "artifacts" / "event_quality_shadow"
    if not eq_dir.exists():
        return {"error": "No event quality shadow data"}
    files = sorted(eq_dir.glob("event_quality_shadow_*.json"), reverse=True)
    if not files:
        return {"error": "No event quality shadow snapshots"}
    return _load_json(files[0]) or {"error": "Failed to load"}


@app.get("/api/event_quality/confusion")
async def api_event_quality_confusion():
    """Event quality confusion dashboard (P/R/F1, confusion matrix, drift)."""
    path = REPO_ROOT / "artifacts" / "event_quality" / "confusion_dashboard.json"
    if not path.exists():
        return {"error": "No confusion dashboard data"}
    return _load_json(path) or {"error": "Failed to load"}


@app.get("/api/event_quality/outlier_queue")
async def api_event_quality_outlier_queue():
    """Outlier review queue — misclassification cases needing human review."""
    path = REPO_ROOT / "artifacts" / "event_quality" / "outlier_review_queue.json"
    if not path.exists():
        return {"error": "No outlier review queue data"}
    return _load_json(path) or {"error": "Failed to load"}


@app.get("/api/event_quality/operator_priority")
async def api_event_quality_operator_priority():
    """Operator priority queue — positions needing attention by urgency."""
    path = REPO_ROOT / "artifacts" / "event_quality_shadow"
    if not path.is_dir():
        return {"error": "No event quality shadow data"}
    files = sorted(path.glob("event_quality_shadow_*.json"), reverse=True)
    if not files:
        return {"error": "No shadow artifacts found"}
    data = _load_json(files[0])
    if not data:
        return {"error": "Failed to load"}
    return {"reviews": data.get("reviews", []), "date": data.get("snapshot_date")}


@app.get("/api/review/packets")
async def api_review_packets():
    """Latest review packet (unified timing + event quality)."""
    review_dir = REPO_ROOT / "artifacts" / "review"
    if not review_dir.exists():
        return {"error": "No review packet data"}
    files = sorted(review_dir.glob("*_review_packet.json"), reverse=True)
    if not files:
        return {"error": "No review packets found"}
    return _load_json(files[0]) or {"error": "Failed to load"}


@app.get("/api/review/priority")
async def api_review_priority():
    """Latest review priority queue."""
    review_dir = REPO_ROOT / "artifacts" / "review"
    if not review_dir.exists():
        return {"error": "No review priority data"}
    files = sorted(review_dir.glob("review_priority_*.json"), reverse=True)
    if not files:
        return {"error": "No review priority data"}
    return _load_json(files[0]) or {"error": "Failed to load"}


@app.get("/api/risk_monitor/latest")
async def api_risk_monitor_latest():
    """Latest risk monitor report (v2: includes vol/corr metrics)."""
    rm_dir = REPO_ROOT / "artifacts" / "risk_monitor"
    if not rm_dir.exists():
        return {"error": "No risk monitor data"}
    files = sorted(rm_dir.glob("*_risk.json"), reverse=True)
    if not files:
        return {"error": "No risk monitor snapshots"}
    return _load_json(files[0]) or {"error": "Failed to load"}


@app.get("/api/risk_monitor/history")
async def api_risk_monitor_history():
    """Risk monitor history (last 30 reports)."""
    rm_dir = REPO_ROOT / "artifacts" / "risk_monitor"
    if not rm_dir.exists():
        return []
    files = sorted(rm_dir.glob("*_risk.json"), reverse=True)[:30]
    result = []
    for f in files:
        data = _load_json(f)
        if data:
            result.append(data)
    return result


@app.get("/api/rebalance_plan/latest")
async def api_rebalance_plan_latest():
    """Latest rebalance plan."""
    rp_dir = REPO_ROOT / "artifacts" / "rebalance_plan"
    if not rp_dir.exists():
        return {"error": "No rebalance plan data"}
    files = sorted(rp_dir.glob("*.json"), reverse=True)
    if not files:
        return {"error": "No rebalance plans"}
    return _load_json(files[0]) or {"error": "Failed to load"}


@app.get("/api/options_qc/{date}")
async def api_options_qc(date: str):
    """Options QC summary with source-separated quality metrics."""
    snap_dir = REPO_ROOT / "data" / "snapshots"
    path = snap_dir / date / "options_diagnostics_summary.json"
    if not path.exists():
        # Try latest snapshot
        candidates = sorted(snap_dir.glob("*/options_diagnostics_summary.json"), reverse=True)
        if not candidates:
            return {"error": "No options diagnostics summary found"}
        path = candidates[0]
    return _load_json(path) or {"error": "Failed to load"}


@app.get("/api/herald_precision/latest")
async def api_herald_precision_latest():
    """Latest Herald precision metrics."""
    hp_dir = REPO_ROOT / "artifacts" / "herald_precision"
    if not hp_dir.exists():
        return {"error": "No Herald precision data"}
    files = sorted(hp_dir.glob("metrics_*.json"), reverse=True)
    if not files:
        return {"error": "No Herald precision snapshots"}
    return _load_json(files[0]) or {"error": "Failed to load"}


# --- Spec 059: Options Event Overlay ---


@app.get("/api/catalyst_risk_matrix/{date}")
async def api_catalyst_risk_matrix(date: str):
    """Catalyst proximity risk matrix for book names near catalysts."""
    snap = REPO_ROOT / "data" / "snapshots" / date / "catalyst_risk_overlay.json"
    data = _load_json(snap)
    if not data:
        return {"error": f"No risk overlay for {date}", "risk_matrix": [], "escalated_alerts": []}
    return data


@app.get("/api/surface_anomalies/{date}")
async def api_surface_anomalies(date: str):
    """Surface anomaly flags from cross-sectional options analysis."""
    snap = REPO_ROOT / "data" / "snapshots" / date / "surface_anomalies.json"
    data = _load_json(snap)
    if not data:
        return {"error": f"No surface anomalies for {date}", "anomalies": []}
    return data


@app.get("/api/options_forward_log/{date}")
async def api_options_forward_log(date: str):
    """Options forward log entries for calibration growth tracking."""
    snap = REPO_ROOT / "data" / "snapshots" / date / "options_forward_log.json"
    data = _load_json(snap)
    if not data:
        return {"error": f"No forward log for {date}", "entries": []}
    return data


# --- Spec 060: Event EV Scoring ---


@app.get("/api/event_ev/leaderboard/{date}")
async def api_event_ev_leaderboard(date: str):
    """Event EV leaderboard — top catalysts by downside-adjusted EV."""
    ev_dir = REPO_ROOT / "artifacts" / "event_ev"
    lb_path = ev_dir / f"{date}_ev_leaderboard.json"
    if lb_path.exists():
        data = _load_json(lb_path)
        if data is not None:
            return {"as_of_date": date, "leaderboard": data}
    # Try scores file as fallback
    scores_path = ev_dir / f"{date}_event_ev_scores.json"
    data = _load_json(scores_path)
    if data:
        return {"as_of_date": date, "leaderboard": data.get("leaderboard", [])}
    return {"error": f"No EV leaderboard for {date}", "leaderboard": []}


@app.get("/api/event_ev/detail/{ticker}/{date}")
async def api_event_ev_detail(ticker: str, date: str):
    """Full EventEV breakdown for a single ticker."""
    ev_dir = REPO_ROOT / "artifacts" / "event_ev"
    full_path = ev_dir / f"{date}_event_ev_full.json"
    data = _load_json(full_path)
    if not data:
        return {"error": f"No EV data for {date}"}
    events = data.get("events", [])
    matches = [e for e in events if e.get("node", {}).get("ticker", "").upper() == ticker.upper()]
    if not matches:
        return {"error": f"No EV data for {ticker} on {date}"}
    return {"ticker": ticker.upper(), "date": date, "events": matches}


@app.get("/api/event_ev/history")
async def api_event_ev_history(limit: int = 30):
    """Recent EV leaderboard snapshots for trend tracking."""
    ev_dir = REPO_ROOT / "artifacts" / "event_ev"
    if not ev_dir.exists():
        return {"error": "No event EV data", "snapshots": []}
    files = sorted(ev_dir.glob("*_event_ev_scores.json"), reverse=True)[:limit]
    snapshots = []
    for f in files:
        data = _load_json(f)
        if data:
            snapshots.append(
                {
                    "as_of_date": data.get("as_of_date"),
                    "n_total": data.get("n_total", 0),
                    "n_actionable": data.get("n_actionable", 0),
                    "top_ev": data.get("stats", {}).get("top_ev"),
                    "mean_ev": data.get("stats", {}).get("mean_ev"),
                }
            )
    return {"snapshots": snapshots}


# ============================================================================
# Expression Overlay (Spec 062)
# ============================================================================


@app.get("/api/expression_overlay/{date}")
async def api_expression_overlay(date: str):
    """Expression overlay summary + tradeable recommendations for a snapshot date."""
    snap_dir = REPO_ROOT / "data" / "snapshots" / date
    summary = _load_json(snap_dir / "expression_overlay_summary.json")
    recs = _load_json(snap_dir / "expression_recommendations.json")

    if not summary and not recs:
        return {"error": f"No expression overlay data for {date}", "as_of_date": date}

    return {
        "as_of_date": date,
        "summary": summary or {},
        "recommendations": recs if isinstance(recs, list) else (recs or {}).get("recommendations", []),
    }


@app.get("/api/expression_overlay/attribution/metrics")
async def api_expression_attribution_metrics():
    """Attribution metrics from resolved records + kill-switch status."""
    attr_path = REPO_ROOT / "data" / "expression_attribution_log.jsonl"
    records = _load_jsonl(attr_path)

    if not records:
        return {
            "n_total": 0,
            "n_resolved": 0,
            "kill_switch": {
                "overlay_enabled": True,
                "sizing_enabled": True,
                "disabled_types": [],
                "triggered_rules": [],
                "evaluation_status": "no_data",
            },
            "metrics": {},
        }

    resolved = [r for r in records if r.get("attribution_status") == "resolved"]
    pending = [r for r in records if r.get("attribution_status") == "pending"]

    # Compute metrics inline (avoid importing expression_attribution to keep dashboard lightweight)
    pnls = [r["pnl_estimate"] for r in resolved if r.get("pnl_estimate") is not None]
    wins = [p for p in pnls if p > 0]

    aggregate = {
        "n_resolved": len(resolved),
        "n_pending": len(pending),
        "n_with_pnl": len(pnls),
        "win_rate": round(len(wins) / len(pnls), 4) if pnls else None,
        "mean_pnl": round(sum(pnls) / len(pnls), 4) if pnls else None,
    }

    # By type
    by_type = {}
    for r in resolved:
        mt = r.get("mispricing_type", "UNKNOWN")
        by_type.setdefault(mt, {"pnls": [], "n": 0})
        by_type[mt]["n"] += 1
        if r.get("pnl_estimate") is not None:
            by_type[mt]["pnls"].append(r["pnl_estimate"])
    for mt in by_type:
        pl = by_type[mt]["pnls"]
        w = [p for p in pl if p > 0]
        by_type[mt] = {
            "n": by_type[mt]["n"],
            "win_rate": round(len(w) / len(pl), 4) if pl else None,
            "mean_pnl": round(sum(pl) / len(pl), 4) if pl else None,
        }

    # Kill-switch evaluation (simplified — full version in expression_attribution.py)
    ks = {
        "overlay_enabled": True,
        "sizing_enabled": True,
        "disabled_types": [],
        "triggered_rules": [],
        "evaluation_status": "insufficient_data" if len(resolved) < 20 else "evaluated",
    }
    if len(resolved) >= 20 and pnls:
        wr = len(wins) / len(pnls) if pnls else 1.0
        if wr < 0.40:
            ks["overlay_enabled"] = False
            ks["triggered_rules"].append(f"aggregate_win_rate={wr:.2%}")
        for mt, data in by_type.items():
            if data.get("n", 0) >= 5 and data.get("win_rate") is not None:
                if data["win_rate"] < 0.30:
                    ks["disabled_types"].append(mt)
                    ks["triggered_rules"].append(f"{mt}_win_rate={data['win_rate']:.2%}")

    return {
        "n_total": len(records),
        "aggregate": aggregate,
        "by_type": by_type,
        "kill_switch": ks,
    }


@app.get("/api/expression_overlay/decisions/{date}")
async def api_expression_decisions(date: str):
    """Decision log entries for a specific date — all evaluated names."""
    dec_path = REPO_ROOT / "data" / "expression_decision_log.jsonl"
    all_records = _load_jsonl(dec_path)
    day_records = [r for r in all_records if r.get("as_of_date") == date]
    if not day_records:
        return {"as_of_date": date, "decisions": [], "n": 0}

    # Summary counts
    counts = Counter(r.get("decision", "unknown") for r in day_records)
    return {
        "as_of_date": date,
        "n": len(day_records),
        "counts": dict(counts),
        "decisions": day_records,
    }


@app.get("/api/expression_overlay/forward_panel")
async def api_expression_forward_panel():
    """Forward evaluation panel — realized performance of tradeable recommendations."""
    panel_path = REPO_ROOT / "output" / "expression_forward_panel" / "forward_panel.json"
    data = _load_json(panel_path)
    if not data:
        return {"status": "not_computed", "message": "Run expression_forward_panel.py first"}
    return data


@app.get("/api/expression_overlay/calibration/{date}")
async def api_expression_calibration(date: str):
    """Calibration diagnostics from expression overlay summary."""
    snap_dir = REPO_ROOT / "data" / "snapshots" / date
    summary = _load_json(snap_dir / "expression_overlay_summary.json")
    if not summary:
        return {"error": f"No overlay data for {date}"}
    return {
        "as_of_date": date,
        "calibration": summary.get("calibration", {}),
        "counts_by_mispricing_type": summary.get("counts_by_mispricing_type", {}),
        "counts_by_overlay_class": summary.get("counts_by_overlay_class", {}),
    }


@app.get("/expression", response_class=HTMLResponse)
async def expression_dashboard(request: Request):
    """Expression overlay dashboard page."""
    return templates.TemplateResponse("expression.html", {"request": request})


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Biotech Screener Dashboard")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    # Dashboard web-server entrypoint (argparse --host/--port), not an MCP tool;
    # accepted per explicit operator review (CI_RED_2026, #521).
    uvicorn.run(app, host=args.host, port=args.port)  # nosemgrep: sc-mcp-opens-socket
