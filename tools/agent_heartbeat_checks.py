#!/usr/bin/env python3
"""Lightweight heartbeat checks for Tier 2 agents.

Replaces 6 LLM-based OpenClaw agents with cheap file/JSON checks.
Anomalies are written to artifacts/heartbeat/ by default (deterministic).
Optional LLM escalation: set HEARTBEAT_LLM_ESCALATE=1 to invoke run_agent_direct.

Usage:
    python tools/agent_heartbeat_checks.py                    # run all checks
    python tools/agent_heartbeat_checks.py --agent qa         # run one agent
    python tools/agent_heartbeat_checks.py --dry-run          # print, don't escalate
    python tools/agent_heartbeat_checks.py --date 2026-04-02  # check specific date
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.artifact_freshness import (  # noqa: E402
    age_days,
    format_stale_source,
    newest_artifact_freshness,
)
from tools.ic_health_memory_hygiene import MemoryHygieneChecker  # noqa: E402
from tools.skills_logger_v2 import log_skill  # noqa: E402

SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshots"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
LOGS_DIR = REPO_ROOT / "logs"
REGISTRY_PATH = REPO_ROOT / "agents" / "AGENT_REGISTRY.json"
TERMINAL_UNSUPERVISED_AGENTS = {"ops_supervisor"}
HERMES_JOB_PREFIX = "hermes-"
CLOUD_ARTIFACT_STALENESS_PREFIXES = ("MISSING:", "NO_ARTIFACTS:", "STALE", "STALE_")

# Date-bounded carried-alert muffle for ic_health_monitor.
# When a signal in this list is at ALERT/CRITICAL AND today is on or before
# `expires_after`, the heartbeat downgrades the anomaly to WARN with a
# [CARRIED] tag and does NOT trigger the FAIL path. Reduces repeated LLM
# escalation when the cause is a known/expected condition with a documented
# self-heal date. Hard ALERT path is preserved for any signal NOT in this
# list (P1 #1, Spec-tracked: see audit memo + supervisor.py EXCEPTIONS).
# Add date-bounded entries only while a known/expected alert is active; remove
# after expires_after (inst_delta_z muffle removed 2026-05-30 post 13F refresh).
IC_HEALTH_CARRIED_ALERTS: dict[str, dict[str, str]] = {}


STALENESS_DAYS_BY_CADENCE = {
    "daily_after_production": 2,
    "daily_premarket": 2,
    "intraday": 1,
    "weekly": 10,
    "on_demand": None,
    "unknown": None,
}

# Stale memory detection: if memory/ mtime is >7 days older than artifact mtime,
# flag for investigation (may indicate code bug or memory write failure).
STALE_MEMORY_THRESHOLD_DAYS = 7


def is_cloud_agent_environment() -> bool:
    """Return true when this checkout is running in Cursor Cloud, not operator WSL."""
    if os.environ.get("CURSOR_CLOUD_AGENT") or os.environ.get("CURSOR_AGENT"):
        return True
    return Path("/tmp/cursor").exists()


# ── Result types ──────────────────────────────────────────────


class CheckResult:
    def __init__(self, agent: str, status: str, detail: str = "", anomalies: list = None):
        self.agent = agent
        self.status = status  # OK, WARN, FAIL, STALE, SKIP
        self.detail = detail
        self.anomalies = anomalies or []

    @property
    def needs_llm(self):
        if self.status not in ("WARN", "FAIL"):
            return False
        if not self.anomalies:
            return False
        # P1 #1: if every anomaly is [CARRIED]-tagged (carried-alert muffle),
        # do NOT escalate to LLM. Carried alerts are known/expected conditions
        # with documented self-heal dates — fresh LLM narrative adds noise
        # without changing operational action. The status (WARN) and the
        # [CARRIED] tag remain visible in the heartbeat log/receipt for audit.
        if all(a.startswith("[CARRIED]") for a in self.anomalies):
            return False
        return True

    def __repr__(self):
        sym = {"OK": "✓", "WARN": "⚠", "FAIL": "✗", "STALE": "◌", "SKIP": "–"}
        s = f"  {sym.get(self.status, '?')} {self.agent}: {self.status}"
        if self.detail:
            s += f" — {self.detail}"
        if self.anomalies:
            for a in self.anomalies:
                s += f"\n      → {a}"
        return s


def as_of_date(dt: date) -> str:
    return dt.strftime("%Y-%m-%d")


# ── QA Agent ──────────────────────────────────────────────────


def check_qa(dt: date) -> CheckResult:
    """Validate today's snapshot exists and is structurally sound."""
    ds = as_of_date(dt)
    snap = SNAPSHOT_DIR / ds
    anomalies = []

    if not snap.is_dir():
        return CheckResult("qa", "STALE", f"No snapshot for {ds}")

    # Check critical files
    for fname in ["rankings.csv", "metadata.json"]:
        if not (snap / fname).exists():
            anomalies.append(f"OUTPUT_MISSING: {fname}")

    # Validate metadata date
    meta_path = snap / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("as_of_date") != ds:
                anomalies.append(f"DATE_MISMATCH: metadata says {meta.get('as_of_date')}, expected {ds}")
        except (json.JSONDecodeError, KeyError) as e:
            anomalies.append(f"METADATA_CORRUPT: {e}")

    # Check phase2 health
    p2 = snap / "phase2_health.json"
    if p2.exists():
        try:
            health = json.loads(p2.read_text())
            status = health.get("status", "UNKNOWN")
            if status == "FAIL":
                anomalies.append(f"PHASE2_FAIL: {health.get('message', 'no detail')}")
        except json.JSONDecodeError:
            anomalies.append("PHASE2_CORRUPT: cannot parse phase2_health.json")

    # Check production log for tracebacks
    log = LOGS_DIR / f"daily_production_{ds}.log"
    if log.exists():
        text = log.read_text()
        if "Traceback" in text and "snapshot promoted" not in text.lower():
            anomalies.append("PIPELINE_CRASH_SUSPECTED: traceback in production log without promotion")

    if anomalies:
        return CheckResult("qa", "FAIL", f"{len(anomalies)} issue(s)", anomalies)
    return CheckResult("qa", "OK", f"Snapshot {ds} valid")


# ── IC Health Monitor ─────────────────────────────────────────


def check_ic_health(dt: date) -> CheckResult:
    """Check IC dashboard exists and signal health."""
    ds = as_of_date(dt)
    dash_path = ARTIFACTS_DIR / "ic_dashboard" / f"{ds}_dashboard.json"
    anomalies = []

    if not dash_path.exists():
        latest, _, _ = newest_artifact_freshness(REPO_ROOT, ["artifacts/ic_dashboard/"])
        detail = f"No dashboard for {ds}"
        if latest is not None:
            detail += f"; latest={latest} ({age_days(dt, latest)}d ago)"
            if age_days(dt, latest) > STALENESS_DAYS_BY_CADENCE["daily_after_production"]:
                anomalies.append(format_stale_source(latest, dt, STALENESS_DAYS_BY_CADENCE["daily_after_production"]))
        return CheckResult("ic_health_monitor", "STALE", detail, anomalies)

    try:
        dash = json.loads(dash_path.read_text())
    except json.JSONDecodeError:
        return CheckResult("ic_health_monitor", "FAIL", "Dashboard corrupt", ["CORRUPT_DASHBOARD"])

    attention = dash.get("attention", "UNKNOWN")
    signals = dash.get("signals", {})

    today_iso = ds  # YYYY-MM-DD
    has_unmuffled_alert = False

    for sig_name, sig_data in signals.items():
        health = sig_data.get("health", "UNKNOWN") if isinstance(sig_data, dict) else "UNKNOWN"
        if health in ("ALERT", "CRITICAL"):
            muffle = IC_HEALTH_CARRIED_ALERTS.get(sig_name)
            if muffle and today_iso <= muffle["expires_after"]:
                # Known-expected alert with documented self-heal date — downgrade
                # to WARN with [CARRIED] tag. Does NOT escalate to FAIL.
                anomalies.append(
                    f"[CARRIED] SIGNAL_{health}: {sig_name} "
                    f"(expected, expires {muffle['expires_after']}; {muffle['reason']})"
                )
            else:
                anomalies.append(f"SIGNAL_{health}: {sig_name}")
                has_unmuffled_alert = True
        elif health == "WARN":
            anomalies.append(f"SIGNAL_WARN: {sig_name}")

    # Check history for trend decay (3+ consecutive IC drops)
    hist_path = ARTIFACTS_DIR / "ic_dashboard" / "history.jsonl"
    if hist_path.exists():
        lines = hist_path.read_text().strip().split("\n")
        if len(lines) >= 5:
            try:
                recent = [json.loads(line) for line in lines[-5:]]
                for sig_name in signals:
                    ics = [
                        r.get("signals", {}).get(sig_name, {}).get("ic")
                        for r in recent
                        if isinstance(r.get("signals", {}).get(sig_name), dict)
                    ]
                    ics = [x for x in ics if x is not None]
                    if len(ics) >= 4:
                        drops = sum(1 for i in range(1, len(ics)) if ics[i] < ics[i - 1])
                        if drops >= 3:
                            anomalies.append(f"TREND_DECAY: {sig_name} ({drops} consecutive IC drops)")
            except json.JSONDecodeError:
                pass

    # FAIL only on unmuffled ALERT/CRITICAL. Carried-alert anomalies (tagged
    # [CARRIED]) escalate to WARN at most. Hard ALERT path preserved for any
    # signal NOT in IC_HEALTH_CARRIED_ALERTS.
    if has_unmuffled_alert:
        return CheckResult("ic_health_monitor", "FAIL", f"attention={attention}", anomalies)
    if anomalies:
        return CheckResult("ic_health_monitor", "WARN", f"attention={attention}", anomalies)
    return CheckResult("ic_health_monitor", "OK", f"attention={attention}")


def check_ic_memory_hygiene(dt: date) -> CheckResult:
    """Check IC health memory consistency — stale memory, missing artifacts, state drift.

    Phase 1 Priority 5: Observability logging for state consistency.
    Advisory-only, non-blocking; logs mismatches for operator investigation.
    """
    checker = MemoryHygieneChecker()
    report = checker.check_as_of_date(dt)
    checker.log_findings(report)

    anomalies = []

    # Issue-level findings (missing/corrupt/drift)
    for issue in report.get("issues", []):
        anomalies.append(f"ISSUE: {issue['type']} ({issue['severity']}) — {issue['description']}")

    # Warning-level findings (stale, undocumented, inconsistent)
    for warning in report.get("warnings", []):
        anomalies.append(f"WARN: {warning['type']} — {warning['description']}")

    summary = report.get("summary", {})
    status_text = summary.get("status", "UNKNOWN")

    if anomalies:
        return CheckResult(
            "ic_memory_hygiene",
            "WARN" if status_text == "HEALTHY" else "FAIL",
            f"{summary.get('issue_count', 0)} issue(s), {summary.get('warning_count', 0)} warning(s)",
            anomalies,
        )
    return CheckResult("ic_memory_hygiene", "OK", "Memory and artifacts consistent")


# ── Fleet Steward ─────────────────────────────────────────────


def check_fleet_steward(dt: date) -> CheckResult:
    """Check fleet-wide artifact freshness."""
    ds = as_of_date(dt)
    anomalies = []

    # Production snapshot
    if not (SNAPSHOT_DIR / ds / "rankings.csv").exists():
        anomalies.append("MISSING: today's snapshot")

    # Artifact freshness checks
    checks = {
        "ops_digest": ARTIFACTS_DIR / "ops_digest" / f"{ds}_digest.json",
        "ic_dashboard": ARTIFACTS_DIR / "ic_dashboard" / f"{ds}_dashboard.json",
        "shadow_monitor": ARTIFACTS_DIR / "shadow_monitor" / f"{ds}_monitor.json",
    }
    for name, path in checks.items():
        if not path.exists():
            # Only flag if past expected production time (5:30 PM ET)
            if datetime.now().hour >= 18:
                anomalies.append(f"STALE: {name} missing for {ds}")

    # Earnings ICS freshness (< 2 days old)
    ics_path = ARTIFACTS_DIR / "earnings_sync" / "biotech_earnings.ics"
    if ics_path.exists():
        age_hours = (datetime.now().timestamp() - ics_path.stat().st_mtime) / 3600
        if age_hours > 48:
            anomalies.append(f"STALE: earnings ICS {age_hours:.0f}h old")
    else:
        anomalies.append("MISSING: earnings ICS")

    # CRT join table freshness (< 3 days)
    crt_join = REPO_ROOT / "output" / "catalyst_ev" / "crt_options_join.json"
    if crt_join.exists():
        age_hours = (datetime.now().timestamp() - crt_join.stat().st_mtime) / 3600
        if age_hours > 72:
            anomalies.append(f"STALE: CRT join table {age_hours:.0f}h old")

    if anomalies:
        return CheckResult(
            "fleet_steward", "WARN" if len(anomalies) <= 2 else "FAIL", f"{len(anomalies)} issue(s)", anomalies
        )
    return CheckResult("fleet_steward", "OK", "All artifacts fresh")


# ── Calibration ───────────────────────────────────────────────


def check_calibration(dt: date) -> CheckResult:
    """Check for new candidate rulesets needing review."""
    rulesets_dir = REPO_ROOT / "production_data" / "decision_rulesets"
    anomalies = []

    if not rulesets_dir.is_dir():
        return CheckResult("calibration", "OK", "No rulesets directory")

    # Find candidate files newer than 48h
    cutoff = datetime.now().timestamp() - 48 * 3600
    candidates = []
    for f in rulesets_dir.glob("candidate_*.json"):
        if f.stat().st_mtime > cutoff:
            candidates.append(f.name)

    if candidates:
        # Check if calibration note exists for today
        note = REPO_ROOT / "output" / "calibration" / "calibration_note.md"
        if not note.exists() or note.stat().st_mtime < cutoff:
            anomalies.append(f"CALIBRATION_REVIEW_NEEDED: {len(candidates)} candidate(s): {', '.join(candidates)}")

    if anomalies:
        return CheckResult("calibration", "WARN", "New candidates", anomalies)
    return CheckResult("calibration", "OK", "No pending candidates")


# ── Shadow Monitor ────────────────────────────────────────────


def check_shadow_monitor(dt: date) -> CheckResult:
    """Check shadow portfolio performance and policy-comparison artifacts (canonical portfolio-risk)."""
    ds = as_of_date(dt)
    anomalies = []

    policy_path = ARTIFACTS_DIR / "policy_shadow" / "tier_weighted" / f"{ds}_comparison.json"
    if not policy_path.exists():
        anomalies.append(f"MISSING: policy_shadow comparison for {ds}")

    monitor_path = ARTIFACTS_DIR / "shadow_monitor" / f"{ds}_monitor.json"
    if not monitor_path.exists():
        latest, _, _ = newest_artifact_freshness(
            REPO_ROOT,
            ["artifacts/shadow_monitor/", "artifacts/policy_shadow/tier_weighted/"],
        )
        detail = f"No monitor for {ds}"
        if latest is not None:
            detail += f"; latest={latest} ({age_days(dt, latest)}d ago)"
            if age_days(dt, latest) > STALENESS_DAYS_BY_CADENCE["daily_after_production"]:
                anomalies.append(format_stale_source(latest, dt, STALENESS_DAYS_BY_CADENCE["daily_after_production"]))
        if anomalies:
            return CheckResult("shadow_monitor", "STALE", detail, anomalies)
        return CheckResult("shadow_monitor", "STALE", detail)

    try:
        monitor = json.loads(monitor_path.read_text())
    except json.JSONDecodeError:
        return CheckResult("shadow_monitor", "FAIL", "Monitor corrupt", ["CORRUPT"])

    # Check alert codes from the monitor
    alerts = monitor.get("alerts", [])
    alert_level = monitor.get("attention", "UNKNOWN")

    for alert in alerts:
        code = alert.get("code", "") if isinstance(alert, dict) else str(alert)
        if any(
            x in str(code).upper()
            for x in ("DRAWDOWN_STREAK", "SINGLE_DAY_LOSS", "EXCESS_DETERIORATION", "MAX_DRAWDOWN")
        ):
            anomalies.append(f"PERF_ALERT: {code}")

    # Fallback: read performance.csv for streak detection
    perf_path = ARTIFACTS_DIR / "live_shadow" / "performance.csv"
    if perf_path.exists() and not anomalies:
        try:
            lines = perf_path.read_text().strip().split("\n")
            if len(lines) > 3:
                # Use the true file header, then inspect the latest rows.
                header = lines[0].split(",")
                recent = lines[-5:]
                pnl_idx = next((i for i, h in enumerate(header) if "pnl" in h.lower() or "return" in h.lower()), None)
                if pnl_idx is not None:
                    losses = 0
                    for row in recent:
                        cols = row.split(",")
                        if len(cols) > pnl_idx:
                            try:
                                if float(cols[pnl_idx]) < 0:
                                    losses += 1
                                else:
                                    losses = 0
                            except ValueError:
                                pass
                    if losses >= 3:
                        anomalies.append(f"DRAWDOWN_STREAK: {losses} consecutive losing days")
        except Exception:
            pass

    if anomalies:
        return CheckResult(
            "shadow_monitor", "FAIL" if len(anomalies) > 1 else "WARN", f"attention={alert_level}", anomalies
        )
    return CheckResult("shadow_monitor", "OK", f"attention={alert_level}")


# ── AACT Trial Ingest ─────────────────────────────────────────


def check_aact_ingest(dt: date) -> CheckResult:
    """Check AACT ingest ran and produced healthy output."""
    ds = as_of_date(dt)
    aact_snap = REPO_ROOT / "data" / "aact" / "snapshots" / ds
    anomalies = []

    if not aact_snap.is_dir():
        # Check if any recent snapshot exists (within 8 days — AACT is weekly)
        aact_base = REPO_ROOT / "data" / "aact" / "snapshots"
        if aact_base.is_dir():
            snaps = sorted(
                (d for d in aact_base.iterdir() if d.is_dir() and (d / "aact_health.json").exists()),
                reverse=True,
            )
            if snaps:
                latest = snaps[0].name
                age = (dt - date.fromisoformat(latest)).days if latest >= "2020" else 999
                if age <= 8:
                    return CheckResult("aact_trial_ingest", "OK", f"Latest snapshot: {latest} ({age}d old, weekly)")
        return CheckResult("aact_trial_ingest", "STALE", f"No AACT snapshot within 8d of {ds}")

    # Check health report
    health_path = aact_snap / "aact_health.json"
    if health_path.exists():
        try:
            health = json.loads(health_path.read_text())
            n_trials = health.get("n_trials", 0)
            n_linked = health.get("n_linked_to_ticker", 0)
            if n_trials == 0:
                anomalies.append("EMPTY_SNAPSHOT: 0 trials")
            if n_linked == 0 and n_trials > 0:
                anomalies.append("LINKAGE_BROKEN: 0 linked to tickers")
            schema_drift = health.get("schema_drift", [])
            if schema_drift:
                anomalies.append(f"SCHEMA_DRIFT: {', '.join(schema_drift[:3])}")
        except json.JSONDecodeError:
            anomalies.append("HEALTH_CORRUPT")
    else:
        anomalies.append("HEALTH_MISSING")

    # Check key outputs exist
    for fname in ["trial_master.json", "trial_status_deltas.jsonl"]:
        if not (aact_snap / fname).exists():
            anomalies.append(f"OUTPUT_MISSING: {fname}")

    if anomalies:
        return CheckResult(
            "aact_trial_ingest", "WARN" if len(anomalies) <= 1 else "FAIL", f"{len(anomalies)} issue(s)", anomalies
        )
    return CheckResult("aact_trial_ingest", "OK", f"Snapshot {ds} healthy")


# ── Biotech News Digest ──────────────────────────────────────


def check_herald_news_pipeline(dt: date) -> CheckResult:
    """Verify Herald news digests and press-release freshness (herald canonical owner).

    Artifact filenames retain biotech_news_digest_* prefix from build_news_digest.py.
    The evening digest arrives at ~18:00 ET, but heartbeat runs at 17:30.
    To avoid false positives, check yesterday's (completed) digest counts
    when running before 19:00, and today's only at/after 19:00.
    """
    agent = "herald"
    ds = as_of_date(dt)
    anomalies = []
    digest_dir = ARTIFACTS_DIR / "news_digest"

    if not digest_dir.is_dir():
        return CheckResult(agent, "STALE", "No news_digest artifact directory")

    hour = datetime.now().hour
    today_digests = list(digest_dir.glob(f"biotech_news_digest_{ds}_*.json"))

    if hour >= 19:
        # After all scheduled digests: check today
        if len(today_digests) == 0:
            anomalies.append("MISSED_ALL: no digest by 19:00")
        elif len(today_digests) < 2:
            anomalies.append(f"DIGEST_LAG: only {len(today_digests)} digest(s) by 19:00")
    else:
        # Before evening digest: check yesterday (completed day) instead
        from datetime import timedelta

        yesterday = dt - timedelta(days=1)
        yesterday_ds = as_of_date(yesterday)
        yesterday_digests = list(digest_dir.glob(f"biotech_news_digest_{yesterday_ds}_*.json"))
        if yesterday_digests is not None and len(yesterday_digests) == 0:
            # Skip weekends — no digests expected
            if yesterday.weekday() < 5:
                anomalies.append(f"MISSED_YESTERDAY: no digest for {yesterday_ds}")

    # Check press release freshness (classified artifacts — authoritative)
    classified_dir = REPO_ROOT / "data" / "press_releases" / "classified"
    latest_classified: date | None = None
    if classified_dir.is_dir():
        for path in classified_dir.glob("classified_*.jsonl"):
            m = re.search(r"classified_(\d{4}-\d{2}-\d{2})\.jsonl$", path.name)
            if not m:
                continue
            try:
                d = date.fromisoformat(m.group(1))
            except ValueError:
                continue
            if latest_classified is None or d > latest_classified:
                latest_classified = d
    if latest_classified is not None:
        age = (dt - latest_classified).days
        if age > 2:
            anomalies.append(f"STALE_SOURCE: press_releases last updated {age}d ago")

    # Check delivery log for failures
    delivery_log = digest_dir / "delivery_log.jsonl"
    if delivery_log.exists():
        try:
            lines = delivery_log.read_text().strip().split("\n")
            today_lines = [ln for ln in lines[-20:] if ds in ln]
            for line in today_lines:
                entry = json.loads(line)
                if entry.get("status") == "FAIL":
                    anomalies.append(f"DELIVERY_FAIL: {entry.get('window', 'unknown')} send failed")
        except (json.JSONDecodeError, ValueError):
            pass

    if anomalies:
        return CheckResult(agent, "WARN" if len(anomalies) <= 1 else "FAIL", f"{len(anomalies)} issue(s)", anomalies)
    return CheckResult(agent, "OK", f"{len(today_digests)} digest(s) for {ds}")


# ── Calibration Evidence ─────────────────────────────────────


def check_calibration_evidence(dt: date) -> CheckResult:
    """Verify calibration evidence ledger is reasonably fresh (weekly tolerance)."""
    evidence_dir = ARTIFACTS_DIR / "calibration_evidence"
    anomalies = []

    if not evidence_dir.is_dir():
        return CheckResult("calibration_evidence", "STALE", "No calibration_evidence directory")

    ledger = evidence_dir / "ledger.jsonl"
    if not ledger.exists():
        return CheckResult("calibration_evidence", "STALE", "No evidence ledger")

    # Check ledger freshness (tolerate up to 10 calendar days / ~7 trading days)
    try:
        mtime = datetime.fromtimestamp(ledger.stat().st_mtime).date()
        age = (dt - mtime).days
        if age > 10:
            anomalies.append(f"STALE_LEDGER: last modified {age}d ago (limit 10d)")
    except OSError:
        anomalies.append("LEDGER_STAT_FAIL: cannot read file mtime")

    # Check for recent evidence files
    evidence_files = sorted(evidence_dir.glob("*_evidence.json"), reverse=True)
    if not evidence_files:
        anomalies.append("NO_EVIDENCE_FILES: directory empty")
    elif evidence_files:
        latest = evidence_files[0].name[:10]
        try:
            latest_date = date.fromisoformat(latest)
            age = (dt - latest_date).days
            if age > 10:
                anomalies.append(f"STALE_EVIDENCE: latest file is {age}d old")
        except ValueError:
            pass

    if anomalies:
        return CheckResult(
            "calibration_evidence", "WARN" if len(anomalies) <= 1 else "FAIL", f"{len(anomalies)} issue(s)", anomalies
        )
    return CheckResult("calibration_evidence", "OK", f"Ledger fresh, {len(evidence_files)} evidence file(s)")


# ── Data Auditor ─────────────────────────────────────────────


def check_data_auditor(dt: date) -> CheckResult:
    """Verify data auditor produced today's integrity report."""
    ds = as_of_date(dt)
    auditor_dir = ARTIFACTS_DIR / "data_auditor"
    anomalies = []

    if not auditor_dir.is_dir():
        return CheckResult("data_auditor", "STALE", "No data_auditor directory")

    report_path = auditor_dir / f"integrity_report_{ds}.json"
    if not report_path.exists():
        # Check most recent report
        reports = sorted(auditor_dir.glob("integrity_report_*.json"), reverse=True)
        if not reports:
            return CheckResult("data_auditor", "STALE", "No integrity reports found")
        latest = reports[0].name
        latest_date_str = latest.replace("integrity_report_", "").replace(".json", "")
        try:
            age = (dt - date.fromisoformat(latest_date_str)).days
            if age > 2:
                return CheckResult("data_auditor", "STALE", f"Latest report {age}d old ({latest_date_str})")
        except ValueError:
            pass
        report_path = reports[0]

    # Read the report verdict
    try:
        report = json.loads(report_path.read_text())
        verdict = report.get("verdict", "UNKNOWN")
        if verdict == "FAIL":
            issues = report.get("issues", [])
            for issue in issues[:5]:
                anomalies.append(
                    f"INTEGRITY_{issue.get('severity', 'UNKNOWN').upper()}: {issue.get('check', 'unknown')}"
                )
            return CheckResult("data_auditor", "FAIL", "Integrity verdict: FAIL", anomalies)
        elif verdict == "WARN":
            return CheckResult(
                "data_auditor",
                "WARN",
                "Integrity verdict: WARN",
                [f"WARN: {i.get('check', '?')}" for i in report.get("issues", [])[:3]],
            )
    except (json.JSONDecodeError, KeyError):
        anomalies.append("REPORT_CORRUPT: cannot parse integrity report")
        return CheckResult("data_auditor", "FAIL", "Report corrupt", anomalies)

    return CheckResult("data_auditor", "OK", f"Integrity report {report_path.name}: PASS")


# ── Production QA Agent ──────────────────────────────────────


def check_review_queue_steward(dt: date) -> CheckResult:
    """Verify review_queue_steward ran by checking its invocation log mtime.

    The agent is chat-mode only — it produces no artifact files by design.
    Its liveness signal is the most recent logs/agents_direct/review_queue_steward_*.json.
    Cadence: daily_after_production (threshold 2d).
    """
    log_dir = LOGS_DIR / "agents_direct"
    threshold = STALENESS_DAYS_BY_CADENCE.get("daily_after_production", 2)

    logs = sorted(log_dir.glob("review_queue_steward_*.json")) if log_dir.is_dir() else []
    if not logs:
        return CheckResult(
            "review_queue_steward",
            "STALE",
            "no invocation logs found",
            ["NO_ARTIFACTS: logs/agents_direct/review_queue_steward_*.json"],
        )

    newest = max(logs, key=lambda p: p.stat().st_mtime)
    age_days = (datetime.now().timestamp() - newest.stat().st_mtime) / 86400
    newest_date = datetime.fromtimestamp(newest.stat().st_mtime).date()

    if age_days > threshold:
        return CheckResult(
            "review_queue_steward",
            "STALE",
            f"newest log={newest_date} ({age_days:.1f}d > {threshold}d)",
            [f"STALE_ARTIFACT: {age_days:.1f}d since last invocation (threshold {threshold}d)"],
        )

    # Peek at status field to surface errors without failing for chat-mode output
    anomalies: list[str] = []
    try:
        data = json.loads(newest.read_text())
        if data.get("status") == "error":
            anomalies.append(f"INVOCATION_ERROR: {data.get('error', 'unknown')}")
    except (json.JSONDecodeError, KeyError):
        pass

    if anomalies:
        return CheckResult("review_queue_steward", "WARN", f"last run={newest_date}", anomalies)
    return CheckResult(
        "review_queue_steward",
        "OK",
        f"last run={newest_date} ({age_days:.1f}d, log-invocation check)",
    )


def check_production_qa(dt: date) -> CheckResult:
    """Verify production_qa ran and produced a report (schema: production_qa.v1).

    production_qa runs post-market (~20:30-21:00 ET). The fleet receipt runs at
    17:30 ET. When the receipt fires before the report has landed, return SKIP
    (pending) instead of STALE to avoid a false alarm on every weekday.
    Threshold: if it's the same calendar day and current hour (ET) < 21, pending.
    """
    ds = as_of_date(dt)
    anomalies: list[str] = []
    verdict = "UNKNOWN"

    report = REPO_ROOT / "artifacts" / "production_qa" / f"{ds}_report.json"
    if not report.exists():
        # Check if we're still within the expected completion window (before 21:00 ET).
        # ET = UTC-4 (EDT) or UTC-5 (EST); use UTC-4 as conservative bound.
        now_et_hour = (datetime.utcnow().hour - 4) % 24
        today_str = datetime.utcnow().date().isoformat()
        if ds == today_str and now_et_hour < 21:
            return CheckResult(
                "production_qa",
                "SKIP",
                f"No production_qa report for {ds} yet — pending (expected ~20:30-21:00 ET, now ~{now_et_hour:02d}:xx ET)",
            )
        latest, _, _ = newest_artifact_freshness(REPO_ROOT, ["artifacts/production_qa/"])
        detail = f"No production_qa report for {ds}"
        if latest is not None:
            detail += f"; latest={latest} ({age_days(dt, latest)}d ago)"
        return CheckResult("production_qa", "STALE", detail)

    try:
        data = json.loads(report.read_text())
        verdict = data.get("verdict", "UNKNOWN")
        failing = [c.get("check", "?") for c in data.get("checks", []) if c.get("status") == "FAIL"]
        if verdict == "RED":
            names = ", ".join(failing) if failing else "no checks named"
            anomalies.append(f"VERDICT_RED: {len(failing)} failing ({names})")
        elif verdict == "YELLOW":
            names = ", ".join(failing) if failing else "no checks named"
            anomalies.append(f"VERDICT_YELLOW: {len(failing)} failing ({names})")
    except (json.JSONDecodeError, KeyError):
        anomalies.append("REPORT_CORRUPT: cannot parse production_qa report")

    if anomalies:
        return CheckResult("production_qa", "WARN", f"{len(anomalies)} issue(s)", anomalies)
    return CheckResult("production_qa", "OK", f"Report {ds}: {verdict}")


# ── Ops Agent ─────────────────────────────────────────────────


def check_ops(dt: date) -> CheckResult:
    """Verify ops digest exists for today; report latest content date when missing."""
    ds = as_of_date(dt)
    digest_path = ARTIFACTS_DIR / "ops_digest" / f"{ds}_digest.json"
    threshold = STALENESS_DAYS_BY_CADENCE["daily_after_production"]

    if digest_path.exists():
        try:
            digest = json.loads(digest_path.read_text())
            attention = digest.get("attention", "UNKNOWN")
            return CheckResult("ops", "OK", f"Digest {ds}: attention={attention}")
        except json.JSONDecodeError:
            return CheckResult("ops", "FAIL", f"Digest corrupt for {ds}", ["CORRUPT_DIGEST"])

    latest, _, _ = newest_artifact_freshness(REPO_ROOT, ["artifacts/ops_digest/"])
    detail = f"No ops_digest for {ds}"
    anomalies: list[str] = []
    if latest is not None:
        detail += f"; latest={latest} ({age_days(dt, latest)}d ago)"
        if age_days(dt, latest) > threshold:
            anomalies.append(format_stale_source(latest, dt, threshold))
    else:
        anomalies.append("NO_ARTIFACTS: artifacts/ops_digest/")
    return CheckResult("ops", "STALE", detail, anomalies)


# ── CTgov Poller ──────────────────────────────────────────────


def check_ctgov_poller(dt: date) -> CheckResult:
    """Verify CTgov cache and daily diff artifacts (cache + artifacts/ctgov_daily/)."""
    ds = as_of_date(dt)
    threshold = STALENESS_DAYS_BY_CADENCE["daily_premarket"]
    anomalies: list[str] = []

    cache_today = REPO_ROOT / "cache" / "ctgov" / f"trial_records_{ds}.json"
    diff_today = ARTIFACTS_DIR / "ctgov_daily" / f"{ds}_diff.json"

    latest_cache, _, _ = newest_artifact_freshness(REPO_ROOT, ["cache/ctgov/"])
    latest_diff, _, _ = newest_artifact_freshness(REPO_ROOT, ["artifacts/ctgov_daily/"])

    if not cache_today.exists():
        if latest_cache is None:
            anomalies.append("NO_CACHE: no cache/ctgov/trial_records_*.json found")
        else:
            cache_age = age_days(dt, latest_cache)
            anomalies.append(f"MISSING_CACHE_TODAY: latest cache {latest_cache} ({cache_age}d ago)")
            if cache_age > threshold:
                anomalies.append(format_stale_source(latest_cache, dt, threshold))

    if not diff_today.exists():
        if latest_diff is None:
            anomalies.append("NO_DIFF: no artifacts/ctgov_daily/*_diff.json found")
        else:
            diff_age = age_days(dt, latest_diff)
            anomalies.append(f"MISSING_DIFF_TODAY: latest diff {latest_diff} ({diff_age}d ago)")
            if diff_age > threshold:
                anomalies.append(format_stale_source(latest_diff, dt, threshold))

    if anomalies:
        status = "STALE" if any("STALE_SOURCE" in a or "NO_" in a for a in anomalies) else "WARN"
        return CheckResult("ctgov_poller", status, f"{len(anomalies)} issue(s) for {ds}", anomalies)
    return CheckResult("ctgov_poller", "OK", f"Cache and diff present for {ds}")


# ── Sentinel ──────────────────────────────────────────────────


def check_sentinel(dt: date) -> CheckResult:
    """Monitor ruleset drift health and promotion receipt presence."""
    ds = as_of_date(dt)
    anomalies: list[str] = []

    health_path = SNAPSHOT_DIR / ds / "ruleset_health.json"
    drift_path = SNAPSHOT_DIR / ds / "drift_report.json"
    history_path = ARTIFACTS_DIR / "ruleset_health_history.jsonl"
    promotions_dir = ARTIFACTS_DIR / "promotions"

    if health_path.exists():
        try:
            health = json.loads(health_path.read_text())
            status = str(health.get("status", "UNKNOWN")).upper()
            consec = int(health.get("consecutive_warn_days", 0) or 0)
            if health.get("recommend_rollback"):
                anomalies.append("ROLLBACK_RECOMMENDED: ruleset_health recommends rollback")
            elif status in ("WARN", "FAIL"):
                anomalies.append(f"RULESET_{status}: consecutive_warn_days={consec}")
            elif status == "PASS":
                pass
            else:
                anomalies.append(f"RULESET_STATUS: {status}")
        except (json.JSONDecodeError, TypeError, ValueError):
            anomalies.append("RULESET_HEALTH_CORRUPT")
    else:
        latest, _, _ = newest_artifact_freshness(REPO_ROOT, ["artifacts/ruleset_health_history.jsonl"])
        detail = f"No ruleset_health.json for {ds}"
        if latest is not None:
            detail += f"; history latest={latest} ({age_days(dt, latest)}d ago)"
        if not drift_path.exists():
            anomalies.append(f"MISSING_DRIFT_REPORT: {detail}")
        else:
            anomalies.append(f"MISSING_RULESET_HEALTH: {detail}")

    if promotions_dir.is_dir():
        receipts = sorted(promotions_dir.glob("*_receipt.json"))
        if not receipts:
            anomalies.append("NO_PROMOTION_RECEIPTS: artifacts/promotions empty")
    else:
        anomalies.append("NO_PROMOTIONS_DIR")

    if history_path.exists() and health_path.exists():
        try:
            lines = history_path.read_text().strip().split("\n")
            recent = [json.loads(ln) for ln in lines[-5:] if ln.strip()]
            warn_streak = sum(1 for r in recent if str(r.get("status", "")).upper() == "WARN")
            if warn_streak >= 2:
                anomalies.append(f"WARN_STREAK: {warn_streak} of last 5 history entries WARN")
        except (json.JSONDecodeError, ValueError):
            pass

    if anomalies:
        status = "FAIL" if any("ROLLBACK" in a or "RULESET_FAIL" in a for a in anomalies) else "WARN"
        return CheckResult("sentinel", status, f"{len(anomalies)} issue(s)", anomalies)
    return CheckResult("sentinel", "OK", f"Ruleset health OK for {ds}")


# ── Shared dated-artifact helper ──────────────────────────────


def _check_dated_artifact_file(
    agent: str,
    dt: date,
    *,
    today_path: Path,
    freshness_globs: list[str],
    cadence_key: str,
    ok_message: str | None = None,
) -> CheckResult:
    """Today's dated artifact exists, else report latest content date vs cadence threshold."""
    ds = as_of_date(dt)
    threshold = STALENESS_DAYS_BY_CADENCE.get(cadence_key)

    if today_path.exists():
        return CheckResult(agent, "OK", ok_message or f"Artifact present for {ds}")

    latest, sample_path, _ = newest_artifact_freshness(REPO_ROOT, freshness_globs)
    detail = f"No artifact for {ds}"
    anomalies: list[str] = []
    if latest is not None:
        detail += f"; latest={latest} ({age_days(dt, latest)}d ago)"
        if threshold is not None and age_days(dt, latest) > threshold:
            anomalies.append(format_stale_source(latest, dt, threshold))
    else:
        anomalies.append(f"NO_ARTIFACTS: {freshness_globs}")
    status = "STALE" if anomalies else "WARN"
    return CheckResult(agent, status, detail, anomalies)


# ── Catalyst Delta ────────────────────────────────────────────


def check_catalyst_delta(dt: date) -> CheckResult:
    ds = as_of_date(dt)
    return _check_dated_artifact_file(
        "catalyst_delta",
        dt,
        today_path=ARTIFACTS_DIR / "catalyst_delta" / f"{ds}_delta.json",
        freshness_globs=["artifacts/catalyst_delta/"],
        cadence_key="daily_after_production",
    )


# ── Postmortem ────────────────────────────────────────────────


def check_postmortem(dt: date) -> CheckResult:
    threshold = STALENESS_DAYS_BY_CADENCE["daily_after_production"]
    latest, sample_path, _ = newest_artifact_freshness(REPO_ROOT, ["artifacts/postmortem/"])
    if latest is None:
        return CheckResult("postmortem", "STALE", "No postmortem artifacts", ["NO_ARTIFACTS: artifacts/postmortem/"])
    age = age_days(dt, latest)
    sample = sample_path.relative_to(REPO_ROOT) if sample_path else "?"
    if age > threshold:
        return CheckResult(
            "postmortem",
            "STALE",
            f"newest event capture {latest} ({age}d > {threshold}d, {sample})",
            [format_stale_source(latest, dt, threshold)],
        )
    return CheckResult("postmortem", "OK", f"newest={latest} ({age}d, {sample})")


# ── CRT Resolution Watcher ────────────────────────────────────


def check_crt_resolution_watcher(dt: date) -> CheckResult:
    ds = as_of_date(dt)
    threshold = STALENESS_DAYS_BY_CADENCE["daily_after_production"]
    anomalies: list[str] = []

    join_path = REPO_ROOT / "output" / "catalyst_ev" / "crt_options_join.json"
    res_dir = SNAPSHOT_DIR / "resolutions"
    latest_join, _, _ = newest_artifact_freshness(REPO_ROOT, ["output/catalyst_ev/"])
    latest_res, _, _ = newest_artifact_freshness(REPO_ROOT, ["data/snapshots/resolutions/"])

    if not join_path.exists():
        if latest_join is None:
            anomalies.append("NO_CRT_JOIN: output/catalyst_ev/ empty")
        else:
            join_age = age_days(dt, latest_join)
            anomalies.append(f"MISSING_CRT_JOIN: latest {latest_join} ({join_age}d ago)")
            if join_age > threshold:
                anomalies.append(format_stale_source(latest_join, dt, threshold))
    elif latest_join is not None and age_days(dt, latest_join) > threshold:
        anomalies.append(format_stale_source(latest_join, dt, threshold))

    if not res_dir.is_dir() or not any(res_dir.iterdir()):
        anomalies.append("NO_RESOLUTIONS: data/snapshots/resolutions/ empty")
    elif latest_res is not None and age_days(dt, latest_res) > threshold + 3:
        anomalies.append(f"STALE_RESOLUTIONS: latest {latest_res} ({age_days(dt, latest_res)}d ago)")

    if anomalies:
        status = "STALE" if any("STALE" in a or "NO_" in a or "MISSING" in a for a in anomalies) else "WARN"
        return CheckResult("crt_resolution_watcher", status, f"{len(anomalies)} issue(s) for {ds}", anomalies)
    return CheckResult("crt_resolution_watcher", "OK", f"CRT join and resolutions fresh for {ds}")


# ── Ops Supervisor ────────────────────────────────────────────


def check_ops_supervisor(dt: date) -> CheckResult:
    ds = as_of_date(dt)
    return _check_dated_artifact_file(
        "ops_supervisor",
        dt,
        today_path=ARTIFACTS_DIR / "ops_supervisor" / f"{ds}_supervisor.json",
        freshness_globs=["artifacts/ops_supervisor/"],
        cadence_key="daily_after_production",
        ok_message=f"Supervisor report for {ds}",
    )


# ── Options / Price Action / Grok / Intraday watches ─────────


def check_options_watch(dt: date) -> CheckResult:
    ds = as_of_date(dt)
    return _check_dated_artifact_file(
        "options_watch",
        dt,
        today_path=ARTIFACTS_DIR / "options_watch" / f"{ds}_watch.json",
        freshness_globs=["artifacts/options_watch/"],
        cadence_key="daily_after_production",
    )


def check_price_action_watch(dt: date) -> CheckResult:
    ds = as_of_date(dt)
    return _check_dated_artifact_file(
        "price_action_watch",
        dt,
        today_path=ARTIFACTS_DIR / "price_action_watch" / f"{ds}_watch.json",
        freshness_globs=["artifacts/price_action_watch/"],
        cadence_key="daily_after_production",
    )


def check_grok_biotech_watch(dt: date) -> CheckResult:
    ds = as_of_date(dt)
    return _check_dated_artifact_file(
        "grok_biotech_watch",
        dt,
        today_path=ARTIFACTS_DIR / "grok_watch" / f"{ds}_alerts.json",
        freshness_globs=["artifacts/grok_watch/"],
        cadence_key="intraday",
    )


def check_intraday_mover_watch(dt: date) -> CheckResult:
    ds = as_of_date(dt)
    return _check_dated_artifact_file(
        "intraday_mover_watch",
        dt,
        today_path=ARTIFACTS_DIR / "intraday_mover_watch" / f"{ds}_digest.json",
        freshness_globs=["artifacts/intraday_mover_watch/"],
        cadence_key="intraday",
    )


# ── Earnings / Event Analyst / Universe ─────────────────────────


def check_earnings_calendar_sync(dt: date) -> CheckResult:
    ds = as_of_date(dt)
    threshold = STALENESS_DAYS_BY_CADENCE["daily_premarket"]
    sync_report = ARTIFACTS_DIR / "earnings_sync" / f"{ds}_sync_report.json"
    ics_path = ARTIFACTS_DIR / "earnings_sync" / "biotech_earnings.ics"

    if sync_report.exists():
        return CheckResult("earnings_calendar_sync", "OK", f"Sync report for {ds}")

    latest, _, _ = newest_artifact_freshness(REPO_ROOT, ["artifacts/earnings_sync/"])
    if ics_path.exists() and latest is not None and age_days(dt, latest) <= threshold:
        return CheckResult("earnings_calendar_sync", "OK", f"ICS present, latest content={latest}")

    detail = f"No sync report for {ds}"
    anomalies: list[str] = []
    if latest is not None:
        detail += f"; latest={latest} ({age_days(dt, latest)}d ago)"
        if age_days(dt, latest) > threshold:
            anomalies.append(format_stale_source(latest, dt, threshold))
    else:
        anomalies.append("NO_ARTIFACTS: artifacts/earnings_sync/")
    return CheckResult("earnings_calendar_sync", "STALE", detail, anomalies)


def check_event_analyst(dt: date) -> CheckResult:
    ds = as_of_date(dt)
    return _check_dated_artifact_file(
        "event_analyst",
        dt,
        today_path=ARTIFACTS_DIR / "event_analyst" / f"{ds}_summary.json",
        freshness_globs=["artifacts/event_analyst/"],
        cadence_key="weekly",
    )


def check_universe_maintenance(dt: date) -> CheckResult:
    ds = as_of_date(dt)
    return _check_dated_artifact_file(
        "universe_maintenance",
        dt,
        today_path=ARTIFACTS_DIR / "universe_maintenance" / f"{ds}_report.json",
        freshness_globs=["artifacts/universe_maintenance/"],
        cadence_key="weekly",
    )


# ── Orchestrator ──────────────────────────────────────────────

# Specialized check functions, keyed by registry name (agents/AGENT_REGISTRY.json).
# Active+supervised agents not listed here fall back to check_generic_freshness().
SPECIALIZED_CHECKS = {
    "qa": check_qa,
    "ic_health_monitor": check_ic_health,
    "ic_memory_hygiene": check_ic_memory_hygiene,
    "fleet_steward": check_fleet_steward,
    "calibration": check_calibration,
    "shadow_monitor": check_shadow_monitor,
    "aact_trial_ingest": check_aact_ingest,
    "herald": check_herald_news_pipeline,
    "calibration_evidence": check_calibration_evidence,
    "data_auditor": check_data_auditor,
    "production_qa": check_production_qa,
    "review_queue_steward": check_review_queue_steward,
    "ops": check_ops,
    "ctgov_poller": check_ctgov_poller,
    "sentinel": check_sentinel,
    "catalyst_delta": check_catalyst_delta,
    "postmortem": check_postmortem,
    "crt_resolution_watcher": check_crt_resolution_watcher,
    "ops_supervisor": check_ops_supervisor,
    "options_watch": check_options_watch,
    "price_action_watch": check_price_action_watch,
    "earnings_calendar_sync": check_earnings_calendar_sync,
    "grok_biotech_watch": check_grok_biotech_watch,
    "intraday_mover_watch": check_intraday_mover_watch,
    "event_analyst": check_event_analyst,
    "universe_maintenance": check_universe_maintenance,
}

# CLI --agent map: registry names only.
# shadow_watch / policy_shadow_watch retired 2026-05-30 (Spec 085 Path B): shadow_monitor canonical.
# biotech_news_digest / company_news_ingest retired into herald; bioshort_watch LLM consumer remains suppressed.
# Artifact files still use biotech_news_digest_{date}_{window}.json from build_news_digest.py.
AGENTS = dict(SPECIALIZED_CHECKS)


def heartbeat_skip_reason(name: str, entry: dict) -> str | None:
    """Return a SKIP reason when daily artifact freshness does not apply."""
    if entry.get("cadence") == "on_demand" and name.startswith(HERMES_JOB_PREFIX):
        return "on_demand Hermes job (invoke agents/<name>/run_job.py)"
    return None


def load_registry() -> dict:
    """Load agents/AGENT_REGISTRY.json; return the 'agents' sub-dict or {}."""
    try:
        data = json.loads(REGISTRY_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Registry unreadable ({e}); registry-driven iteration skipped.", file=sys.stderr)
        return {}
    return data.get("agents", {})


def check_generic_freshness(name: str, entry: dict, dt: date) -> CheckResult:
    """Deterministic artifact-freshness fallback for agents without a specialized check."""
    cadence = entry.get("cadence", "unknown")
    threshold = STALENESS_DAYS_BY_CADENCE.get(cadence)
    paths = entry.get("artifact_paths", [])

    if not paths:
        return CheckResult(name, "SKIP", "no artifact_paths declared")

    newest_date, sample_path, method = newest_artifact_freshness(REPO_ROOT, paths)

    if newest_date is None:
        return CheckResult(name, "STALE", "no artifacts at any declared path", [f"NO_ARTIFACTS: {paths}"])

    age = age_days(dt, newest_date)
    sample = sample_path.relative_to(REPO_ROOT) if sample_path else "?"
    method_note = f"via {method}"

    if threshold is None:
        return CheckResult(name, "OK", f"newest={newest_date} ({age}d, cadence={cadence}, {method_note}, {sample})")
    if age > threshold:
        return CheckResult(
            name,
            "STALE",
            f"newest={newest_date} ({age}d > {threshold}d for cadence={cadence}, {method_note}, {sample})",
            [format_stale_source(newest_date, dt, threshold)],
        )
    return CheckResult(name, "OK", f"newest={newest_date} ({age}d, cadence={cadence}, {method_note}, {sample})")


def _is_artifact_gap(result: CheckResult) -> bool:
    """Classify results caused by missing/stale local artifacts, not agent logic."""
    if result.status == "STALE":
        return True
    if result.status != "WARN" or not result.anomalies:
        return False
    return all(a.startswith(CLOUD_ARTIFACT_STALENESS_PREFIXES) for a in result.anomalies)


def _cloud_unknown_result(result: CheckResult) -> CheckResult:
    detail = (
        "UNKNOWN_CLOUD_ENV: operator-host artifacts unavailable on this host; "
        f"original {result.status}: {result.detail}"
    )
    return CheckResult(result.agent, "SKIP", detail)


def check_stale_memory(name: str, entry: dict, dt: date) -> str | None:
    """Check if agent memory content dates lag artifact content dates.

    Uses YYYY-MM-DD in filenames (not mtime) so git checkout does not
    mask stale LLM memory notes.
    """
    memory_paths = [p for p in entry.get("artifact_paths", []) if "memory/" in p]
    if not memory_paths:
        return None

    memory_dir = REPO_ROOT / memory_paths[0]
    if not memory_dir.is_dir():
        return None

    latest_memory, _, _ = newest_artifact_freshness(REPO_ROOT, memory_paths)

    artifact_paths = [p for p in entry.get("artifact_paths", []) if "memory/" not in p]
    if not artifact_paths:
        return None

    latest_artifact, _, _ = newest_artifact_freshness(REPO_ROOT, artifact_paths)
    if latest_memory is None or latest_artifact is None:
        return None

    gap_days = (latest_artifact - latest_memory).days
    if gap_days > STALE_MEMORY_THRESHOLD_DAYS:
        return (
            f"STALE_MEMORY: memory latest {latest_memory} is {gap_days}d behind "
            f"artifacts ({latest_artifact}); threshold {STALE_MEMORY_THRESHOLD_DAYS}d"
        )

    return None


def run_registry_checks(dt: date) -> tuple[list[CheckResult], dict]:
    """Iterate every active+supervised agent in the registry and run its check.

    Returns (results, counts). Counts keys:
      monitored_count, active_count, stale_count, missing_count, deprecated_count.
    missing_count = active agents with supervised_by_orchestrator=false (coverage gap).
    """
    registry = load_registry()
    empty = {"monitored_count": 0, "active_count": 0, "stale_count": 0, "missing_count": 0, "deprecated_count": 0}
    if not registry:
        return [], empty

    active = {n: e for n, e in registry.items() if e.get("status") == "active"}
    deprecated_count = sum(1 for e in registry.values() if e.get("status") == "deprecated")

    supervised = {n: e for n, e in active.items() if e.get("supervised_by_orchestrator", True)}
    terminal_unsupervised = {
        n: e
        for n, e in active.items()
        if n in TERMINAL_UNSUPERVISED_AGENTS and not e.get("supervised_by_orchestrator", True)
    }
    opted_out = {
        n: e
        for n, e in active.items()
        if not e.get("supervised_by_orchestrator", True) and n not in TERMINAL_UNSUPERVISED_AGENTS
    }

    results: list[CheckResult] = []
    cloud_env = is_cloud_agent_environment()
    for name in sorted(supervised):
        entry = supervised[name]
        skip_reason = heartbeat_skip_reason(name, entry)
        if skip_reason:
            results.append(CheckResult(name, "SKIP", skip_reason))
            continue
        check_fn = SPECIALIZED_CHECKS.get(name)
        try:
            start_time = time.time()
            result = check_fn(dt) if check_fn is not None else check_generic_freshness(name, entry, dt)
            latency_ms = (time.time() - start_time) * 1000

            # Log check execution
            try:
                log_skill(
                    skill_name=f"{name}_check",
                    task_context=f"Health check for {as_of_date(dt)}",
                    inputs={"check_date": as_of_date(dt), "check_name": name},
                    outputs={
                        "status": result.status,
                        "anomaly_count": len(result.anomalies),
                        "detail": result.detail[:100],
                    },
                    latency_ms=latency_ms,
                    success=(result.status in ("OK", "WARN", "SKIP")),
                    error=result.detail if result.status == "FAIL" else None,
                    environment="prod",
                )
            except Exception:
                pass  # Non-blocking: don't let logging failures break the check
        except Exception as e:  # noqa: BLE001
            latency_ms = (time.time() - start_time) * 1000
            result = CheckResult(name, "FAIL", f"Check crashed: {e}", [f"EXCEPTION: {e}"])
            # Log failure
            try:
                log_skill(
                    skill_name=f"{name}_check",
                    task_context=f"Health check for {as_of_date(dt)}",
                    inputs={"check_date": as_of_date(dt), "check_name": name},
                    outputs={"status": "CRASH"},
                    latency_ms=latency_ms,
                    success=False,
                    error=str(e)[:500],
                    environment="prod",
                )
            except Exception:
                pass  # Non-blocking
        if cloud_env and _is_artifact_gap(result):
            result = _cloud_unknown_result(result)

        # Check for stale memory artifacts (memory mtime >7d behind artifact mtime)
        if result.status not in ("SKIP", "FAIL"):  # Don't override SKIP/FAIL
            stale_memory_anomaly = check_stale_memory(name, entry, dt)
            if stale_memory_anomaly:
                # Add stale-memory finding without changing overall status
                # (treats it as a WARN-level observation, not a blocker)
                result.anomalies.append(stale_memory_anomaly)
                if result.status == "OK":
                    result.status = "WARN"
                    result.detail = "Memory staleness detected"

        results.append(result)

    for name in sorted(opted_out):
        note = opted_out[name].get("notes", "")[:80]
        results.append(CheckResult(name, "SKIP", f"unsupervised coverage gap: {note}"))

    for name in sorted(terminal_unsupervised):
        note = terminal_unsupervised[name].get("notes", "")[:80]
        results.append(CheckResult(name, "SKIP", f"terminal unsupervised: {note}"))

    counts = {
        "monitored_count": len(supervised),
        "active_count": len(active),
        "stale_count": sum(1 for r in results if r.status == "STALE"),
        "missing_count": len(opted_out),
        "deprecated_count": deprecated_count,
    }
    return results, counts


def _derive_verdict(results: list[CheckResult], counts: dict, snapshot_ok: bool) -> str:
    """Deterministic fleet verdict from heartbeat results + registry counts."""
    if not snapshot_ok:
        return "RED"
    if any(r.status == "FAIL" for r in results):
        return "RED"
    if counts.get("missing_count", 0) > 0:
        return "RED"
    if any(r.status in ("WARN", "STALE") for r in results):
        return "YELLOW"
    return "GREEN"


def _find_previous_snapshot(dt: date) -> str | None:
    """Walk back from dt to find the most recent prior snapshot date, if any."""
    if not SNAPSHOT_DIR.is_dir():
        return None
    dates = sorted(
        d.name
        for d in SNAPSHOT_DIR.iterdir()
        if d.is_dir() and (d / "rankings.csv").exists() and d.name < as_of_date(dt)
    )
    return dates[-1] if dates else None


def write_fleet_receipt(results: list[CheckResult], counts: dict, dt: date) -> Path:
    """Write a deterministic daily fleet receipt to agents/fleet_steward/memory/.

    This is the Conductor/Director output restored after the fleet_steward LLM
    agent was replaced by this orchestrator. Format matches the historical
    receipts but scope is narrower: status-only, no analyst synthesis.
    """
    ds = as_of_date(dt)
    receipt_dir = REPO_ROOT / "agents" / "fleet_steward" / "memory"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    out_path = receipt_dir / f"{ds}_receipt.md"

    snapshot_ok = (SNAPSHOT_DIR / ds / "rankings.csv").exists()
    snapshot_unknown = is_cloud_agent_environment() and not snapshot_ok
    prev_snapshot = _find_previous_snapshot(dt)
    verdict = _derive_verdict(results, counts, snapshot_ok or snapshot_unknown)

    buckets: dict[str, list[CheckResult]] = {"OK": [], "WARN": [], "FAIL": [], "STALE": [], "SKIP": []}
    for r in results:
        buckets.setdefault(r.status, []).append(r)

    lines: list[str] = []
    lines.append(f"# Fleet Receipt — {ds}\n")
    lines.append(f"\n**Verdict: {verdict}**\n")
    lines.append(
        f"\nGenerated by `tools/agent_heartbeat_checks.py` at {datetime.now().isoformat(timespec='seconds')}\n"
    )

    lines.append("\n## Pipeline\n")
    if snapshot_unknown:
        snapshot_status = "UNKNOWN_CLOUD_ENV"
    else:
        snapshot_status = "OK" if snapshot_ok else "MISSING"
    lines.append(f"- Today's snapshot ({ds}): {snapshot_status}\n")
    lines.append(f"- Previous snapshot: {prev_snapshot if prev_snapshot else 'none found'}\n")

    lines.append("\n## Fleet (AGENT_REGISTRY.json)\n")
    lines.append(f"- Active: {counts.get('active_count', 0)}\n")
    lines.append(f"- Supervised this run: {counts.get('monitored_count', 0)}\n")
    lines.append(f"- Coverage gap (active but unsupervised): {counts.get('missing_count', 0)}\n")
    lines.append(f"- Deprecated: {counts.get('deprecated_count', 0)}\n")
    lines.append(f"- Stale artifacts: {counts.get('stale_count', 0)}\n")

    lines.append(f"\n## Agent Status ({len(buckets['OK'])}/{len(results)} OK)\n")
    for status in ("FAIL", "WARN", "STALE", "SKIP", "OK"):
        entries = buckets.get(status, [])
        if not entries:
            continue
        lines.append(f"\n### {status} ({len(entries)})\n")
        for r in sorted(entries, key=lambda x: x.agent):
            detail = f" — {r.detail}" if r.detail else ""
            lines.append(f"- **{r.agent}**{detail}\n")
            for a in r.anomalies:
                lines.append(f"  - {a}\n")

    escalated = [r for r in results if r.needs_llm]
    if escalated:
        lines.append(f"\n## Escalated to ops ({len(escalated)})\n")
        for r in escalated:
            lines.append(f"- **{r.agent}** ({r.status}): {r.detail}\n")

    lines.append(
        "\n---\n"
        "_This is a deterministic status receipt. Historical receipts written by the "
        "fleet_steward LLM agent also included: carried-issues history, performance "
        "summary, accumulation-gate progress, and escalation tracker — those require "
        "analyst synthesis and are not generated here._\n"
    )

    out_path.write_text("".join(lines))
    return out_path


def _heartbeat_llm_escalate_enabled() -> bool:
    """Return true when operator opts into optional LLM escalation."""
    return os.environ.get("HEARTBEAT_LLM_ESCALATE", "").strip().lower() in ("1", "true", "yes")


def _write_anomaly_file(results: list[CheckResult], dt: date) -> Path:
    """Write anomaly summary to local file as durable record."""
    ds = as_of_date(dt)
    heartbeat_dir = ARTIFACTS_DIR / "heartbeat"
    heartbeat_dir.mkdir(parents=True, exist_ok=True)
    out_path = heartbeat_dir / f"{ds}_anomalies.md"

    lines = [f"# Heartbeat Anomalies — {ds}\n"]
    lines.append(f"Generated: {datetime.now().isoformat()}\n")
    for r in results:
        lines.append(f"\n## [{r.agent}] {r.status} — {r.detail}\n")
        for a in r.anomalies:
            lines.append(f"- {a}\n")
    out_path.write_text("".join(lines))
    print(f"  Anomaly summary written to {out_path}")
    return out_path


def _write_escalation_json(
    anomaly_results: list[CheckResult],
    dt: date,
    *,
    mode: str,
    llm_requested: bool,
    llm_status: str | None = None,
) -> Path:
    """Write structured escalation receipt for ops_supervisor / fleet_ops_status."""
    ds = as_of_date(dt)
    heartbeat_dir = ARTIFACTS_DIR / "heartbeat"
    heartbeat_dir.mkdir(parents=True, exist_ok=True)
    out_path = heartbeat_dir / f"{ds}_escalation.json"

    payload = {
        "schema": "heartbeat_escalation.v1",
        "as_of_date": ds,
        "generated_at": datetime.now().isoformat(),
        "mode": mode,
        "llm_requested": llm_requested,
        "llm_status": llm_status,
        "agent_count": len(anomaly_results),
        "agents": [
            {
                "agent": r.agent,
                "status": r.status,
                "detail": r.detail,
                "anomalies": r.anomalies,
            }
            for r in anomaly_results
        ],
        "artifacts": {
            "anomalies_md": f"artifacts/heartbeat/{ds}_anomalies.md",
            "escalation_json": f"artifacts/heartbeat/{ds}_escalation.json",
        },
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"  Escalation receipt written to {out_path}")
    return out_path


def _build_llm_summary(anomaly_results: list[CheckResult]) -> str:
    summary = "ANOMALIES DETECTED — interpret and recommend action:\n\n"
    for r in anomaly_results:
        summary += f"[{r.agent}] {r.status}: {r.detail}\n"
        for a in r.anomalies:
            summary += f"  - {a}\n"
        summary += "\n"
    summary += "For each anomaly: (1) is it actionable? (2) what's the fix? (3) severity 1-5?"
    return summary


def _invoke_llm_escalation(summary: str) -> str:
    """Run optional ops LLM escalation; return status token for escalation receipt."""
    try:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "run_agent_direct.py"), "--agent", "ops", "--message", summary],
            capture_output=True,
            text=True,
            timeout=150,
            cwd=str(REPO_ROOT),
        )
        response_text = (result.stdout or "").strip()
        if response_text:
            if "rejected" in response_text.lower() or "credit" in response_text.lower():
                print("\n  LLM escalation rejected (API billing). Anomalies saved to artifacts/heartbeat/.")
                return "rejected"
            print(f"\n  LLM response:\n{response_text}")
            return "ok"
        if result.returncode != 0:
            err = (result.stderr or "").strip()
            print(f"\n  LLM escalation failed (exit {result.returncode}): {err}")
            return "failed"
        return "empty"
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"\n  LLM escalation failed: {e}. Anomalies saved to artifacts/heartbeat/.")
        return "failed"


def escalate_anomalies(results: list[CheckResult], dry_run: bool = False, dt: date | None = None):
    """Write anomaly artifacts; optionally invoke LLM when HEARTBEAT_LLM_ESCALATE=1."""
    anomaly_results = [r for r in results if r.needs_llm]
    if not anomaly_results:
        return

    if dt is None:
        return

    _write_anomaly_file(anomaly_results, dt)
    llm_enabled = _heartbeat_llm_escalate_enabled()
    summary = _build_llm_summary(anomaly_results)

    if dry_run:
        mode = "dry_run"
        if llm_enabled:
            print(f"\n  Would escalate to LLM (HEARTBEAT_LLM_ESCALATE=1):\n  {summary}")
        else:
            print(
                f"\n  Artifact-only escalation for {len(anomaly_results)} agent(s) "
                "(set HEARTBEAT_LLM_ESCALATE=1 for optional LLM)."
            )
        _write_escalation_json(
            anomaly_results,
            dt,
            mode=mode,
            llm_requested=llm_enabled,
            llm_status="skipped_dry_run" if llm_enabled else None,
        )
        return

    llm_status: str | None = None
    mode = "artifact_only"
    if llm_enabled:
        print(f"\n  Escalating {len(anomaly_results)} agent(s) to LLM (HEARTBEAT_LLM_ESCALATE=1)...")
        llm_status = _invoke_llm_escalation(summary)
        mode = "llm"
    else:
        print(
            f"\n  Artifact-only escalation for {len(anomaly_results)} agent(s) "
            "(set HEARTBEAT_LLM_ESCALATE=1 for optional LLM)."
        )

    _write_escalation_json(
        anomaly_results,
        dt,
        mode=mode,
        llm_requested=llm_enabled,
        llm_status=llm_status,
    )


def escalate_to_llm(results: list[CheckResult], dry_run: bool = False, dt: date | None = None):
    """Backward-compatible alias for escalate_anomalies."""
    escalate_anomalies(results, dry_run=dry_run, dt=dt)


def main():
    parser = argparse.ArgumentParser(description="Tier 2 agent heartbeat checks")
    parser.add_argument("--agent", choices=list(AGENTS.keys()), help="Run single agent check")
    parser.add_argument("--date", type=str, help="Check date (YYYY-MM-DD), default today")
    parser.add_argument("--dry-run", action="store_true", help="Print anomalies, don't escalate")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    dt = date.fromisoformat(args.date) if args.date else date.today()

    print(f"Heartbeat checks for {as_of_date(dt)}")
    print(f"{'=' * 50}")

    counts: dict | None = None
    if args.agent:
        try:
            result = AGENTS[args.agent](dt)
        except Exception as e:  # noqa: BLE001
            result = CheckResult(args.agent, "FAIL", f"Check crashed: {e}", [f"EXCEPTION: {e}"])
        results = [result]
        print(result)
    else:
        results, counts = run_registry_checks(dt)
        for r in results:
            print(r)

    # Summary
    ok = sum(1 for r in results if r.status == "OK")
    warn = sum(1 for r in results if r.status == "WARN")
    fail = sum(1 for r in results if r.status == "FAIL")
    stale = sum(1 for r in results if r.status == "STALE")
    skip = sum(1 for r in results if r.status == "SKIP")
    total_anomalies = sum(len(r.anomalies) for r in results)

    print(f"\n  Summary: {ok} OK, {warn} WARN, {fail} FAIL, {stale} STALE, {skip} SKIP — {total_anomalies} anomalies")

    coverage_gap = False
    if counts is not None:
        print(
            f"  Registry: active={counts['active_count']}, monitored={counts['monitored_count']}, "
            f"stale={counts['stale_count']}, missing={counts['missing_count']}, "
            f"deprecated={counts['deprecated_count']}"
        )
        if counts["missing_count"] > 0:
            coverage_gap = True
            unsupervised = [r.agent for r in results if r.status == "SKIP"]
            print(
                f"  ⚠ COVERAGE GAP: {counts['missing_count']} active agent(s) not supervised — "
                f"{', '.join(unsupervised)}"
            )

        # Restore the daily fleet_steward receipt (Fix #3). Registry-mode only.
        receipt_path = write_fleet_receipt(results, counts, dt)
        print(f"  Fleet receipt: {receipt_path.relative_to(REPO_ROOT)}")

    if args.json:
        out: dict = {
            "results": [
                {"agent": r.agent, "status": r.status, "detail": r.detail, "anomalies": r.anomalies} for r in results
            ]
        }
        if counts is not None:
            out["counts"] = counts
        print(json.dumps(out, indent=2))

    # Escalate actionable anomalies (specialized checks only; generic SKIP/STALE is artifact-level)
    escalatable = [r for r in results if r.needs_llm]
    if escalatable:
        escalate_anomalies(results, dry_run=args.dry_run, dt=dt)
    elif total_anomalies > 0:
        print("  Anomalies present but none require escalation (carried-only or non-actionable).")
    else:
        print("  No anomalies — escalation not needed.")

    sys.exit(1 if fail > 0 or coverage_gap else 0)


if __name__ == "__main__":
    main()
