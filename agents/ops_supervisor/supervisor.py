#!/usr/bin/env python3
"""ops_supervisor — daily ops triage.

Read-only. Reads heartbeat anomalies + ops_digest + production artifacts +
prior-day supervisor JSON; classifies each anomaly as new/carried/resolved/
worsened/expected; emits one daily verdict (GREEN/YELLOW/ORANGE/RED) +
one action (no_action/watch/investigate/fix_now).

Architecture position: agents → heartbeat monitor → ops_supervisor →
sentinel. This is the LAST interpretive layer (per
feedback_no_recursive_supervision.md).

Outputs:
  artifacts/ops_supervisor/{as_of_date}_supervisor.json
  artifacts/ops_supervisor/{as_of_date}_supervisor.md

Exit codes: 0=GREEN/YELLOW, 1=ORANGE, 2=RED.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
HEARTBEAT_DIR = REPO / "artifacts" / "heartbeat"
OPS_DIGEST_DIR = REPO / "artifacts" / "ops_digest"
SUPERVISOR_DIR = REPO / "artifacts" / "ops_supervisor"
SNAP_DIR = REPO / "data" / "snapshots"
SEC_CACHE_DIR = REPO / "cache" / "sec" / "8k_catalysts"
REGISTRY = REPO / "agents" / "AGENT_REGISTRY.json"

# Production-due gate (24h clock, ET). After this hour, missing rankings.csv = RED.
PRODUCTION_DUE_HOUR_ET = 18

# Runtime-health job classification. Times are local (ET on this host).
# Source: crontab -l weekday-only entries.
NON_CRITICAL_JOB_TIMES_ET = [
    (6, 30),  # bellringer
    (7, 30),  # herald pre-morning fetch
    (8, 0),  # news digest morning
    (15, 0),  # news digest midday
]
PRODUCTION_CRITICAL_JOB_TIMES_ET = [
    (14, 0),  # data refresh — feeds production
    (16, 30),  # cron_daily_production.sh — main pipeline
    (16, 45),  # PIT archiver
    (17, 30),  # tier-2 heartbeat checks
    (18, 0),  # ops digest, data_auditor, agent stagger
    (18, 55),  # production_qa
]

# ---------------------------------------------------------------------------
# Exception table — canonical source of truth for known/expected anomalies
# ---------------------------------------------------------------------------
# Each rule:
#   id: stable identifier
#   match: callable(agent, raw_status, raw_text) -> bool
#   classify: function returning supervisor_severity + reason
#   expires_after: ISO date — after this, the rule yields ORANGE not YELLOW
EXCEPTIONS: list[dict] = [
    {
        "id": "inst_delta_z_signal_alert",
        "agent": "ic_health_monitor",
        "match_substring": "inst_delta_z",
        "expires_after": "2026-05-15",
        "yellow_reason": (
            "Expected: inst_delta_z byte-identical 04-25 → 04-28 due to 13F cohort rebuild; "
            "self-heal at next 13F refresh (~2026-05-15). See "
            "regime_post_cohort_change_distortion_2026_04_28.md."
        ),
        "orange_reason": (
            "inst_delta_z SIGNAL_ALERT persists past expected 2026-05-15 13F-refresh "
            "self-heal date. Investigate 13F ingest health."
        ),
    },
    {
        "id": "calibration_evidence_stale",
        "agent": "calibration_evidence",
        "match_substring": "STALE",
        "expires_after": "2026-05-01",
        "yellow_reason": (
            "Single missed Friday cron (2026-04-24). Next scheduled fire 2026-05-01 19:00 ET. " "Watch only until then."
        ),
        "orange_reason": (
            "calibration_evidence still stale after 2026-05-01 19:00 ET retry gate. " "Investigate cron reliability."
        ),
    },
    {
        "id": "phase2_fail_carried",
        "agent": "qa",
        "match_substring": "PHASE2_FAIL",
        "expires_after": None,  # Indefinitely tied to model state, not a date
        "yellow_reason": (
            "Carried governance state (catalyst_7d_count_high, bucket_drift, "
            "health_gate_stability). Verify decision_diff is not materially worsening; "
            "see ops_digest carried-flags."
        ),
        "orange_reason": (
            "PHASE2_FAIL with decision_diff Spearman <0.95 OR Top-30 overlap <80%. " "Material decision change."
        ),
        "decision_diff_check": True,  # supervisor will inspect ops_digest decision diff
    },
    {
        "id": "shadow_monitor_perf_alert",
        "agent": "shadow_monitor",
        "match_substring": "PERF_ALERT",
        "expires_after": None,
        "yellow_reason": "Informational WARN. No action.",
        "orange_reason": None,  # Never escalates by date
    },
]

# Per-agent suppression reasons. `in SUPPRESSED_AGENTS` still works (dict
# membership), so existing callers do not need to change.
# Retired overlapping agents removed from fleet 2026-05-30 (registry + agents/ dirs deleted).
SUPPRESSED_AGENTS: dict[str, str] = {}
SUPPRESS_AGENT_PATTERNS = ["massive"]  # paused per license downgrade


def load_registry() -> dict:
    if not REGISTRY.exists():
        return {}
    try:
        return json.load(open(REGISTRY))
    except Exception:
        return {}


def parse_heartbeat_anomalies_md(path: Path) -> list[dict]:
    """Parse the heartbeat anomalies markdown into structured records.

    Format (from tools/agent_heartbeat_checks.py):
      ## [agent] STATUS — N issue(s)
      - CODE: detail
      - CODE: detail
    """
    if not path.exists():
        return []
    text = path.read_text()
    anomalies = []
    blocks = re.split(r"^## \[", text, flags=re.MULTILINE)[1:]
    for block in blocks:
        # block now starts with "agent_name] STATUS — ..."
        m = re.match(r"([^\]]+)\]\s+(\w+)\s+—\s+([^\n]+)", block)
        if not m:
            continue
        agent, status, header_detail = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        body_lines = [line.strip("- ").strip() for line in block.split("\n")[1:] if line.strip().startswith("-")]
        anomalies.append(
            {
                "agent": agent,
                "raw_status": status,
                "header": header_detail,
                "issues": body_lines,
                "raw_text": header_detail + " | " + " | ".join(body_lines),
            }
        )
    return anomalies


def load_prior_supervisor(prior_date_iso: str) -> dict | None:
    p = SUPERVISOR_DIR / f"{prior_date_iso}_supervisor.json"
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def find_prior_supervisor(today_iso: str) -> tuple[dict | None, str | None]:
    """Most recent supervisor JSON strictly before today_iso. Returns (data, date_iso)."""
    if not SUPERVISOR_DIR.exists():
        return None, None
    candidates = sorted(SUPERVISOR_DIR.glob("*_supervisor.json"))
    eligible = [p for p in candidates if (p.name.replace("_supervisor.json", "")) < today_iso]
    if not eligible:
        return None, None
    p = eligible[-1]
    try:
        return json.load(open(p)), p.name.replace("_supervisor.json", "")
    except Exception:
        return None, p.name.replace("_supervisor.json", "")


def is_suppressed(agent: str, registry: dict) -> tuple[bool, str | None]:
    if agent in SUPPRESSED_AGENTS:
        return True, SUPPRESSED_AGENTS[agent]
    for pat in SUPPRESS_AGENT_PATTERNS:
        if pat in agent.lower():
            return True, f"agent paused (pattern '{pat}', license/policy)"
    reg_agents = registry.get("agents") or {}
    entry = reg_agents.get(agent, {})
    if entry.get("status") == "deprecated":
        return True, "agent deprecated per registry"
    return False, None


def match_exception(agent: str, raw_text: str) -> dict | None:
    for rule in EXCEPTIONS:
        if rule.get("agent") and rule["agent"] != agent:
            continue
        sub = rule.get("match_substring", "")
        if sub and sub.upper() in raw_text.upper():
            return rule
    return None


def classify_anomaly(
    a: dict,
    today: date,
    prior_anomalies: list[dict],
    registry: dict,
    digest_decision_diff: dict | None,
) -> dict:
    suppressed, supp_reason = is_suppressed(a["agent"], registry)
    if suppressed:
        return {
            "id": f"{a['agent']}_{a['raw_status']}",
            "agent": a["agent"],
            "raw_status": a["raw_status"],
            "category": "suppressed",
            "classification": "suppressed",
            "expected_resolution": None,
            "supervisor_severity": "SUPPRESSED",
            "reason": supp_reason,
            "fix_prompt": None,
        }

    # Did this exact anomaly appear yesterday?
    prior_match = next(
        (
            p
            for p in prior_anomalies
            if p.get("agent") == a["agent"]
            and p.get("raw_status") == a["raw_status"]
            and p.get("raw_text") == a.get("raw_text")
        ),
        None,
    )
    if prior_match is None:
        delta = "new"
    else:
        delta = "carried"

    # Try the exception table
    rule = match_exception(a["agent"], a["raw_text"])
    if rule:
        expires = rule.get("expires_after")
        past_expiry = bool(expires) and (today.isoformat() > expires)
        if rule["id"] == "phase2_fail_carried" and digest_decision_diff is not None:
            spearman = digest_decision_diff.get("spearman_rho")
            top30 = digest_decision_diff.get("top30_overlap_pct")
            material_worsening = (spearman is not None and spearman < 0.95) or (top30 is not None and top30 < 80.0)
        else:
            material_worsening = False
        if past_expiry or material_worsening:
            severity = "ORANGE"
            reason = rule.get("orange_reason") or "exception expired"
        else:
            severity = "YELLOW"
            reason = rule["yellow_reason"]
        return {
            "id": rule["id"],
            "agent": a["agent"],
            "raw_status": a["raw_status"],
            "category": "known_exception",
            "classification": delta if not past_expiry else "expected_until_date_passed",
            "expected_resolution": expires,
            "supervisor_severity": severity,
            "reason": reason,
            "fix_prompt": _fix_prompt_for(rule["id"], a) if severity == "ORANGE" else None,
        }

    # No exception match → unknown anomaly
    if delta == "new":
        return {
            "id": f"{a['agent']}_unknown_new",
            "agent": a["agent"],
            "raw_status": a["raw_status"],
            "category": "unknown",
            "classification": "new",
            "expected_resolution": None,
            "supervisor_severity": "ORANGE",
            "reason": (
                f"New unclassified anomaly: {a['raw_text'][:200]}. " f"Not matched by any exception rule. Investigate."
            ),
            "fix_prompt": _generic_fix_prompt(a),
        }
    else:
        return {
            "id": f"{a['agent']}_unknown_carried",
            "agent": a["agent"],
            "raw_status": a["raw_status"],
            "category": "unknown",
            "classification": "carried",
            "expected_resolution": None,
            "supervisor_severity": "YELLOW",
            "reason": (
                f"Carried unclassified anomaly (also present {prior_match.get('classification', 'prior day') if prior_match else 'previously'}). "
                "Add an exception rule to supervisor.py if intentional."
            ),
            "fix_prompt": None,
        }


def _fix_prompt_for(rule_id: str, anomaly: dict) -> str:
    if rule_id == "inst_delta_z_signal_alert":
        return (
            "Verify production_data/institutional_summary.json mtime advanced past "
            "2026-05-15 (13F refresh). If not, investigate 13F ingest pipeline."
        )
    if rule_id == "calibration_evidence_stale":
        return (
            "Inspect logs/calibration_evidence.log for the 2026-05-01 19:00 ET fire. "
            "If absent, debug WSL2 cron daemon health."
        )
    if rule_id == "phase2_fail_carried":
        return (
            "Read the latest ops_digest/{date}_digest.md decision_diff section. "
            "If Spearman ρ < 0.95 OR Top-30 overlap < 80%, dig into the new entrants/exits."
        )
    return f"Investigate {anomaly['agent']} {anomaly['raw_status']}."


def _generic_fix_prompt(a: dict) -> str:
    return (
        f"New anomaly from {a['agent']}: {a['raw_status']}. "
        f"Read artifacts/heartbeat/{{date}}_anomalies.md for detail. "
        "If intentional, add an exception rule to "
        "agents/ops_supervisor/supervisor.py EXCEPTIONS."
    )


def check_production_artifacts(today: date) -> dict:
    """Verify required daily artifacts exist for today."""
    iso = today.isoformat()
    rankings = SNAP_DIR / iso / "rankings.csv"
    manifest = SNAP_DIR / iso / "run_manifest.json"
    sec_cache_glob = list(SEC_CACHE_DIR.glob(f"8k_catalysts_{iso}_*.json"))
    return {
        "today_rankings_csv": "found" if rankings.exists() else "missing",
        "today_run_manifest": "found" if manifest.exists() else "missing",
        "today_8k_cache": "found" if sec_cache_glob else "missing",
    }


def extract_decision_diff_from_digest(digest_json: dict | None) -> dict | None:
    if not digest_json:
        return None
    # ops_digest schema isn't strictly fixed; do best-effort extraction
    for key in ("decision_diff", "decisionDiff", "diff"):
        d = digest_json.get(key)
        if isinstance(d, dict):
            spearman = d.get("spearman_rho") or d.get("spearman")
            top30 = d.get("top_20_overlap_pct") or d.get("top_30_overlap_pct") or d.get("top30_overlap_pct")
            return {"spearman_rho": spearman, "top30_overlap_pct": top30}
    return None


def _run_subproc(cmd: list[str], timeout: float = 5.0) -> tuple[str | None, int]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.returncode
    except Exception:
        return None, 1


def _parse_systemctl_timestamp(s: str) -> datetime | None:
    """Parse `Tue 2026-04-28 22:22:45 EDT` → naive local datetime.

    Robust to missing day-name, missing TZ, or 'n/a' (inactive service).
    """
    if not s:
        return None
    parts = s.strip().split()
    date_str, time_str = None, None
    for p in parts:
        if "-" in p and date_str is None and len(p) == 10:
            date_str = p
        elif ":" in p and time_str is None:
            time_str = p
    if not date_str or not time_str:
        return None
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def check_runtime_health(today: date, now_dt: datetime, input_status: dict) -> dict:
    """Inspect cron service + WSL/runtime uptime. Detect missed jobs.

    Classification (per spec):
      GREEN  — cron active and active_since on/before today's first job, no misses
      YELLOW — cron active but came up after some non-critical jobs only
      ORANGE — cron came up after production-critical job times (artifacts may be recovered)
      RED    — cron inactive, OR a production job missed AND artifact missing
    """
    out: dict = {
        "cron_active": None,
        "cron_active_since_raw": None,
        "cron_active_since_parsed": None,
        "system_boot_time": None,
        "first_job_time_today_et": None,
        "production_critical_job_times_today_et": [f"{h:02d}:{m:02d}" for h, m in PRODUCTION_CRITICAL_JOB_TIMES_ET],
        "cron_active_before_first_job": None,
        "system_restarted_after_scheduled_jobs_today": None,
        "missed_critical_job_times": [],
        "missed_noncritical_job_times": [],
        "missed_jobs_correlate_with_downtime": None,
        "runtime_alive_for_scheduled_window": "unknown",
        "severity": "GREEN",
        "reasons": [],
    }

    is_weekday = today.weekday() < 5

    # 1) cron active state
    stdout, _rc = _run_subproc(["systemctl", "is-active", "cron"])
    if stdout is None:
        out["reasons"].append("systemctl unavailable; runtime health unknown")
        out["runtime_alive_for_scheduled_window"] = "unknown"
        return out
    out["cron_active"] = stdout
    if stdout != "active":
        out["severity"] = "RED"
        out["runtime_alive_for_scheduled_window"] = "no"
        out["reasons"].append(f"cron service is `{stdout}` (not active)")
        return out

    # 2) cron active_since timestamp
    stdout, _rc = _run_subproc(["systemctl", "show", "cron", "--property=ActiveEnterTimestamp"])
    cron_active_since_dt = None
    if stdout and "=" in stdout:
        ts_str = stdout.split("=", 1)[1].strip()
        out["cron_active_since_raw"] = ts_str
        cron_active_since_dt = _parse_systemctl_timestamp(ts_str)
        if cron_active_since_dt is not None:
            out["cron_active_since_parsed"] = cron_active_since_dt.isoformat()

    # 3) System boot time (WSL/runtime)
    stdout, _rc = _run_subproc(["uptime", "-s"])
    out["system_boot_time"] = stdout
    system_boot_dt = None
    if stdout:
        try:
            system_boot_dt = datetime.strptime(stdout, "%Y-%m-%d %H:%M:%S")
        except Exception:
            system_boot_dt = None

    # 4) Weekend short-circuit — production cron does not fire
    if not is_weekday:
        out["runtime_alive_for_scheduled_window"] = "yes"
        out["severity"] = "GREEN"
        out["reasons"].append("weekend — production cron not scheduled today")
        return out

    # 5) Today's first scheduled job (anchor for GREEN)
    first_h, first_m = NON_CRITICAL_JOB_TIMES_ET[0]
    midnight = datetime.combine(today, datetime.min.time())
    first_job_dt = midnight.replace(hour=first_h, minute=first_m)
    out["first_job_time_today_et"] = first_job_dt.strftime("%Y-%m-%d %H:%M")

    if cron_active_since_dt is not None:
        out["cron_active_before_first_job"] = cron_active_since_dt <= first_job_dt

    # 6) WSL restart after any scheduled job that should have fired by now?
    if system_boot_dt is not None and system_boot_dt.date() == today:
        all_times = NON_CRITICAL_JOB_TIMES_ET + PRODUCTION_CRITICAL_JOB_TIMES_ET
        for h, m in all_times:
            job_dt = midnight.replace(hour=h, minute=m)
            if job_dt < now_dt and system_boot_dt > job_dt:
                out["system_restarted_after_scheduled_jobs_today"] = True
                break
        else:
            out["system_restarted_after_scheduled_jobs_today"] = False
    else:
        out["system_restarted_after_scheduled_jobs_today"] = False

    # 7) Which jobs missed normal cron firing?
    # A job is "missed" if its scheduled time is past AND cron came up after it.
    if cron_active_since_dt is not None and cron_active_since_dt.date() == today:
        for h, m in PRODUCTION_CRITICAL_JOB_TIMES_ET:
            job_dt = midnight.replace(hour=h, minute=m)
            if job_dt < now_dt and cron_active_since_dt > job_dt:
                out["missed_critical_job_times"].append(f"{h:02d}:{m:02d}")
        for h, m in NON_CRITICAL_JOB_TIMES_ET:
            job_dt = midnight.replace(hour=h, minute=m)
            if job_dt < now_dt and cron_active_since_dt > job_dt:
                out["missed_noncritical_job_times"].append(f"{h:02d}:{m:02d}")

    # 8) Correlate missed jobs with missing production artifacts
    rankings_missing = input_status.get("today_rankings_csv") == "missing"
    manifest_missing = input_status.get("today_run_manifest") == "missing"
    artifact_missing = rankings_missing or manifest_missing
    missed_critical = out["missed_critical_job_times"]
    missed_noncritical = out["missed_noncritical_job_times"]

    if missed_critical and artifact_missing:
        out["missed_jobs_correlate_with_downtime"] = "yes"
    elif missed_critical and not artifact_missing:
        out["missed_jobs_correlate_with_downtime"] = "no_artifact_recovered"
    else:
        out["missed_jobs_correlate_with_downtime"] = "no"

    # 9) Severity
    cron_pre_first = out["cron_active_before_first_job"]
    if missed_critical and artifact_missing:
        out["severity"] = "RED"
        out["runtime_alive_for_scheduled_window"] = "no"
        out["reasons"].append(
            f"production-critical job(s) at {missed_critical} ET missed; "
            f"rankings/manifest absent — runtime down during production window"
        )
    elif missed_critical:
        out["severity"] = "ORANGE"
        out["runtime_alive_for_scheduled_window"] = "no"
        out["reasons"].append(
            f"cron came up at {out['cron_active_since_raw']} — after production-critical "
            f"job time(s) {missed_critical} ET. Artifacts present (likely watchdog/@reboot recovery); verify."
        )
    elif missed_noncritical:
        out["severity"] = "YELLOW"
        out["runtime_alive_for_scheduled_window"] = "no"
        out["reasons"].append(
            f"cron came up after non-critical job time(s) {missed_noncritical} ET; " f"production window unaffected"
        )
    elif cron_pre_first is True:
        out["severity"] = "GREEN"
        out["runtime_alive_for_scheduled_window"] = "yes"
        out["reasons"].append("cron active before today's first scheduled job and still active")
    elif cron_pre_first is None:
        out["severity"] = "GREEN"
        out["runtime_alive_for_scheduled_window"] = "unknown"
        out["reasons"].append("cron active but ActiveEnterTimestamp could not be parsed")
    else:
        # cron_pre_first is False but no job times have fired yet (very early in the day)
        out["severity"] = "GREEN"
        out["runtime_alive_for_scheduled_window"] = "yes"
        out["reasons"].append("cron active; no scheduled jobs have fired yet today")

    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--as-of", default=None, help="Override as-of date (YYYY-MM-DD).")
    p.add_argument("--force-now-hour", type=int, default=None, help="Override 'now' hour for due-time gate.")
    args = p.parse_args()

    as_of = args.as_of or datetime.now().date().isoformat()
    today = date.fromisoformat(as_of)
    now_dt = datetime.now()
    now_hour = args.force_now_hour if args.force_now_hour is not None else now_dt.hour

    SUPERVISOR_DIR.mkdir(parents=True, exist_ok=True)

    # -- Load inputs --
    heartbeat_path = HEARTBEAT_DIR / f"{as_of}_anomalies.md"
    digest_md = OPS_DIGEST_DIR / f"{as_of}_digest.md"
    digest_json_path = OPS_DIGEST_DIR / f"{as_of}_digest.json"

    input_status = {
        "heartbeat_anomalies_md": "found" if heartbeat_path.exists() else "missing",
        "ops_digest_md": "found" if digest_md.exists() else "missing",
        "ops_digest_json": "found" if digest_json_path.exists() else "missing",
    }
    artifact_status = check_production_artifacts(today)
    input_status.update(artifact_status)

    digest_json = None
    if digest_json_path.exists():
        try:
            digest_json = json.load(open(digest_json_path))
        except Exception:
            input_status["ops_digest_json"] = "malformed"

    anomalies_raw = parse_heartbeat_anomalies_md(heartbeat_path)
    prior, prior_date = find_prior_supervisor(as_of)
    input_status["prior_supervisor_json"] = "found" if prior else "missing"
    input_status["prior_supervisor_date"] = prior_date

    prior_anomalies = []
    if prior:
        for a in prior.get("anomalies", []):
            prior_anomalies.append(
                {
                    "agent": a.get("agent"),
                    "raw_status": a.get("raw_status"),
                    "raw_text": a.get("raw_text"),
                    "classification": a.get("classification"),
                }
            )

    registry = load_registry()
    decision_diff = extract_decision_diff_from_digest(digest_json)

    # -- Runtime health (cron service + WSL/runtime uptime) --
    runtime_health = check_runtime_health(today, now_dt, input_status)

    # -- Classify each anomaly --
    classified = [classify_anomaly(a, today, prior_anomalies, registry, decision_diff) for a in anomalies_raw]

    # -- Compute final severity --
    final_severity = "GREEN"
    final_action = "no_action"

    # Fail-closed checks first
    monitor_input_missing = (
        input_status["heartbeat_anomalies_md"] == "missing" and input_status["ops_digest_md"] == "missing"
    )
    rankings_missing_past_due = (
        input_status["today_rankings_csv"] == "missing"
        and now_hour >= PRODUCTION_DUE_HOUR_ET
        and today.weekday() < 5  # weekday only
    )

    if monitor_input_missing:
        final_severity = "RED"
        final_action = "fix_now"
    elif rankings_missing_past_due:
        final_severity = "RED"
        final_action = "fix_now"
    else:
        # Roll up severities (anomalies + runtime health)
        sevs_seen = set(c["supervisor_severity"] for c in classified)
        sevs_seen.add(runtime_health["severity"])
        if "RED" in sevs_seen:
            final_severity = "RED"
            final_action = "fix_now"
        elif "ORANGE" in sevs_seen:
            final_severity = "ORANGE"
            final_action = "investigate"
        elif "YELLOW" in sevs_seen:
            final_severity = "YELLOW"
            final_action = "watch"
        else:
            final_severity = "GREEN"
            final_action = "no_action"

    fix_prompts = [c["fix_prompt"] for c in classified if c.get("fix_prompt")]

    # Build summary line
    if final_severity == "GREEN":
        summary = f"GREEN — all clean ({len(classified)} anomalies, all suppressed/expected)."
    elif final_severity == "YELLOW":
        n_known = sum(1 for c in classified if c["supervisor_severity"] == "YELLOW")
        summary = f"YELLOW — {n_known} known/expected anomalies; watch only."
    elif final_severity == "ORANGE":
        n_orange = sum(1 for c in classified if c["supervisor_severity"] == "ORANGE")
        summary = f"ORANGE — {n_orange} new or expired-window anomalies; investigate."
    else:
        if monitor_input_missing:
            summary = "RED — monitoring layer unreachable. Heartbeat + ops_digest both missing."
        elif rankings_missing_past_due:
            summary = (
                f"RED — rankings.csv missing for {as_of} past production-due-time " f"({PRODUCTION_DUE_HOUR_ET}:00 ET)."
            )
        else:
            summary = "RED — see anomalies."

    # -- Emit JSON --
    out = {
        "schema": "ops_supervisor.v1",
        "as_of_date": as_of,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input_status": input_status,
        "runtime_health": runtime_health,
        "anomalies": classified,
        "agent_count": len(classified),
        "checked_items_count": len(classified) + len(artifact_status),
        "final_severity": final_severity,
        "final_action": final_action,
        "summary_one_line": summary,
        "fix_prompts": fix_prompts,
        "exception_table_version": "2026-04-28",
    }

    out_json = SUPERVISOR_DIR / f"{as_of}_supervisor.json"
    with open(out_json, "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    # Phase B: cron_missed → Town when runtime health shows missed critical windows
    try:
        from common.town_bridge_events import notify_cron_missed_from_runtime_health

        notify_cron_missed_from_runtime_health(
            as_of,
            runtime_health,
            artifact=str(out_json.relative_to(REPO)),
        )
    except Exception:
        pass

    # -- Emit Markdown --
    lines = [
        f"# Ops Supervisor — {as_of}",
        "",
        f"**Final severity**: `{final_severity}`",
        f"**Final action**: `{final_action}`",
        "",
        f"> {summary}",
        "",
        "## Input status",
        "",
    ]
    for k, v in input_status.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")

    # Runtime health section
    rh = runtime_health
    lines.append("## Runtime health")
    lines.append("")
    lines.append(f"- **cron_active**: `{rh['cron_active']}`")
    lines.append(f"- **cron_active_since**: `{rh['cron_active_since_raw'] or 'unknown'}`")
    lines.append(f"- **system_boot_time**: `{rh['system_boot_time'] or 'unknown'}`")
    lines.append(f"- **first_job_time_today_et**: `{rh['first_job_time_today_et'] or 'n/a'}`")
    lines.append(f"- **cron_active_before_first_job**: `{rh['cron_active_before_first_job']}`")
    lines.append(
        f"- **system_restarted_after_scheduled_jobs_today**: " f"`{rh['system_restarted_after_scheduled_jobs_today']}`"
    )
    lines.append(f"- **missed_critical_job_times**: " f"`{rh['missed_critical_job_times'] or 'none'}`")
    lines.append(f"- **missed_noncritical_job_times**: " f"`{rh['missed_noncritical_job_times'] or 'none'}`")
    lines.append(f"- **missed_jobs_correlate_with_downtime**: " f"`{rh['missed_jobs_correlate_with_downtime']}`")
    lines.append(f"- **runtime_severity**: `{rh['severity']}`")
    if rh["reasons"]:
        for r in rh["reasons"]:
            lines.append(f"  - {r}")
    lines.append("")
    lines.append(f"> **Runtime was alive for scheduled window: {rh['runtime_alive_for_scheduled_window']}.**")
    lines.append("")

    if classified:
        lines.append("## Anomalies")
        lines.append("")
        lines.append("| Agent | Raw status | Classification | Severity | Reason |")
        lines.append("|---|---|---|---|---|")
        for c in classified:
            reason_short = (c.get("reason") or "")[:140]
            lines.append(
                f"| {c['agent']} | {c['raw_status']} | {c['classification']} | "
                f"{c['supervisor_severity']} | {reason_short} |"
            )
        lines.append("")
    if fix_prompts:
        lines.append("## Fix prompts (Claude-ready, only for ORANGE/RED items)")
        lines.append("")
        for fp in fix_prompts:
            lines.append(f"- {fp}")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Daily question this answers")
    lines.append("")
    lines.append("> *Do I need to babysit anything today?*")
    lines.append("")
    lines.append(
        f"- `{final_severity}` ⇒ **{final_action}** "
        + (
            "(no, you don't)"
            if final_severity == "GREEN"
            else (
                "(skim only)"
                if final_severity == "YELLOW"
                else "(investigate)" if final_severity == "ORANGE" else "(fix now)"
            )
        )
    )

    out_md = SUPERVISOR_DIR / f"{as_of}_supervisor.md"
    with open(out_md, "w") as fh:
        fh.write("\n".join(lines))

    # Telemetry + self-learning capture (non-blocking)
    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        exec_id = log_agent_run(
            "ops_supervisor",
            f"Daily supervisor verdict for {as_of}",
            inputs={"as_of_date": as_of},
            outputs={
                "final_severity": final_severity,
                "final_action": final_action,
                "anomaly_count": len(classified),
            },
            success=final_severity in ("GREEN", "YELLOW"),
            error=None if final_severity in ("GREEN", "YELLOW") else summary[:500],
        )
        if exec_id:
            attach_outcome_verdict(
                exec_id,
                was_correct=final_severity in ("GREEN", "YELLOW"),
                evidence=f"severity={final_severity} anomalies={len(classified)}",
            )
    except Exception:
        pass

    if final_severity in ("ORANGE", "RED"):
        try:
            corrections = REPO / ".learnings" / "corrections.md"
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            entry = (
                f"\n## [{stamp}] ops_supervisor {final_severity}\n"
                f"- action: {final_action}\n"
                f"- summary: {summary}\n"
                f"- artifact: artifacts/ops_supervisor/{as_of}_supervisor.json\n"
                f"- Promotion-lane: skill\n"
            )
            corrections.parent.mkdir(parents=True, exist_ok=True)
            with open(corrections, "a", encoding="utf-8") as fh:
                fh.write(entry)
        except OSError:
            pass

    # Console
    print(f"[ops_supervisor] {as_of} → {final_severity} ({final_action})")
    print(f"  {summary}")
    print(
        f"  Runtime was alive for scheduled window: "
        f"{runtime_health['runtime_alive_for_scheduled_window']}. "
        f"(runtime severity: {runtime_health['severity']})"
    )
    print(f"  artifact: {out_json.relative_to(REPO)}")

    return {"GREEN": 0, "YELLOW": 0, "ORANGE": 1, "RED": 2}[final_severity]


if __name__ == "__main__":
    sys.exit(main())
