#!/usr/bin/env python3
"""
build_hermes_knowledge_layer.py — Spec 089 Phase 1

Reads repo state, cron, specs, agent registry, and key artifacts.
Writes normalized ledgers to artifacts/ops/knowledge_layer/ and
artifacts/ops/first_fire_ledger/.

READ-ONLY with respect to all production files.
Does not modify code, cron, scoring, or agent registry.
"""

import json
import shutil
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
TODAY = datetime.now().strftime("%Y-%m-%d")

OUT_KL = REPO / "artifacts/ops/knowledge_layer"
OUT_HELD = REPO / "artifacts/ops/held_spec_ledger"
OUT_FF = REPO / "artifacts/ops/first_fire_ledger"
OUT_CONTRA = REPO / "artifacts/ops/contradiction_ledger"

for d in [OUT_KL, OUT_HELD, OUT_FF, OUT_CONTRA]:
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(cmd, cwd=None):
    """Run shell command, return stdout string (empty on error)."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd or str(REPO), timeout=15)
        return r.stdout.strip()
    except Exception:
        return ""


def file_age_days(path):
    """Return age of file in days, or None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    mtime = p.stat().st_mtime
    return (datetime.now().timestamp() - mtime) / 86400


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Layer 1 — Capture
# ---------------------------------------------------------------------------


def capture_git():
    head = run("git log --format='%H' -1")
    branch = run("git rev-parse --abbrev-ref HEAD")
    status_lines = run("git status --porcelain")
    uncommitted = []
    for line in status_lines.splitlines():
        if line.strip():
            uncommitted.append(line.strip())
    recent_log = run("git log --oneline -10")
    return {
        "head": head[:12] if head else "unknown",
        "branch": branch or "unknown",
        "uncommitted": uncommitted,
        "recent_log": recent_log.splitlines(),
    }


def capture_crontab():
    """Capture operator crontab when available.

    Cloud Agent VMs often lack the crontab binary. In that case return
    availability=UNKNOWN_CLOUD_ENV so contradiction checks do not treat
    a missing crontab as a hard governance failure.
    """
    if shutil.which("crontab") is None:
        return {
            "available": False,
            "availability": "UNKNOWN_CLOUD_ENV",
            "active_jobs": [],
            "suppressed_jobs": [],
        }

    raw = run("crontab -l 2>/dev/null")
    active = []
    suppressed = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            # capture SUPPRESSED markers
            if "SUPPRESSED" in stripped.upper():
                suppressed.append(stripped)
        elif stripped.startswith("SHELL=") or stripped.startswith("PATH="):
            pass
        else:
            active.append(stripped)
    return {
        "available": True,
        "availability": "OPERATOR_HOST",
        "active_jobs": active,
        "suppressed_jobs": suppressed,
    }


def capture_agent_registry():
    reg_path = REPO / "agents/AGENT_REGISTRY.json"
    data = read_json(reg_path)
    if not data:
        return {}
    agents = data.get("agents", {})
    result = {}
    for name, entry in agents.items():
        result[name] = {
            "status": entry.get("status", "unknown"),
            "category": entry.get("category", ""),
            "cadence": entry.get("cadence", ""),
            "authority_level": entry.get("authority_level", ""),
        }
    return result


def capture_specs():
    specs_dir = REPO / "specs/changes"
    specs = []
    if specs_dir.exists():
        for f in sorted(specs_dir.glob("*.md")):
            specs.append(f.name)
    return specs


def capture_key_artifacts():
    checks = {}

    # BIOSHORT_VERDICT
    verdict_path = REPO / "output/hedge_report/BIOSHORT_VERDICT.json"
    d = read_json(verdict_path)
    if d:
        checks["BIOSHORT_VERDICT"] = {
            "as_of_date": d.get("as_of_date"),
            "recommendation": d.get("recommendation"),
            "age_days": round(file_age_days(verdict_path) or 0, 1),
        }
    else:
        checks["BIOSHORT_VERDICT"] = {"status": "MISSING"}

    # Latest hedge report
    hr_dir = REPO / "output/hedge_report"
    hr_files = sorted(hr_dir.glob("hedge_report_*.json")) if hr_dir.exists() else []
    checks["latest_hedge_report"] = hr_files[-1].name if hr_files else "NONE"

    # Latest snapshot date
    snap_dir = REPO / "data/snapshots"
    dated = (
        sorted([d for d in snap_dir.iterdir() if d.is_dir() and d.name[:4].isdigit()], key=lambda x: x.name)
        if snap_dir.exists()
        else []
    )
    checks["latest_snapshot_date"] = dated[-1].name if dated else "NONE"

    # production ruleset
    ruleset_path = REPO / "production_data/decision_rulesets/v1.14.0_coinvest_only_selector.json"
    checks["active_ruleset_file_exists"] = ruleset_path.exists()

    # watchlist_current.json — generated artifact, no longer git-tracked
    # (Spec 091 follow-on, ledger 2026-05-07 §4 closure). Check freshness instead
    # of git status: cron rewrites this file on every run with as_of_date.
    wl = REPO / "data/snapshots/resolutions/watchlist_current.json"
    wl_check = {"exists": wl.exists(), "as_of_date": None, "stale_days": None}
    if wl.exists():
        try:
            wl_data = json.loads(wl.read_text())
            wl_check["as_of_date"] = wl_data.get("as_of_date")
            if wl_check["as_of_date"]:
                d = datetime.strptime(wl_check["as_of_date"], "%Y-%m-%d").date()
                wl_check["stale_days"] = (datetime.now().date() - d).days
        except (json.JSONDecodeError, ValueError, KeyError):
            wl_check["parse_error"] = True
    checks["watchlist_current_json"] = wl_check

    return checks


# ---------------------------------------------------------------------------
# Layer 2 — Held items (static seed, updated manually per session)
# ---------------------------------------------------------------------------

HELD_ITEMS_SEED = [
    {
        "id": "spec_087_b1b",
        "title": "Spec 087 B1b — bioshort weekly producer first-fire",
        "status": "AWAITING_FIRST_FIRE",
        "last_evidence": "crontab installed 2026-05-07, env-readiness cleared (07259611)",
        "blocker": "Friday 2026-05-08 18:00 EDT cron must fire; hedge_report_2026-05-08.json must exist",
        "next_allowed_action": "Read first-fire validation outputs after 2026-05-08 18:00 ET",
        "not_allowed": [
            "manual extra producer run",
            "bioshort_watch LLM reactivation",
            "any B2/B3 work before first-fire passes",
        ],
        "requires_operator_approval": True,
        "related_artifacts": [
            "output/hedge_report/hedge_report_2026-05-08.json",
            "output/hedge_report/BIOSHORT_VERDICT.json",
            "logs/biotech_hedge_report.log",
            "artifacts/audit/spec_087_b1b_env_readiness_2026_05_07.md",
        ],
        "related_cron": "0 18 * * 5  # biotech_hedge_report.py --portfolio-csv",
        "alert_condition": "artifact missing by 2026-05-09 09:00 ET → NEEDS_OPERATOR_DECISION",
    },
    {
        "id": "spec_087_b2",
        "title": "Spec 087 B2 — dashboard freshness envelope",
        "status": "HELD",
        "last_evidence": "Phase A memo artifacts/audit/spec_087_phase_a_bioshort_hedge_governance_decision_2026_05_06.md",
        "blocker": "B1b first-fire validation must pass",
        "next_allowed_action": "Draft dashboard staleness banner spec after first-fire pass",
        "not_allowed": [
            "imply hedge_report data is alpha-generating",
            "dashboard code changes before B1b closes",
        ],
        "requires_operator_approval": True,
        "related_artifacts": [],
        "related_cron": None,
        "alert_condition": "none until B1b passes",
    },
    {
        "id": "spec_087c",
        "title": "Spec 087C — bioshort alpha research",
        "status": "HELD",
        "last_evidence": "Phase A memo 2026-05-06. Only 1 fresh weekly report exists (2026-05-07).",
        "blocker": "Need ≥4 fresh weekly hedge reports OR a historical reconstruction plan",
        "next_allowed_action": "Phase A research design only — no implementation",
        "not_allowed": [
            "selector/ranker integration",
            "EV/sizing changes",
            "Module 3/5 changes",
            "bioshort_watch LLM reactivation",
        ],
        "requires_operator_approval": True,
        "related_artifacts": [],
        "related_cron": None,
        "alert_condition": "NEEDS_OPERATOR_DECISION if fresh report count ≥4 before 087C formally opened",
    },
    {
        "id": "bioshort_watch_llm",
        "title": "bioshort_watch LLM reactivation",
        "status": "HELD_SUPPRESSED",
        "last_evidence": "crontab comment: # SUPPRESSED 2026-05-06 (bioshort upstream P2)",
        "blocker": "Separate reactivation decision required; watcher stability unconfirmed",
        "next_allowed_action": "none",
        "not_allowed": [
            "cron reactivation",
            "run_agent_direct.py invocation",
            "any LLM call against bioshort artifacts",
        ],
        "requires_operator_approval": True,
        "related_artifacts": ["artifacts/bioshort_watch/"],
        "related_cron": "# SUPPRESSED: 10 18 * * 5 ... bioshort_watch HEARTBEAT",
        "alert_condition": "any cron/log entry for bioshort_watch LLM → escalate immediately",
    },
    {
        "id": "spec_088_phase_b",
        "title": "Spec 088 Phase B — catalyst_delta filtered artifacts",
        "status": "HELD",
        "last_evidence": "Phase A design doc 7471f77f committed 2026-05-07",
        "blocker": "Spec 087 active branch must close first",
        "next_allowed_action": "Implement raw+filtered companion artifacts only after 087 closes",
        "not_allowed": [
            "build_options_watch rewiring",
            "Module 3/5 changes",
            "any scoring changes",
            "options_watch pathway changes",
        ],
        "requires_operator_approval": True,
        "related_artifacts": ["artifacts/audit/spec_088_phase_a_catalyst_delta_filter_design_2026_05_07.md"],
        "related_cron": None,
        "alert_condition": "none until 087 closes",
    },
    {
        "id": "score_rank_pct_spec_required",
        "title": "score_rank_pct — SPEC_REQUIRED (Day 3+ WARN streak)",
        "status": "SPEC_REQUIRED",
        "last_evidence": "mean_ic=-0.0119, hit_rate=28.95%. Streak monitor fires nightly 22:00 ET.",
        "blocker": "CRT+IC+PIT+Checklist v2 required before any weight change",
        "next_allowed_action": "Spec writeup (if streak continues)",
        "not_allowed": [
            "weight change without Spec",
            "any scoring modification without CRT+IC+PIT+Checklist v2",
        ],
        "requires_operator_approval": True,
        "related_artifacts": [],
        "related_cron": "0 22 * * 1-5  # score-rank-pct-streak-monitor (4a96ad05405c)",
        "alert_condition": "streak continues → Spec writeup is next action",
    },
]


# ---------------------------------------------------------------------------
# Layer 2 — First-fire items
# ---------------------------------------------------------------------------

FIRST_FIRE_SEED = [
    {
        "job": "biotech_hedge_report",
        "cron": "0 18 * * 5",
        "expected_first_fire": "2026-05-08T18:00:00-04:00",
        "expected_artifacts": [
            "output/hedge_report/hedge_report_2026-05-08.json",
            "output/hedge_report/BIOSHORT_VERDICT.json",
        ],
        "expected_log": "logs/biotech_hedge_report.log",
        "status": "PENDING",
        "notes": "Spec 087 B1b install. Producer only; no LLM consumer active.",
        "pass_criteria": [
            "hedge_report_2026-05-08.json exists",
            "BIOSHORT_VERDICT.json as_of_date == 2026-05-08",
            "recommendation line non-empty",
            "no MASSIVE_API_KEY warnings in log",
            "biotech_hedge_report.log updated after 18:00",
        ],
        "fail_criteria": [
            "artifact missing by 2026-05-09 09:00 ET",
            "as_of_date != 2026-05-08",
            "exception/traceback in log",
        ],
        "alert_deadline": "2026-05-09T09:00:00-04:00",
    }
]


# ---------------------------------------------------------------------------
# Layer 3 — Automated checks
# ---------------------------------------------------------------------------


def check_first_fire_status(ff_item):
    """Evaluate a first-fire item against current filesystem state."""
    result = dict(ff_item)

    # Check if expected date has passed
    expected_dt = datetime.fromisoformat(ff_item["expected_first_fire"])
    now = datetime.now(tz=timezone(timedelta(hours=-4)))  # ET

    if now < expected_dt:
        result["eval"] = "PENDING_NOT_YET_DUE"
        return result

    # Past expected time — check artifacts
    missing = []
    for art in ff_item["expected_artifacts"]:
        p = REPO / art
        if not p.exists():
            missing.append(art)

    if missing:
        alert_dt = datetime.fromisoformat(ff_item["alert_deadline"])
        if now > alert_dt:
            result["eval"] = "FAIL_ARTIFACT_MISSING_PAST_DEADLINE"
        else:
            result["eval"] = "WARN_ARTIFACT_NOT_YET_PRESENT"
        result["missing_artifacts"] = missing
    else:
        # Artifacts exist — check as_of_date in BIOSHORT_VERDICT
        verdict = read_json(REPO / "output/hedge_report/BIOSHORT_VERDICT.json")
        expected_date = ff_item["expected_first_fire"][:10]  # YYYY-MM-DD
        if verdict and verdict.get("as_of_date") == expected_date:
            result["eval"] = "PASS"
        elif verdict:
            result["eval"] = "WARN_DATE_MISMATCH"
            result["verdict_as_of_date"] = verdict.get("as_of_date")
        else:
            result["eval"] = "WARN_VERDICT_UNREADABLE"

    return result


def detect_contradictions(git_info, crontab_info, agents, artifacts):
    """
    Lightweight contradiction scan.
    Returns list of {severity, description, recommendation}.
    """
    issues = []
    crontab_available = crontab_info.get("available", True)

    # C1: bioshort_watch registry says suppressed; crontab should have no active line
    active_bioshort_cron = any(
        "bioshort_watch" in line and not line.strip().startswith("#") for line in crontab_info["active_jobs"]
    )
    if not crontab_available:
        issues.append(
            {
                "id": "C1",
                "severity": "UNKNOWN_CLOUD_ENV",
                "description": "Cannot verify bioshort_watch cron suppression — crontab unavailable on this host.",
                "recommendation": "Re-run on operator host (crontab -l).",
            }
        )
    elif active_bioshort_cron:
        issues.append(
            {
                "id": "C1",
                "severity": "HARD_CONTRADICTION",
                "description": "bioshort_watch LLM is suppressed in AGENT_REGISTRY but an active crontab line references it.",
                "recommendation": "Comment out bioshort_watch cron line immediately.",
            }
        )
    else:
        issues.append(
            {
                "id": "C1",
                "severity": "OK",
                "description": "bioshort_watch: registry=suppressed, crontab=no active line. Consistent.",
                "recommendation": None,
            }
        )

    # C2: watchlist_current.json freshness check (file is git-untracked per
    # ledger 2026-05-07 §4 closure; cron producer rewrites it daily).
    wl_status = artifacts.get("watchlist_current_json", {})
    if not wl_status.get("exists"):
        issues.append(
            {
                "id": "C2",
                "severity": "NEEDS_OPERATOR_DECISION",
                "description": "watchlist_current.json missing. Producer (catalyst_resolution_tracker) has not run or output dir is wrong.",
                "recommendation": "Verify catalyst_resolution_tracker cron and rerun if needed.",
            }
        )
    elif wl_status.get("parse_error") or wl_status.get("as_of_date") is None:
        issues.append(
            {
                "id": "C2",
                "severity": "NEEDS_OPERATOR_DECISION",
                "description": "watchlist_current.json present but unparseable or missing as_of_date.",
                "recommendation": "Inspect file; rerun catalyst_resolution_tracker.",
            }
        )
    elif (wl_status.get("stale_days") or 0) > 3:
        issues.append(
            {
                "id": "C2",
                "severity": "WARN",
                "description": f"watchlist_current.json as_of_date={wl_status['as_of_date']} is {wl_status['stale_days']}d stale.",
                "recommendation": "Verify catalyst_resolution_tracker cron is firing.",
            }
        )
    else:
        issues.append(
            {
                "id": "C2",
                "severity": "OK",
                "description": f"watchlist_current.json fresh (as_of_date={wl_status.get('as_of_date')}, {wl_status.get('stale_days')}d old).",
                "recommendation": None,
            }
        )

    # C3: bioshort producer cron present and active
    producer_active = any("biotech_hedge_report.py" in line for line in crontab_info["active_jobs"])
    if not crontab_available:
        issues.append(
            {
                "id": "C3",
                "severity": "UNKNOWN_CLOUD_ENV",
                "description": (
                    "Cannot verify Spec 087 B1b biotech_hedge_report.py cron — crontab unavailable on this host."
                ),
                "recommendation": "Re-run on operator host: crontab -l | grep biotech_hedge_report",
            }
        )
    elif not producer_active:
        issues.append(
            {
                "id": "C3",
                "severity": "HARD_CONTRADICTION",
                "description": "Spec 087 B1b installed producer cron but no active biotech_hedge_report.py line found in crontab.",
                "recommendation": "Verify cron install. Check crontab -l for B1b line.",
            }
        )
    else:
        issues.append(
            {
                "id": "C3",
                "severity": "OK",
                "description": "biotech_hedge_report.py cron line is active. Consistent with Spec 087 B1b.",
                "recommendation": None,
            }
        )

    # C4: uncommitted files in git status (general)
    uncommitted = git_info.get("uncommitted", [])
    if uncommitted:
        issues.append(
            {
                "id": "C4",
                "severity": "POSSIBLE_DRIFT",
                "description": f"Uncommitted working tree changes: {uncommitted}",
                "recommendation": "Review each file. Do not commit held items as part of unrelated changes.",
            }
        )
    else:
        issues.append(
            {
                "id": "C4",
                "severity": "OK",
                "description": "Working tree is clean.",
                "recommendation": None,
            }
        )

    # C5: BIOSHORT_VERDICT as_of_date vs expected first-fire date
    verdict_info = artifacts.get("BIOSHORT_VERDICT", {})
    verdict_date = verdict_info.get("as_of_date")
    if verdict_date and verdict_date < "2026-05-08":
        issues.append(
            {
                "id": "C5",
                "severity": "POSSIBLE_DRIFT",
                "description": f"BIOSHORT_VERDICT.json as_of_date={verdict_date}. First-fire expected 2026-05-08. Verdict is pre-first-fire.",
                "recommendation": "Normal pre-first-fire state. Re-check after 2026-05-08 18:00 ET.",
            }
        )
    else:
        issues.append(
            {
                "id": "C5",
                "severity": "OK" if verdict_date else "POSSIBLE_DRIFT",
                "description": f"BIOSHORT_VERDICT as_of_date={verdict_date or 'MISSING'}.",
                "recommendation": None if verdict_date else "Verify BIOSHORT_VERDICT.json.",
            }
        )

    return issues


# ---------------------------------------------------------------------------
# Layer 4 — Write outputs
# ---------------------------------------------------------------------------


def write_state_json(git_info, crontab_info, agents, artifacts, held_items, ff_items, contradictions):
    state = {
        "as_of_date": TODAY,
        "generated_at": datetime.now().isoformat(),
        "git": git_info,
        "cron": {
            "available": crontab_info.get("available", True),
            "availability": crontab_info.get("availability", "OPERATOR_HOST"),
            "active_job_count": len(crontab_info["active_jobs"]),
            "suppressed_job_count": len(crontab_info["suppressed_jobs"]),
            "suppressed_markers": crontab_info["suppressed_jobs"],
        },
        "agents": {
            "total": len(agents),
            "by_status": {},
        },
        "key_artifacts": artifacts,
        "held_items": [{"id": h["id"], "title": h["title"], "status": h["status"]} for h in held_items],
        "first_fire_items": [
            {"job": f["job"], "expected_first_fire": f["expected_first_fire"], "status": f.get("eval", f["status"])}
            for f in ff_items
        ],
        "warnings": [c for c in contradictions if c["severity"] != "OK"],
    }

    # agent status breakdown
    for name, info in agents.items():
        st = info["status"]
        state["agents"]["by_status"].setdefault(st, []).append(name)

    out = OUT_KL / "latest_state.json"
    with open(out, "w") as f:
        json.dump(state, f, indent=2)
    print(f"  wrote {out}")
    return state


def write_state_md(state):
    warnings = state["warnings"]
    held = state["held_items"]
    ff = state["first_fire_items"]
    arts = state["key_artifacts"]

    lines = [
        "# Hermes Knowledge Layer — State Snapshot",
        "",
        f"as_of_date: {state['as_of_date']}",
        f"generated: {state['generated_at']}",
        "",
        "---",
        "",
        "## Git",
        "",
        f"- head: {state['git']['head']}",
        f"- branch: {state['git']['branch']}",
    ]
    uncommitted = state["git"]["uncommitted"]
    if uncommitted:
        lines.append(f"- uncommitted ({len(uncommitted)} files):")
        for u in uncommitted:
            lines.append(f"    {u}")
    else:
        lines.append("- uncommitted: (none)")

    lines += [
        "",
        "## Cron",
        "",
        f"- active entries: {state['cron']['active_job_count']}",
        f"- suppressed markers: {state['cron']['suppressed_job_count']}",
    ]
    for s in state["cron"]["suppressed_markers"]:
        lines.append(f"    {s}")

    lines += [
        "",
        "## Agents",
        "",
    ]
    for status, names in state["agents"]["by_status"].items():
        lines.append(f"- {status}: {', '.join(names)}")

    lines += [
        "",
        "## Key Artifacts",
        "",
    ]
    for k, v in arts.items():
        if isinstance(v, dict):
            lines.append(f"- {k}: {json.dumps(v)}")
        else:
            lines.append(f"- {k}: {v}")

    lines += [
        "",
        f"## Held Items ({len(held)})",
        "",
    ]
    for h in held:
        lines.append(f"- [{h['status']}] {h['title']}")

    lines += [
        "",
        f"## First-Fire Items ({len(ff)})",
        "",
    ]
    for item in ff:
        lines.append(f"- {item['job']}: {item['status']}  (expected: {item['expected_first_fire']})")

    if warnings:
        lines += [
            "",
            f"## Warnings / Contradictions ({len(warnings)})",
            "",
        ]
        for w in warnings:
            lines.append(f"- [{w['severity']}] {w['id']}: {w['description']}")
            if w.get("recommendation"):
                lines.append(f"    -> {w['recommendation']}")
    else:
        lines += [
            "",
            "## Warnings / Contradictions",
            "",
            "None detected.",
        ]

    out = OUT_KL / "latest_state.md"
    out.write_text("\n".join(lines))
    print(f"  wrote {out}")


def write_first_fire_json(ff_items):
    payload = {
        "as_of_date": TODAY,
        "generated_at": datetime.now().isoformat(),
        "jobs": ff_items,
    }
    out = OUT_FF / "latest.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {out}")

    # dated copy
    dated = OUT_FF / f"{TODAY}.json"
    with open(dated, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {dated}")


def write_first_fire_md(ff_items):
    lines = [
        f"# First-Fire Ledger — {TODAY}",
        "",
        "Generated by: tools/build_hermes_knowledge_layer.py",
        "",
        "---",
        "",
    ]
    for item in ff_items:
        eval_status = item.get("eval", item["status"])
        lines += [
            f"## {item['job']}",
            "",
            f"- cron: `{item['cron']}`",
            f"- expected first fire: {item['expected_first_fire']}",
            f"- alert deadline: {item['alert_deadline']}",
            f"- status: **{eval_status}**",
            "",
            "### Pass criteria",
            "",
        ]
        for c in item["pass_criteria"]:
            lines.append(f"- {c}")
        lines += [
            "",
            "### Fail criteria",
            "",
        ]
        for c in item["fail_criteria"]:
            lines.append(f"- {c}")

        if item.get("missing_artifacts"):
            lines += [
                "",
                "### Missing artifacts",
                "",
            ]
            for a in item["missing_artifacts"]:
                lines.append(f"- {a}")

        lines += [
            "",
            "### Notes",
            "",
            f"{item['notes']}",
            "",
            "---",
            "",
        ]

    out = OUT_FF / "latest.md"
    out.write_text("\n".join(lines))
    print(f"  wrote {out}")


def write_contradiction_md(contradictions):
    hard = [c for c in contradictions if c["severity"] == "HARD_CONTRADICTION"]
    cloud_env = [c for c in contradictions if c["severity"] == "UNKNOWN_CLOUD_ENV"]
    possible = [
        c
        for c in contradictions
        if c["severity"] in ("POSSIBLE_DRIFT", "WARN", "NEEDS_OPERATOR_DECISION")
    ]
    ok = [c for c in contradictions if c["severity"] == "OK"]

    lines = [
        f"# Contradiction Ledger — {TODAY}",
        "",
        "Generated by: tools/build_hermes_knowledge_layer.py",
        "",
        "---",
        "",
        f"## 1. Hard Contradictions ({len(hard)})",
        "",
    ]
    if hard:
        for c in hard:
            lines.append(f"- **{c['id']}**: {c['description']}")
            if c.get("recommendation"):
                lines.append(f"  - Action: {c['recommendation']}")
    else:
        lines.append("None.")

    lines += [
        "",
        f"## 2. Non-Authoritative Host (Cloud / No Crontab) ({len(cloud_env)})",
        "",
    ]
    if cloud_env:
        for c in cloud_env:
            lines.append(f"- **{c['id']}**: {c['description']}")
            if c.get("recommendation"):
                lines.append(f"  - Action: {c['recommendation']}")
    else:
        lines.append("None.")

    lines += [
        "",
        f"## 3. Possible Drift / Warnings ({len(possible)})",
        "",
    ]
    if possible:
        for c in possible:
            lines.append(f"- **{c['id']}**: {c['description']}")
            if c.get("recommendation"):
                lines.append(f"  - Recommendation: {c['recommendation']}")
    else:
        lines.append("None.")

    lines += [
        "",
        f"## 4. OK / Consistent ({len(ok)})",
        "",
    ]
    for c in ok:
        lines.append(f"- {c['id']}: {c['description']}")

    out = OUT_CONTRA / "latest.md"
    out.write_text("\n".join(lines))
    print(f"  wrote {out}")

    # dated copy
    dated = OUT_CONTRA / f"{TODAY}.md"
    dated.write_text("\n".join(lines))
    print(f"  wrote {dated}")


def write_held_spec_json(held_items):
    payload = {
        "as_of_date": TODAY,
        "generated_at": datetime.now().isoformat(),
        "items": held_items,
    }
    out = OUT_HELD / "latest.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    started = time.perf_counter()
    print(f"[build_hermes_knowledge_layer] {TODAY}")
    print(f"  repo: {REPO}")
    print()

    # Layer 1 — capture
    print("Layer 1: capture...")
    git_info = capture_git()
    cron_info = capture_crontab()
    agents = capture_agent_registry()
    artifacts = capture_key_artifacts()

    # Layer 2 — normalize first-fire items with eval
    print("Layer 2: normalize...")
    ff_items = [check_first_fire_status(item) for item in FIRST_FIRE_SEED]

    # Layer 3 — contradiction scan
    print("Layer 3: contradiction scan...")
    contradictions = detect_contradictions(git_info, cron_info, agents, artifacts)

    # Layer 4 — write
    print("Layer 4: write outputs...")
    state = write_state_json(git_info, cron_info, agents, artifacts, HELD_ITEMS_SEED, ff_items, contradictions)
    write_state_md(state)
    write_first_fire_json(ff_items)
    write_first_fire_md(ff_items)
    write_contradiction_md(contradictions)
    write_held_spec_json(HELD_ITEMS_SEED)

    # Phase B: route hard contradictions to Town (dry-run unless OPERATOR_DELIVERY_DRY_RUN=0)
    try:
        import sys

        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from common.town_bridge_events import notify_hard_contradictions

        notify_hard_contradictions(contradictions)
    except Exception as exc:
        print(f"  town bridge (contradiction_detected): {exc}")

    # Summary
    print()
    hard_count = sum(1 for c in contradictions if c["severity"] == "HARD_CONTRADICTION")
    cloud_count = sum(1 for c in contradictions if c["severity"] == "UNKNOWN_CLOUD_ENV")
    warn_count = sum(
        1
        for c in contradictions
        if c["severity"] in ("POSSIBLE_DRIFT", "WARN", "NEEDS_OPERATOR_DECISION")
    )
    ff_status = ff_items[0].get("eval", ff_items[0]["status"]) if ff_items else "N/A"
    cron_availability = cron_info.get("availability", "OPERATOR_HOST")

    print("=== Summary ===")
    print(f"  git head:           {git_info['head']}")
    print(f"  uncommitted files:  {len(git_info['uncommitted'])}")
    print(f"  crontab surface:    {cron_availability}")
    print(f"  held items:         {len(HELD_ITEMS_SEED)}")
    print(f"  first-fire status:  {ff_status}")
    print(f"  contradictions:     {hard_count} hard  /  {cloud_count} cloud-env  /  {warn_count} possible")
    print()
    if hard_count > 0:
        print("  HARD CONTRADICTIONS detected — review contradiction_ledger/latest.md")
    elif cloud_count > 0:
        print("  Cron checks skipped (non-authoritative host) — verify on operator machine.")
    elif warn_count > 0:
        print("  Possible drift items — review contradiction_ledger/latest.md")
    else:
        print("  No contradictions detected.")
    print()
    print("  Outputs:")
    print(f"    {OUT_KL}/latest_state.json")
    print(f"    {OUT_KL}/latest_state.md")
    print(f"    {OUT_FF}/latest.json")
    print(f"    {OUT_FF}/latest.md")
    print(f"    {OUT_CONTRA}/latest.md")
    print(f"    {OUT_HELD}/latest.json")

    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        exec_id = log_agent_run(
            "build_hermes_knowledge_layer",
            f"Hermes knowledge layer for {TODAY}",
            outputs={
                "hard_contradictions": hard_count,
                "held_items": len(HELD_ITEMS_SEED),
                "first_fire_status": ff_status,
            },
            success=hard_count == 0,
            error=f"{hard_count} hard contradictions" if hard_count else None,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if exec_id:
            attach_outcome_verdict(
                exec_id,
                was_correct=hard_count == 0,
                evidence=f"hard_contradictions={hard_count} warn={warn_count}",
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
