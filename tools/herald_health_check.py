#!/usr/bin/env python3
"""Herald pipeline health check for host cron and operator triage.

Read-only. Validates fetch → dedupe → classify → digest artifacts against
the supervisor done predicate and HEARTBEAT.md staleness rules.

Exit codes:
  0 — HEALTHY (or acceptable weekend skip)
  1 — WARN (partial pipeline, stale digest, aged source)
  2 — FAIL (dark pipeline, missing done predicate on trading day)

Usage:
    python3 tools/herald_health_check.py
    python3 tools/herald_health_check.py --as-of-date 2026-06-24
    python3 tools/herald_health_check.py --json
    python3 tools/herald_health_check.py --stdout
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PR_DIR = REPO / "data" / "press_releases"
DEDUPED_DIR = PR_DIR / "deduped"
CLASSIFIED_DIR = PR_DIR / "classified"
DIGEST_DIR = REPO / "artifacts" / "news_digest"
OUT_DIR = REPO / "artifacts" / "herald"
STATE_PATH = PR_DIR / "fetch_state.json"

CLASSIFIED_RE = re.compile(r"classified_(\d{4}-\d{2}-\d{2})\.jsonl$")
STALE_SOURCE_DAYS = 2
DARK_FAIL_DAYS = 7


def _iso_today() -> date:
    return date.today()


def _artifact_paths(as_of: str) -> dict[str, Path]:
    return {
        "releases": PR_DIR / f"releases_{as_of}.jsonl",
        "deduped": DEDUPED_DIR / f"deduped_{as_of}.jsonl",
        "classified": CLASSIFIED_DIR / f"classified_{as_of}.jsonl",
        "fetch_health": PR_DIR / f"health_{as_of}.json",
    }


def _file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "bytes": 0}
    st = path.stat()
    return {
        "exists": True,
        "bytes": st.st_size,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }


def _latest_classified_date() -> date | None:
    latest: date | None = None
    if not CLASSIFIED_DIR.is_dir():
        return None
    for path in CLASSIFIED_DIR.glob("classified_*.jsonl"):
        m = CLASSIFIED_RE.match(path.name)
        if not m:
            continue
        try:
            d = date.fromisoformat(m.group(1))
        except ValueError:
            continue
        if latest is None or d > latest:
            latest = d
    return latest


def _digest_count(ds: str) -> int:
    if not DIGEST_DIR.is_dir():
        return 0
    return len(list(DIGEST_DIR.glob(f"biotech_news_digest_{ds}_*.json")))


def _delivery_failures(ds: str) -> list[str]:
    log_path = DIGEST_DIR / "delivery_log.jsonl"
    if not log_path.exists():
        return []
    failures: list[str] = []
    for line in log_path.read_text(encoding="utf-8").strip().split("\n")[-50:]:
        if not line.strip() or ds not in line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("status") == "FAIL":
            failures.append(entry.get("window", "unknown"))
    return failures


def herald_done(as_of: str) -> bool:
    paths = _artifact_paths(as_of)
    return paths["deduped"].exists() and paths["classified"].exists()


def run_check(as_of: date | None = None) -> dict[str, Any]:
    as_of = as_of or _iso_today()
    ds = as_of.isoformat()
    paths = _artifact_paths(ds)
    artifacts = {k: _file_info(p) for k, p in paths.items()}

    issues: list[str] = []
    status_codes: list[str] = []

    done = artifacts["deduped"]["exists"] and artifacts["classified"]["exists"]
    if not done and as_of.weekday() < 5:
        missing = []
        if not artifacts["deduped"]["exists"]:
            missing.append("deduped")
        if not artifacts["classified"]["exists"]:
            missing.append("classified")
        issues.append(f"INCOMPLETE_PIPELINE: missing {', '.join(missing)} for {ds}")

    latest_classified = _latest_classified_date()
    source_age_days: int | None = None
    if latest_classified is not None:
        source_age_days = (as_of - latest_classified).days
        if source_age_days > STALE_SOURCE_DAYS:
            status_codes.append("STALE_SOURCE")
            issues.append(
                f"STALE_SOURCE: latest classified {latest_classified.isoformat()} "
                f"({source_age_days}d ago; threshold {STALE_SOURCE_DAYS}d)"
            )
    else:
        status_codes.append("STALE_SOURCE")
        issues.append("STALE_SOURCE: no classified_*.jsonl files found")

    hour = datetime.now().hour
    today_digests = _digest_count(ds)
    if as_of.weekday() < 5:
        if hour >= 19 and today_digests == 0:
            status_codes.append("MISSED_DIGEST")
            issues.append(f"MISSED_DIGEST: no digest artifacts for {ds} by 19:00 ET check")
        elif hour < 19:
            yesterday = as_of - timedelta(days=1)
            if yesterday.weekday() < 5 and _digest_count(yesterday.isoformat()) == 0:
                status_codes.append("MISSED_DIGEST")
                issues.append(f"MISSED_DIGEST: no digest for prior trading day {yesterday.isoformat()}")

    send_fails = _delivery_failures(ds)
    if send_fails:
        status_codes.append("SEND_FAILURE")
        issues.append(f"SEND_FAILURE: windows failed: {', '.join(send_fails)}")

    if STATE_PATH.exists():
        artifacts["fetch_state"] = _file_info(STATE_PATH)
    else:
        artifacts["fetch_state"] = {"exists": False, "bytes": 0}
        if as_of.weekday() < 5:
            issues.append("WARN: fetch_state.json missing")

    # Overall verdict
    if source_age_days is not None and source_age_days >= DARK_FAIL_DAYS:
        verdict = "FAIL"
        status_codes.append("DARK")
    elif done and not status_codes:
        verdict = "HEALTHY"
    elif done and all(c in ("MISSED_DIGEST",) for c in status_codes):
        verdict = "WARN"
    elif issues and not done:
        verdict = "FAIL" if (source_age_days or 999) >= DARK_FAIL_DAYS else "WARN"
    elif status_codes:
        verdict = "WARN"
    else:
        verdict = "HEALTHY"

    return {
        "schema": "herald_health_check.v1",
        "as_of_date": ds,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "status_codes": sorted(set(status_codes)),
        "herald_done": done,
        "latest_classified_date": latest_classified.isoformat() if latest_classified else None,
        "source_age_days": source_age_days,
        "digest_count_today": today_digests,
        "artifacts": artifacts,
        "issues": issues,
        "recovery": {
            "health_check": "python3 tools/herald_health_check.py",
            "fetch": f"python3 tools/fetch_company_press_releases.py --as-of-date {ds}",
            "dedupe": f"python3 tools/dedupe_press_releases.py --input data/press_releases/releases_{ds}.jsonl",
            "classify": (
                f"python3 tools/classify_press_releases.py "
                f"--input data/press_releases/deduped/deduped_{ds}.jsonl"
            ),
            "digest": f"python3 scripts/build_news_digest.py --window evening --as-of-date {ds}",
        },
    }


def _exit_code(verdict: str) -> int:
    if verdict == "HEALTHY":
        return 0
    if verdict == "WARN":
        return 1
    return 2


def main() -> int:
    ap = argparse.ArgumentParser(description="Herald pipeline health check (read-only)")
    ap.add_argument("--as-of-date", help="YYYY-MM-DD (default: today)")
    ap.add_argument("--json", action="store_true", help="Print JSON to stdout only")
    ap.add_argument("--stdout", action="store_true", help="Print human summary to stdout (no file write)")
    ap.add_argument("--no-write", action="store_true", help="Do not write artifact file")
    args = ap.parse_args()

    as_of = date.fromisoformat(args.as_of_date) if args.as_of_date else None
    report = run_check(as_of)

    if not args.no_write and not args.stdout:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"health_check_{report['as_of_date']}.json"
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if not args.json:
            print(f"Wrote {out_path}")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.stdout or args.no_write:
        print(f"Herald health: {report['verdict']} (done={report['herald_done']})")
        for issue in report["issues"]:
            print(f"  - {issue}")
        if report["latest_classified_date"]:
            print(f"  latest classified: {report['latest_classified_date']} ({report['source_age_days']}d ago)")

    return _exit_code(report["verdict"])


if __name__ == "__main__":
    raise SystemExit(main())
