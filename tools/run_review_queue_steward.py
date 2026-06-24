#!/usr/bin/env python3
"""
tools/run_review_queue_steward.py
Lightweight replacement for run_agent_direct.py --agent review_queue_steward.

Reads today's review_queue.csv from the snapshot, triages into
immediate/monitor/watch buckets, and writes a memory file.
No LLM call — deterministic, fast, always produces an artifact.
"""

import argparse
import csv
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MEMORY_DIR = REPO / "agents" / "review_queue_steward" / "memory"
MEMORY_DIR.mkdir(exist_ok=True)


def triage_action(row: dict) -> str:
    action = row.get("action", "").lower()
    if action in ("no_add_until_review", "immediate_review"):
        return "IMMEDIATE"
    if action in ("monitor", "watch"):
        return "MONITOR"
    return "WATCH"


def main():
    parser = argparse.ArgumentParser(description="Review queue steward (deterministic)")
    parser.add_argument("--as-of-date", default=None, help="Snapshot date (YYYY-MM-DD); default today")
    args = parser.parse_args()
    started = time.perf_counter()

    snapshot_date = args.as_of_date or date.today().isoformat()
    snapshot_dir = REPO / "data" / "snapshots" / snapshot_date
    queue_csv = snapshot_dir / "review_queue.csv"
    memory_file = MEMORY_DIR / f"{snapshot_date}.md"

    if not queue_csv.exists():
        print(f"[review_queue_steward] No queue for {snapshot_date} — skip", file=sys.stderr)
        sys.exit(0)

    rows = []
    with open(queue_csv, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    immediate = [r for r in rows if triage_action(r) == "IMMEDIATE"]
    monitor = [r for r in rows if triage_action(r) == "MONITOR"]
    watch = [r for r in rows if triage_action(r) == "WATCH"]

    as_of_d = date.fromisoformat(snapshot_date)
    yesterday = date.fromordinal(as_of_d.toordinal() - 1).isoformat()
    yesterday_file = MEMORY_DIR / f"{yesterday}.md"
    prev_immediate = set()
    if yesterday_file.exists():
        for line in yesterday_file.read_text().split("\n"):
            if line.startswith("- **") or line.startswith("- "):
                ticker = line.strip().lstrip("- **").split("**")[0].split(" ")[0]
                if ticker and len(ticker) <= 6 and ticker.isupper():
                    prev_immediate.add(ticker)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    lines = [
        f"# Review Queue Steward Memory — {snapshot_date}",
        "",
        f"Generated: {now}  |  Total queue: {len(rows)} names",
        "",
        f"## IMMEDIATE ({len(immediate)}) — no_add_until_review / immediate_review",
        "",
    ]
    for r in immediate:
        ticker = r.get("ticker", "?")
        tier = r.get("tier", "?")
        reason = r.get("action_reason", r.get("action", ""))
        cat = r.get("catalyst_family", "")
        days = r.get("catalyst_days", "")
        new_flag = " ⚠️ NEW" if ticker not in prev_immediate else ""
        lines.append(f"- **{ticker}** ({tier}) — {reason}" + (f" | {cat} {days}d" if cat else "") + new_flag)

    lines += [
        "",
        f"## MONITOR ({len(monitor)})",
        "",
    ]
    for r in monitor[:10]:
        ticker = r.get("ticker", "?")
        tier = r.get("tier", "?")
        reason = r.get("action_reason", r.get("action", ""))
        lines.append(f"- {ticker} ({tier}) — {reason}")
    if len(monitor) > 10:
        lines.append(f"- ... and {len(monitor) - 10} more")

    lines += [
        "",
        f"## WATCH ({len(watch)})",
        "",
        "_(top 5 only)_",
    ]
    for r in watch[:5]:
        ticker = r.get("ticker", "?")
        tier = r.get("tier", "?")
        lines.append(f"- {ticker} ({tier})")

    lines += [
        "",
        "## Alert",
        "",
    ]
    if len(immediate) >= 5:
        lines.append(f"HIGH — {len(immediate)} names in IMMEDIATE bucket")
    elif len(immediate) >= 3:
        lines.append(f"MEDIUM — {len(immediate)} names in IMMEDIATE bucket")
    elif len(immediate) >= 1:
        lines.append(f"LOW — {len(immediate)} name(s) in IMMEDIATE bucket")
    else:
        lines.append("NONE — queue clean")

    memory_file.write_text("\n".join(lines) + "\n")
    print(
        f"[review_queue_steward] {snapshot_date}: {len(immediate)} immediate, "
        f"{len(monitor)} monitor, {len(watch)} watch → {memory_file}"
    )

    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        exec_id = log_agent_run(
            "review_queue_steward",
            f"Review queue steward for {snapshot_date}",
            inputs={"as_of_date": snapshot_date},
            outputs={"n_immediate": len(immediate), "n_monitor": len(monitor), "n_watch": len(watch)},
            success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if exec_id:
            attach_outcome_verdict(
                exec_id,
                was_correct=len(immediate) < 5,
                evidence=f"immediate={len(immediate)} monitor={len(monitor)}",
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
