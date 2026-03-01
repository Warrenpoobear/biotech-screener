#!/usr/bin/env python3
"""Company Calendar Future-Date Quality Gate (v1).

Validates IR events + press release caches against future-date invariants:
  - All event_date values must be day-precision (YYYY-MM-DD)
  - All event_date values must be >= as_of_date (no past dates)

Exit semantics:
  0 = PASS   — both streams valid, above minimum thresholds
  1 = WARN   — cache missing or below min-event thresholds
  2 = FAIL   — bad day precision or past dates detected
  3 = ERROR  — unreadable JSON / unexpected exception

Usage:
    python3 scripts/eval_company_calendar_gate.py \
      --as-of-date 2026-02-28 \
      --ir-cache cache/ir/ir_events_2026-02-28.json \
      --pr-cache cache/press/press_release_events_2026-02-28.json \
      --out-dir data/snapshots/2026-02-28/company_calendar_gate
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

VERSION = "1.0.0"
SCHEMA = "company_calendar_gate.v1"

EXIT_PASS = 0
EXIT_WARN = 1
EXIT_FAIL = 2
EXIT_ERROR = 3

_DAY_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stable_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n"


def _load_cache(path: Path) -> Dict[str, Any] | None:
    """Load a cache JSON file. Returns None if missing or unreadable."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Per-stream validation
# ---------------------------------------------------------------------------

def validate_stream(
    cache: Dict[str, Any] | None,
    stream_name: str,
    as_of_date: str,
    max_unmatched_samples: int,
) -> Dict[str, Any]:
    """Validate a single IR or PR cache stream.

    Returns a dict with per-stream stats:
      event_count, bad_day_precision, past_dates,
      unmatched_samples (bounded), missing (bool).
    """
    result: Dict[str, Any] = {
        "stream": stream_name,
        "missing": cache is None,
        "event_count": 0,
        "bad_day_precision": 0,
        "bad_day_precision_samples": [],
        "past_dates": 0,
        "past_date_samples": [],
        "unmatched_samples": [],
    }

    if cache is None:
        return result

    events: List[Dict[str, Any]] = cache.get("events", [])
    result["event_count"] = len(events)

    for ev in events:
        ed = ev.get("event_date", "")
        if not _DAY_RE.fullmatch(str(ed)):
            result["bad_day_precision"] += 1
            if len(result["bad_day_precision_samples"]) < 5:
                result["bad_day_precision_samples"].append(
                    {"ticker": ev.get("ticker", "?"), "event_date": ed}
                )
        elif str(ed) < as_of_date:
            result["past_dates"] += 1
            if len(result["past_date_samples"]) < 5:
                result["past_date_samples"].append(
                    {"ticker": ev.get("ticker", "?"), "event_date": ed}
                )

    # Bounded unmatched samples from stats
    stats = cache.get("stats", {})
    raw_unmatched = stats.get("unmatched_samples", [])
    if isinstance(raw_unmatched, list):
        result["unmatched_samples"] = raw_unmatched[:max_unmatched_samples]

    return result


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

def evaluate_gate(
    ir_result: Dict[str, Any],
    pr_result: Dict[str, Any],
    *,
    min_ir_events: int,
    min_pr_events: int,
) -> Tuple[str, List[str]]:
    """Determine verdict from stream validation results.

    Returns (verdict, notes).
    """
    notes: List[str] = []

    # FAIL if any stream has bad precision or past dates
    fail = False
    for r in (ir_result, pr_result):
        name = r["stream"]
        if r["bad_day_precision"] > 0:
            fail = True
            notes.append(
                f"{name}: {r['bad_day_precision']} event(s) with non-day precision"
            )
        if r["past_dates"] > 0:
            fail = True
            notes.append(
                f"{name}: {r['past_dates']} event(s) with past date"
            )

    if fail:
        return "FAIL", notes

    # WARN if cache missing or below minimums
    warn = False
    for r, label, minimum in [
        (ir_result, "IR", min_ir_events),
        (pr_result, "PR", min_pr_events),
    ]:
        if r["missing"]:
            warn = True
            notes.append(f"{label} cache file missing")
        elif r["event_count"] < minimum:
            warn = True
            notes.append(
                f"{label} events={r['event_count']} < min={minimum}"
            )

    if warn:
        return "WARN", notes

    notes.append(
        f"IR={ir_result['event_count']} events, PR={pr_result['event_count']} events"
    )
    return "PASS", notes


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

def render_markdown(
    verdict: str,
    as_of_date: str,
    ir_result: Dict[str, Any],
    pr_result: Dict[str, Any],
    notes: List[str],
) -> str:
    lines = [
        "## Company Calendar Gate", "",
        "| Field | IR Events | Press Releases |",
        "|---|---|---|",
        f"| cache_missing | `{ir_result['missing']}` | `{pr_result['missing']}` |",
        f"| event_count | `{ir_result['event_count']}` | `{pr_result['event_count']}` |",
        f"| bad_day_precision | `{ir_result['bad_day_precision']}` | `{pr_result['bad_day_precision']}` |",
        f"| past_dates | `{ir_result['past_dates']}` | `{pr_result['past_dates']}` |",
        "",
        f"**Verdict**: **{verdict}**  ",
        f"**as_of_date**: `{as_of_date}`",
    ]
    if notes:
        lines += ["", "**Notes**"]
        lines += [f"- {n}" for n in notes]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Company calendar future-date quality gate.")
    ap.add_argument("--as-of-date", required=True, help="YYYY-MM-DD reference date.")
    ap.add_argument(
        "--ir-cache", default="",
        help="Path to IR events cache JSON. Default: cache/ir/ir_events_{date}.json",
    )
    ap.add_argument(
        "--pr-cache", default="",
        help="Path to PR events cache JSON. Default: cache/press/press_release_events_{date}.json",
    )
    ap.add_argument("--out-dir", default="", help="Output directory for gate artifacts.")
    ap.add_argument("--min-ir-events", type=int, default=1, help="Min IR events for PASS.")
    ap.add_argument("--min-pr-events", type=int, default=1, help="Min PR events for PASS.")
    ap.add_argument("--max-unmatched-samples", type=int, default=100,
                     help="Cap on unmatched samples in output.")
    args = ap.parse_args(argv)

    try:
        as_of = args.as_of_date.strip()
        # Validate date format
        date.fromisoformat(as_of)

        ir_path = Path(args.ir_cache) if args.ir_cache else (
            _PROJECT_ROOT / "cache" / "ir" / f"ir_events_{as_of}.json"
        )
        pr_path = Path(args.pr_cache) if args.pr_cache else (
            _PROJECT_ROOT / "cache" / "press" / f"press_release_events_{as_of}.json"
        )

        ir_cache = _load_cache(ir_path)
        pr_cache = _load_cache(pr_path)

        ir_result = validate_stream(ir_cache, "IR", as_of, args.max_unmatched_samples)
        pr_result = validate_stream(pr_cache, "PR", as_of, args.max_unmatched_samples)

        verdict, notes = evaluate_gate(
            ir_result, pr_result,
            min_ir_events=args.min_ir_events,
            min_pr_events=args.min_pr_events,
        )

        # Build output
        gate_json = {
            "schema": SCHEMA,
            "version": VERSION,
            "as_of_date": as_of,
            "verdict": verdict,
            "notes": notes,
            "streams": {
                "ir": {k: v for k, v in ir_result.items() if k != "unmatched_samples"},
                "pr": {k: v for k, v in pr_result.items() if k != "unmatched_samples"},
            },
            "thresholds": {
                "min_ir_events": args.min_ir_events,
                "min_pr_events": args.min_pr_events,
                "max_unmatched_samples": args.max_unmatched_samples,
            },
        }

        combined_unmatched = ir_result["unmatched_samples"] + pr_result["unmatched_samples"]

        md_text = render_markdown(verdict, as_of, ir_result, pr_result, notes)

        # Write outputs
        if args.out_dir:
            out_dir = Path(args.out_dir)
        else:
            out_dir = _PROJECT_ROOT / "data" / "snapshots" / as_of / "company_calendar_gate"
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "company_calendar_gate.json").write_text(
            _stable_json(gate_json), encoding="utf-8",
        )
        (out_dir / "company_calendar_gate.md").write_text(md_text, encoding="utf-8")
        (out_dir / "unmatched_samples.json").write_text(
            _stable_json(combined_unmatched), encoding="utf-8",
        )

        print(f"Company Calendar Gate: {verdict}")
        for n in notes:
            print(f"  {n}")
        print(f"Wrote: {out_dir}")

        if verdict == "FAIL":
            return EXIT_FAIL
        elif verdict == "WARN":
            return EXIT_WARN
        return EXIT_PASS

    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
