#!/usr/bin/env python3
"""Event analyst — aggregate postmortem patterns.

Second-layer reader over postmortem artifacts. Slices resolved event
outcomes by family, tier, bucket, shadow membership, and trade-plan
presence. Surfaces pattern-level lessons, not one-off narratives.

Read-only — does not affect rulesets, scoring, or promotion packets.

Output:
    artifacts/event_analyst/{date}_summary.json
    artifacts/event_analyst/{date}_summary.md

Usage:
    python tools/build_event_analyst.py --as-of-date 2026-04-15
    python tools/build_event_analyst.py --as-of-date 2026-04-15 --lookback 90
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("event_analyst")

SCHEMA_VERSION = "event_analyst.v1"
DEFAULT_LOOKBACK = 90  # days


def _sf(val: Any) -> float:
    if val is None or val == "":
        return math.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return math.nan


def load_postmortems(
    postmortem_dir: Path,
    as_of_date: str,
    lookback_days: int,
) -> List[Dict[str, Any]]:
    """Load all postmortem records within the lookback window."""
    records = []
    if not postmortem_dir.exists():
        return records

    for date_dir in sorted(postmortem_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        if date_dir.name > as_of_date:
            continue

        for pm_file in date_dir.glob("*.json"):
            try:
                with open(pm_file, encoding="utf-8") as f:
                    record = json.load(f)
                if record.get("schema") == "postmortem.v1":
                    records.append(record)
            except (json.JSONDecodeError, OSError):
                continue

    return records


def compute_slice_stats(
    records: List[Dict[str, Any]],
    slice_key: str,
) -> Dict[str, Dict[str, Any]]:
    """Compute aggregate stats grouped by a pre_event field."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        pre = r.get("pre_event", {})
        outcome = r.get("outcome", {})
        key_val = str(pre.get(slice_key, "unknown"))
        groups[key_val].append(outcome)

    stats = {}
    for key_val, outcomes in sorted(groups.items()):
        returns_t1 = [_sf(o.get("return_t1")) for o in outcomes]
        returns_t3 = [_sf(o.get("return_t3")) for o in outcomes]
        returns_t5 = [_sf(o.get("return_t5")) for o in outcomes]
        excess_t1 = [_sf(o.get("excess_vs_xbi_t1")) for o in outcomes]
        abs_gaps = [_sf(o.get("abs_gap")) for o in outcomes]

        valid_t1 = [r for r in returns_t1 if not math.isnan(r)]
        valid_t3 = [r for r in returns_t3 if not math.isnan(r)]
        valid_t5 = [r for r in returns_t5 if not math.isnan(r)]
        valid_excess = [r for r in excess_t1 if not math.isnan(r)]
        valid_gaps = [r for r in abs_gaps if not math.isnan(r)]

        def _median(vals):
            if not vals:
                return None
            s = sorted(vals)
            n = len(s)
            return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

        def _hit_rate(vals):
            if not vals:
                return None
            return round(sum(1 for v in vals if v > 0) / len(vals), 4)

        stats[key_val] = {
            "n": len(outcomes),
            "median_return_t1": round(_median(valid_t1), 4) if valid_t1 else None,
            "median_return_t3": round(_median(valid_t3), 4) if valid_t3 else None,
            "median_return_t5": round(_median(valid_t5), 4) if valid_t5 else None,
            "median_excess_t1": round(_median(valid_excess), 4) if valid_excess else None,
            "hit_rate_t1": _hit_rate(valid_t1),
            "hit_rate_t5": _hit_rate(valid_t5),
            "median_abs_gap": round(_median(valid_gaps), 4) if valid_gaps else None,
        }

    return stats


def find_extremes(
    records: List[Dict[str, Any]],
    n: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """Find largest winners and losers by abs_gap."""
    with_gap = []
    for r in records:
        gap = _sf(r.get("outcome", {}).get("abs_gap"))
        ret = _sf(r.get("outcome", {}).get("return_t1"))
        if not math.isnan(gap) or not math.isnan(ret):
            val = ret if math.isnan(gap) else gap
            with_gap.append((val, r))

    with_gap.sort(key=lambda x: x[0])

    def _summary(record):
        pre = record.get("pre_event", {})
        out = record.get("outcome", {})
        return {
            "ticker": record.get("ticker", ""),
            "event_date": record.get("event_date", ""),
            "tier": pre.get("tier_dev", ""),
            "family": pre.get("catalyst_family", ""),
            "in_shadow": pre.get("in_shadow", False),
            "return_t1": out.get("return_t1"),
            "abs_gap": out.get("abs_gap"),
        }

    return {
        "large_gap_losers": [_summary(r) for _, r in with_gap[:n]],
        "large_gap_winners": [_summary(r) for _, r in with_gap[-n:]],
    }


def build_event_analyst(
    as_of_date: str,
    *,
    artifacts_dir: Path = REPO_ROOT / "artifacts",
    lookback_days: int = DEFAULT_LOOKBACK,
) -> Dict[str, Any]:
    """Build event analyst summary."""
    postmortem_dir = artifacts_dir / "postmortem"
    records = load_postmortems(postmortem_dir, as_of_date, lookback_days)

    if not records:
        return {
            "schema": SCHEMA_VERSION,
            "as_of_date": as_of_date,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_postmortems": 0,
            "status": "NO_DATA",
            "message": "No postmortem records found. Waiting for resolved catalyst events.",
        }

    # Slice by key dimensions
    by_family = compute_slice_stats(records, "catalyst_family")
    by_tier = compute_slice_stats(records, "tier_dev")
    by_shadow = compute_slice_stats(records, "in_shadow")
    by_trade_plan = compute_slice_stats(records, "in_trade_plan")
    by_hard = compute_slice_stats(records, "is_hard_catalyst")

    # Extremes
    extremes = find_extremes(records)

    result = {
        "schema": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": lookback_days,
        "n_postmortems": len(records),
        "status": "OK",
        "by_family": by_family,
        "by_tier": by_tier,
        "by_shadow_membership": by_shadow,
        "by_trade_plan_membership": by_trade_plan,
        "by_hard_catalyst": by_hard,
        "extremes": extremes,
    }

    # Write artifacts
    out_dir = artifacts_dir / "event_analyst"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{as_of_date}_summary.json"
    md_path = out_dir / f"{as_of_date}_summary.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)

    md_path.write_text(format_summary_md(result), encoding="utf-8")
    logger.info("Wrote %s", md_path)

    result["_json_path"] = str(json_path)
    result["_md_path"] = str(md_path)
    return result


def format_summary_md(d: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Event Analyst Summary — {d['as_of_date']}")
    lines.append("")

    if d.get("status") == "NO_DATA":
        lines.append(d.get("message", "No data."))
        lines.append("")
        return "\n".join(lines)

    lines.append(f"**Postmortems analyzed**: {d['n_postmortems']} (lookback {d['lookback_days']}d)")
    lines.append("")

    # Slice tables
    for label, key in [
        ("By Family", "by_family"),
        ("By Tier", "by_tier"),
        ("By Shadow Membership", "by_shadow_membership"),
        ("By Trade Plan Membership", "by_trade_plan_membership"),
        ("By Hard Catalyst", "by_hard_catalyst"),
    ]:
        data = d.get(key, {})
        if not data:
            continue
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| Slice | N | Med T+1 | Med T+5 | Hit T+1 | Hit T+5 | Med Gap |")
        lines.append("|-------|---|---------|---------|---------|---------|---------|")
        for slice_val, stats in data.items():
            n = stats.get("n", 0)
            t1 = f"{stats['median_return_t1']:.2%}" if stats.get("median_return_t1") is not None else "-"
            t5 = f"{stats['median_return_t5']:.2%}" if stats.get("median_return_t5") is not None else "-"
            h1 = f"{stats['hit_rate_t1']:.0%}" if stats.get("hit_rate_t1") is not None else "-"
            h5 = f"{stats['hit_rate_t5']:.0%}" if stats.get("hit_rate_t5") is not None else "-"
            gap = f"{stats['median_abs_gap']:.2%}" if stats.get("median_abs_gap") is not None else "-"
            lines.append(f"| {slice_val} | {n} | {t1} | {t5} | {h1} | {h5} | {gap} |")
        lines.append("")

    # Extremes
    extremes = d.get("extremes", {})
    for label, key in [("Largest Winners", "large_gap_winners"), ("Largest Losers", "large_gap_losers")]:
        items = extremes.get(key, [])
        if items:
            lines.append(f"## {label}")
            lines.append("")
            for item in items:
                ret = f"{item['return_t1']:.1%}" if item.get("return_t1") is not None else "?"
                lines.append(
                    f"- **{item['ticker']}** ({item.get('event_date', '?')}): "
                    f"{ret}, tier {item.get('tier', '?')}, {item.get('family', '?')}, "
                    f"shadow={'yes' if item.get('in_shadow') else 'no'}"
                )
            lines.append("")

    lines.append(f"*Generated: {d.get('generated_at', '')}*")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Event analyst — aggregate postmortem patterns")
    parser.add_argument("--as-of-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()
    started = time.perf_counter()

    result = build_event_analyst(
        args.as_of_date,
        artifacts_dir=args.artifacts_dir,
        lookback_days=args.lookback,
    )

    if result.get("status") == "NO_DATA":
        logger.info("No postmortem data yet")
    else:
        logger.info("Analyst: %d postmortems analyzed", result["n_postmortems"])

    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        exec_id = log_agent_run(
            "build_event_analyst",
            f"Event analyst for {args.as_of_date}",
            inputs={"as_of_date": args.as_of_date, "lookback": args.lookback},
            outputs={"status": result.get("status"), "n_postmortems": result.get("n_postmortems")},
            success=result.get("status") != "NO_DATA",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if exec_id and result.get("status") == "OK":
            attach_outcome_verdict(
                exec_id,
                was_correct=result.get("n_postmortems", 0) > 0,
                evidence=f"status=OK n_postmortems={result.get('n_postmortems', 0)}",
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
