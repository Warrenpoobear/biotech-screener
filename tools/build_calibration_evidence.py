#!/usr/bin/env python3
"""Calibration evidence accumulator — structured feedback loop for governance.

Reads resolved postmortem artifacts, pulls pre-event DEM state (score
components, sort contributions, tier, signals), compares to post-event
outcomes, and writes structured evidence records.

Produces three outputs:
  1. Signal contribution tracker — which signals earned their weight
  2. Threshold audit log — gates that excluded winners or included losers
  3. Prediction calibration curve — hit rates by score decile

Read-only — produces evidence, never recommendations.

Output:
    artifacts/calibration_evidence/{date}_evidence.json
    artifacts/calibration_evidence/{date}_evidence.md
    artifacts/calibration_evidence/ledger.jsonl (append)

Usage:
    python tools/build_calibration_evidence.py --as-of-date 2026-04-15
"""

from __future__ import annotations

import argparse
import csv
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
logger = logging.getLogger("calibration_evidence")

SCHEMA_VERSION = "calibration_evidence.v1"

# Sort contribution fields from decision engine
SORT_CONTRIB_FIELDS = [
    "de_sort_contrib_calendar_alpha",
    "de_sort_contrib_institutional",
    "de_sort_contrib_binary_quality",
    "de_sort_contrib_clinical_quality",
]

# Score component fields
SCORE_COMPONENTS = [
    "clinical_score",
    "clinical_optionality_pct_dev",
    "clinical_score_v2_z",
    "inst_delta_z",
    "binary_quality_score",
]

# Threshold gates that might exclude names
GATE_FIELDS = [
    ("eligible", "1", "eligibility"),
    ("drawdown_gate_hit", "1", "drawdown_gate"),
    ("financials_missing", "1", "financials_gate"),
]


def _sf(val: Any) -> float:
    if val is None or val == "":
        return math.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return math.nan


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_postmortems(postmortem_dir: Path, as_of_date: str) -> List[Dict]:
    """Load all postmortem records up to as_of_date."""
    records = []
    if not postmortem_dir.exists():
        return records
    for date_dir in sorted(postmortem_dir.iterdir()):
        if not date_dir.is_dir() or date_dir.name > as_of_date:
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


def load_snapshot_row(
    snapshots_dir: Path,
    snapshot_date: str,
    ticker: str,
) -> Dict[str, str]:
    """Load a single ticker's row from a snapshot's rankings.csv."""
    path = snapshots_dir / snapshot_date / "rankings.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("ticker") == ticker:
                return row
    return {}


# ---------------------------------------------------------------------------
# 1. Signal contribution tracker
# ---------------------------------------------------------------------------
def build_signal_tracker(
    postmortems: List[Dict],
    snapshots_dir: Path,
) -> Dict[str, Any]:
    """For each resolved event, track which signals contributed most."""
    records = []
    signal_outcomes: Dict[str, List[tuple]] = defaultdict(list)

    for pm in postmortems:
        ticker = pm.get("ticker", "")
        pre = pm.get("pre_event", {})
        outcome = pm.get("outcome", {})
        snap_date = pre.get("snapshot_date", "")

        ret_t5 = _sf(outcome.get("return_t5"))
        if math.isnan(ret_t5) or not snap_date:
            continue

        # Load full rankings row for sort contributions
        row = load_snapshot_row(snapshots_dir, snap_date, ticker)
        if not row:
            continue

        # Extract sort contributions
        contribs = {}
        for field in SORT_CONTRIB_FIELDS:
            val = _sf(row.get(field, ""))
            if not math.isnan(val):
                contribs[field.replace("de_sort_contrib_", "")] = val

        # Extract score components
        components = {}
        for field in SCORE_COMPONENTS:
            val = _sf(row.get(field, ""))
            if not math.isnan(val):
                components[field] = val

        record = {
            "ticker": ticker,
            "event_date": pm.get("event_date", ""),
            "return_t5": round(ret_t5, 4),
            "tier": pre.get("tier_dev", ""),
            "rank": pre.get("actionable_rank"),
            "in_shadow": pre.get("in_shadow", False),
            "sort_contributions": contribs,
            "score_components": components,
        }
        records.append(record)

        # Accumulate per-signal IC data
        win = ret_t5 > 0
        for sig, val in components.items():
            signal_outcomes[sig].append((val, ret_t5, win))

    # Compute per-signal summary
    signal_summary = {}
    for sig, data in signal_outcomes.items():
        if len(data) < 5:
            signal_summary[sig] = {"n": len(data), "status": "insufficient_data"}
            continue

        vals = [d[0] for d in data]
        rets = [d[1] for d in data]

        # Simple rank IC
        from scipy import stats

        ic, _ = stats.spearmanr(vals, rets) if len(vals) >= 10 else (math.nan, 0)

        # Split by median signal value
        median_val = sorted(vals)[len(vals) // 2]
        high_rets = [r for v, r, _ in data if v >= median_val]
        low_rets = [r for v, r, _ in data if v < median_val]

        signal_summary[sig] = {
            "n": len(data),
            "ic": round(ic, 4) if not math.isnan(ic) else None,
            "high_signal_median_ret": round(sorted(high_rets)[len(high_rets) // 2], 4) if high_rets else None,
            "low_signal_median_ret": round(sorted(low_rets)[len(low_rets) // 2], 4) if low_rets else None,
            "spread": (
                round((sorted(high_rets)[len(high_rets) // 2] - sorted(low_rets)[len(low_rets) // 2]), 4)
                if high_rets and low_rets
                else None
            ),
        }

    return {
        "n_events": len(records),
        "signal_summary": signal_summary,
        "records": records[:50],  # cap for file size
    }


# ---------------------------------------------------------------------------
# 2. Threshold audit log
# ---------------------------------------------------------------------------
def build_threshold_audit(
    postmortems: List[Dict],
    snapshots_dir: Path,
) -> List[Dict[str, Any]]:
    """Find cases where gates excluded eventual winners or included losers."""
    misses = []

    for pm in postmortems:
        ticker = pm.get("ticker", "")
        pre = pm.get("pre_event", {})
        outcome = pm.get("outcome", {})
        snap_date = pre.get("snapshot_date", "")

        ret_t5 = _sf(outcome.get("return_t5"))
        if math.isnan(ret_t5) or not snap_date:
            continue

        row = load_snapshot_row(snapshots_dir, snap_date, ticker)
        if not row:
            continue

        eligible = row.get("eligible", "")
        tier = row.get("tier_dev", "")
        rank = _sf(row.get("actionable_rank", ""))

        # Excluded winners: not eligible or low-tier but had big positive return
        if ret_t5 > 0.10 and (eligible != "1" or tier in ("D", "E", "")):
            reasons = []
            if eligible != "1":
                reasons.append("ineligible")
            if tier in ("D", "E", ""):
                reasons.append(f"tier={tier}")
            misses.append(
                {
                    "type": "excluded_winner",
                    "ticker": ticker,
                    "event_date": pm.get("event_date", ""),
                    "return_t5": round(ret_t5, 4),
                    "tier": tier,
                    "eligible": eligible,
                    "reasons": reasons,
                }
            )

        # Included losers: top-20 but big negative return
        if ret_t5 < -0.10 and not math.isnan(rank) and rank <= 20:
            misses.append(
                {
                    "type": "included_loser",
                    "ticker": ticker,
                    "event_date": pm.get("event_date", ""),
                    "return_t5": round(ret_t5, 4),
                    "rank": int(rank),
                    "tier": tier,
                }
            )

    return misses


# ---------------------------------------------------------------------------
# 3. Prediction calibration curve
# ---------------------------------------------------------------------------
def build_calibration_curve(
    postmortems: List[Dict],
) -> Dict[str, Any]:
    """Bin names by pre-event rank decile and compute hit rates."""
    from collections import defaultdict

    decile_data: Dict[int, List[float]] = defaultdict(list)

    for pm in postmortems:
        pre = pm.get("pre_event", {})
        outcome = pm.get("outcome", {})
        rank = pre.get("actionable_rank")
        ret_t5 = _sf(outcome.get("return_t5"))

        if rank is None or math.isnan(ret_t5):
            continue

        try:
            rank_int = int(float(rank))
        except (ValueError, TypeError):
            continue

        # Decile: 1-10 = top decile, 11-20 = second, etc.
        decile = min(10, max(1, (rank_int - 1) // 10 + 1))
        decile_data[decile].append(ret_t5)

    curve = {}
    for decile in sorted(decile_data):
        rets = decile_data[decile]
        hits = sum(1 for r in rets if r > 0)
        curve[f"decile_{decile}"] = {
            "n": len(rets),
            "hit_rate": round(hits / len(rets), 4) if rets else None,
            "median_return": round(sorted(rets)[len(rets) // 2], 4) if rets else None,
            "mean_return": round(sum(rets) / len(rets), 4) if rets else None,
        }

    return curve


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_calibration_evidence(
    as_of_date: str,
    *,
    artifacts_dir: Path = REPO_ROOT / "artifacts",
    snapshots_dir: Path = REPO_ROOT / "data" / "snapshots",
) -> Dict[str, Any]:
    """Build calibration evidence artifact."""
    postmortem_dir = artifacts_dir / "postmortem"
    postmortems = load_postmortems(postmortem_dir, as_of_date)

    if not postmortems:
        return {
            "schema": SCHEMA_VERSION,
            "as_of_date": as_of_date,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_postmortems": 0,
            "status": "NO_DATA",
            "message": "No postmortem records. Waiting for resolved catalyst events.",
        }

    # Build all three evidence types
    signal_tracker = build_signal_tracker(postmortems, snapshots_dir)
    threshold_audit = build_threshold_audit(postmortems, snapshots_dir)
    calibration_curve = build_calibration_curve(postmortems)

    result = {
        "schema": SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_postmortems": len(postmortems),
        "status": "OK",
        "signal_tracker": signal_tracker,
        "threshold_audit": threshold_audit,
        "calibration_curve": calibration_curve,
    }

    # Write artifacts
    out_dir = artifacts_dir / "calibration_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{as_of_date}_evidence.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Wrote %s", json_path)

    md_path = out_dir / f"{as_of_date}_evidence.md"
    md_path.write_text(format_evidence_md(result), encoding="utf-8")
    logger.info("Wrote %s", md_path)

    # Append to ledger
    ledger_path = out_dir / "ledger.jsonl"
    ledger_entry = {
        "date": as_of_date,
        "n_postmortems": len(postmortems),
        "n_signals_tracked": len(signal_tracker.get("signal_summary", {})),
        "n_threshold_misses": len(threshold_audit),
        "n_calibration_deciles": len(calibration_curve),
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(ledger_entry, default=str) + "\n")

    result["_json_path"] = str(json_path)
    return result


def format_evidence_md(d: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Calibration Evidence — {d['as_of_date']}")
    lines.append("")

    if d.get("status") == "NO_DATA":
        lines.append(d.get("message", "No data."))
        return "\n".join(lines)

    lines.append(f"**Postmortems analyzed**: {d['n_postmortems']}")
    lines.append("")

    # Signal tracker
    tracker = d.get("signal_tracker", {})
    signal_summary = tracker.get("signal_summary", {})
    if signal_summary:
        lines.append("## Signal Contribution Tracker")
        lines.append("")
        lines.append("| Signal | N | IC | High Med | Low Med | Spread |")
        lines.append("|--------|---|-----|----------|---------|--------|")
        for sig, stats in signal_summary.items():
            if stats.get("status") == "insufficient_data":
                lines.append(f"| {sig} | {stats['n']} | - | - | - | insufficient |")
            else:
                ic = f"{stats['ic']:+.4f}" if stats.get("ic") is not None else "-"
                high = (
                    f"{stats['high_signal_median_ret']:.2%}" if stats.get("high_signal_median_ret") is not None else "-"
                )
                low = f"{stats['low_signal_median_ret']:.2%}" if stats.get("low_signal_median_ret") is not None else "-"
                spread = f"{stats['spread']:.2%}" if stats.get("spread") is not None else "-"
                lines.append(f"| {sig} | {stats['n']} | {ic} | {high} | {low} | {spread} |")
        lines.append("")

    # Threshold audit
    audit = d.get("threshold_audit", [])
    if audit:
        lines.append("## Threshold Audit")
        lines.append("")
        excluded = [a for a in audit if a["type"] == "excluded_winner"]
        included = [a for a in audit if a["type"] == "included_loser"]
        if excluded:
            lines.append(f"### Excluded Winners ({len(excluded)})")
            lines.append("")
            for a in excluded:
                lines.append(
                    f"- **{a['ticker']}** ({a['event_date']}): +{a['return_t5']:.1%} T+5, {', '.join(a.get('reasons', []))}"
                )
            lines.append("")
        if included:
            lines.append(f"### Included Losers ({len(included)})")
            lines.append("")
            for a in included:
                lines.append(f"- **{a['ticker']}** ({a['event_date']}): {a['return_t5']:.1%} T+5, rank {a['rank']}")
            lines.append("")

    # Calibration curve
    curve = d.get("calibration_curve", {})
    if curve:
        lines.append("## Calibration Curve (by rank decile)")
        lines.append("")
        lines.append("| Decile | N | Hit Rate | Median Return | Mean Return |")
        lines.append("|--------|---|----------|---------------|-------------|")
        for decile, stats in sorted(curve.items()):
            hr = f"{stats['hit_rate']:.0%}" if stats.get("hit_rate") is not None else "-"
            med = f"{stats['median_return']:.2%}" if stats.get("median_return") is not None else "-"
            mean = f"{stats['mean_return']:.2%}" if stats.get("mean_return") is not None else "-"
            lines.append(f"| {decile} | {stats['n']} | {hr} | {med} | {mean} |")
        lines.append("")

    lines.append(f"*Generated: {d.get('generated_at', '')}*")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Calibration evidence accumulator")
    parser.add_argument("--as-of-date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    args = parser.parse_args()
    started = time.perf_counter()

    result = build_calibration_evidence(args.as_of_date)
    if result.get("status") == "NO_DATA":
        logger.info("No postmortem data yet")
    else:
        logger.info(
            "Evidence: %d postmortems, %d signals tracked",
            result["n_postmortems"],
            len(result.get("signal_tracker", {}).get("signal_summary", {})),
        )
    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        status = result.get("status")
        n_pm = result.get("n_postmortems", 0)
        exec_id = log_agent_run(
            "build_calibration_evidence",
            f"Calibration evidence for {args.as_of_date}",
            inputs={"as_of_date": args.as_of_date},
            outputs={"status": status, "n_postmortems": n_pm},
            success=status != "NO_DATA",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if exec_id and status == "OK":
            attach_outcome_verdict(
                exec_id,
                was_correct=n_pm > 0,
                evidence=f"status={status} n_postmortems={n_pm}",
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
