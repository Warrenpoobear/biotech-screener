#!/usr/bin/env python3
"""Post-promotion health monitor for active rulesets.

Compares daily drift metrics against the active ruleset's promotion baseline.
Tracks rolling history in JSONL and recommends rollback after K consecutive WARN days.

Called from run_daily_production.py as a WARN-only gate.

CLI (standalone):
  python tools/ruleset_health_monitor.py \
      --drift-report data/snapshots/2026-02-28/drift_report.json \
      --receipts-dir artifacts/promotions \
      --history-file artifacts/ruleset_health_history.jsonl \
      --output-dir data/snapshots/2026-02-28
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HealthThresholds:
    """Configurable thresholds for ruleset health checks."""

    overlap_warn_delta: float = 10.0
    """WARN if today's top60_overlap < baseline - delta."""

    rank_shift_warn_factor: float = 3.0
    """WARN if today's max_rank_shift > baseline * factor."""

    consecutive_warn_days_for_rollback: int = 3
    """Recommend rollback after this many consecutive WARN days."""


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _manifest_active_id(manifest_path: Path) -> Optional[str]:
    """Return the id of the manifest entry with status == 'active', or None."""
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in data.get("rulesets", []):
            if entry.get("status") == "active":
                return entry.get("id")
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _find_active_receipt(
    receipts_dir: Path,
    active_ruleset_id: Optional[str] = None,
    manifest_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    """Find the promotion receipt for the active ruleset.

    Resolution order for the canonical active id:
      1. Explicit ``active_ruleset_id`` argument (caller override).
      2. ``manifest_path`` -> entry with ``status == "active"`` (canonical SoT).
      3. Fallback: most recent receipt by filename sort (legacy behavior;
         preserved when neither override nor manifest is provided).

    If a canonical id is known but no matching receipt exists in
    ``receipts_dir``, returns a stub ``{"new_active_id": <id>,
    "missing_receipt": True, ...}`` so callers report the right id
    instead of falling back to a stale rollback receipt that happens to
    sort first.
    """
    if not receipts_dir.exists():
        # Even with no receipts dir, if we know the canonical id from the
        # manifest, surface it as a stub.
        if active_ruleset_id is None and manifest_path is not None:
            active_ruleset_id = _manifest_active_id(manifest_path)
        if active_ruleset_id:
            return {
                "schema": "promote_receipt.stub.v1",
                "new_active_id": active_ruleset_id,
                "missing_receipt": True,
                "created_at_utc": "",
                "gate": {},
            }
        return None

    if active_ruleset_id is None and manifest_path is not None:
        active_ruleset_id = _manifest_active_id(manifest_path)

    candidates = sorted(receipts_dir.glob("promotion_*.json"), reverse=True)
    # Also check rollback receipts (they become the new active)
    candidates.extend(sorted(receipts_dir.glob("rollback_*.json"), reverse=True))
    # Re-sort all by name descending (date-based naming)
    candidates.sort(key=lambda p: p.name, reverse=True)

    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if active_ruleset_id:
            if data.get("new_active_id") == active_ruleset_id:
                return data
        else:
            return data

    # Canonical id known but no matching receipt: return explicit stub so
    # the monitor reports the right id with baseline metrics absent, rather
    # than silently surfacing a stale rollback receipt for an old id.
    if active_ruleset_id:
        return {
            "schema": "promote_receipt.stub.v1",
            "new_active_id": active_ruleset_id,
            "missing_receipt": True,
            "created_at_utc": "",
            "gate": {},
        }
    return None


def _load_history(history_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL health history."""
    if not history_path.exists():
        return []
    entries = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipped malformed health history line: %.80s", line)
                continue
    return entries


def _count_consecutive_warns(
    history: List[Dict[str, Any]],
    active_ruleset_id: str,
) -> int:
    """Count consecutive WARN days at the tail of history for the given ruleset."""
    count = 0
    for entry in reversed(history):
        if entry.get("active_ruleset_id") != active_ruleset_id:
            break
        if entry.get("status") == "WARN":
            count += 1
        else:
            break
    return count


def evaluate_health(
    drift_report: Optional[Dict[str, Any]],
    receipt: Optional[Dict[str, Any]],
    history_path: Optional[Path] = None,
    thresholds: Optional[HealthThresholds] = None,
) -> Dict[str, Any]:
    """Evaluate ruleset health. Returns the health check result dict.

    Args:
        drift_report: Parsed drift_report.json (or None if missing)
        receipt: Promotion receipt for active ruleset (or None if not found)
        history_path: Path to JSONL history file (will be appended to)
        thresholds: Configurable thresholds

    Returns:
        Health check result dict matching ruleset_health.v1 schema.
    """
    th = thresholds or HealthThresholds()

    # No receipt → cold start, return PASS
    if not receipt:
        return {
            "schema": "ruleset_health.v1",
            "active_ruleset_id": "",
            "promotion_date": "",
            "days_since_promotion": 0,
            "today": {},
            "promotion_baseline": {},
            "status": "PASS",
            "detail": "No promotion receipt found; health check skipped",
            "consecutive_warn_days": 0,
            "recommend_rollback": False,
        }

    active_id = receipt.get("new_active_id", "")
    promo_date = receipt.get("created_at_utc", "")[:10]
    gate = receipt.get("gate") or {}

    # Stub receipt (manifest knows the canonical id but no real receipt
    # exists): report active_id correctly with baseline absent. Skips
    # threshold evaluation and history append (consistent with cold-start
    # and no-drift early returns).
    if receipt.get("missing_receipt"):
        if receipt.get("receipt_type") == "synthetic_backfill":
            detail = (
                f"Active ruleset {active_id}: synthetic backfilled receipt "
                "present; promotion baseline metrics unavailable"
            )
        else:
            detail = (
                f"Active ruleset {active_id}: no promotion receipt available — "
                "baseline metrics unavailable (consider backfilling receipt)"
            )
        return {
            "schema": "ruleset_health.v1",
            "active_ruleset_id": active_id,
            "promotion_date": "",
            "days_since_promotion": 0,
            "today": {},
            "promotion_baseline": {},
            "status": "PASS",
            "detail": detail,
            "consecutive_warn_days": 0,
            "recommend_rollback": False,
        }

    # Extract baseline metrics from receipt
    baseline_overlap = gate.get("mean_top60_overlap")
    baseline_rank_shift = gate.get("max_rank_shift")
    baseline_turnover = gate.get("mean_turnover")

    promotion_baseline = {
        "mean_top60_overlap": baseline_overlap,
        "max_rank_shift": baseline_rank_shift,
        "mean_turnover": baseline_turnover,
    }

    # No drift report → graceful pass
    if not drift_report:
        return {
            "schema": "ruleset_health.v1",
            "active_ruleset_id": active_id,
            "promotion_date": promo_date,
            "days_since_promotion": 0,
            "today": {},
            "promotion_baseline": promotion_baseline,
            "status": "PASS",
            "detail": "No drift report available; health check skipped",
            "consecutive_warn_days": 0,
            "recommend_rollback": False,
        }

    # Extract today's metrics from drift report
    metrics = drift_report.get("metrics") or {}
    today_overlap = metrics.get("top60_overlap_pct")
    today_rank_shift = metrics.get("mean_abs_rank_delta_top60")

    today_metrics = {
        "top60_overlap_pct": today_overlap,
        "max_rank_shift": today_rank_shift,
    }

    # Days since promotion
    days_since = 0
    if promo_date:
        try:
            promo_dt = datetime.strptime(promo_date, "%Y-%m-%d")
            current_date_str = drift_report.get("current_date", "")
            if current_date_str:
                current_dt = datetime.strptime(current_date_str, "%Y-%m-%d")
                days_since = (current_dt - promo_dt).days
        except (ValueError, TypeError):
            pass

    # Evaluate thresholds
    warn_reasons: List[str] = []

    if baseline_overlap is not None and today_overlap is not None:
        floor = baseline_overlap - th.overlap_warn_delta
        if today_overlap < floor:
            warn_reasons.append(f"top60_overlap {today_overlap:.1f}% < baseline floor {floor:.1f}%")

    if baseline_rank_shift is not None and today_rank_shift is not None:
        ceiling = baseline_rank_shift * th.rank_shift_warn_factor
        if today_rank_shift > ceiling:
            warn_reasons.append(f"rank_shift {today_rank_shift:.1f} > baseline ceiling {ceiling:.1f}")

    status = "WARN" if warn_reasons else "OK"

    # Load history and compute consecutive warns. Same-day reruns replace
    # today's existing entry instead of counting it as an additional day.
    history = _load_history(history_path) if history_path else []
    current_date = drift_report.get("current_date", "")
    history_without_today = [
        entry
        for entry in history
        if not (entry.get("active_ruleset_id") == active_id and entry.get("date") == current_date)
    ]
    consecutive = _count_consecutive_warns(history_without_today, active_id)
    if status == "WARN":
        consecutive += 1  # include today
    else:
        consecutive = 0  # reset on good day

    recommend_rollback = consecutive >= th.consecutive_warn_days_for_rollback

    detail = "; ".join(warn_reasons) if warn_reasons else "within baseline"

    result = {
        "schema": "ruleset_health.v1",
        "active_ruleset_id": active_id,
        "promotion_date": promo_date,
        "days_since_promotion": days_since,
        "today": today_metrics,
        "promotion_baseline": promotion_baseline,
        "status": status,
        "detail": detail,
        "consecutive_warn_days": consecutive,
        "recommend_rollback": recommend_rollback,
    }

    # Append to history
    if history_path is not None:
        history_entry = {
            "date": current_date,
            "active_ruleset_id": active_id,
            "status": status,
            "top60_overlap_pct": today_overlap,
            "max_rank_shift": today_rank_shift,
            "consecutive_warn_days": consecutive,
            "recommend_rollback": recommend_rollback,
        }
        updated_history = history_without_today + [history_entry]
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, "w", encoding="utf-8") as f:
            for entry in updated_history:
                f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    return result


def run_health_check(
    drift_report_path: Optional[Path],
    receipts_dir: Path,
    history_path: Path,
    output_dir: Optional[Path] = None,
    active_ruleset_id: Optional[str] = None,
    thresholds: Optional[HealthThresholds] = None,
    manifest_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Full health check pipeline: load inputs, evaluate, write outputs.

    Returns the health check result dict.
    """
    # Load drift report
    drift_report = None
    if drift_report_path and drift_report_path.exists():
        try:
            drift_report = json.loads(drift_report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Find active receipt (manifest-aware when no explicit override)
    receipt = _find_active_receipt(receipts_dir, active_ruleset_id, manifest_path)

    # Evaluate
    result = evaluate_health(drift_report, receipt, history_path, thresholds)

    # Write sidecar
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "ruleset_health.json"
        out_path.write_text(
            json.dumps(result, indent=2) + "\n",
            encoding="utf-8",
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> int:
    import time

    repo_root = Path(__file__).resolve().parent.parent
    started = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Post-promotion ruleset health monitor.",
    )
    parser.add_argument(
        "--as-of-date",
        default=None,
        help="Snapshot date (YYYY-MM-DD). Sets drift-report and output-dir under data/snapshots/.",
    )
    parser.add_argument(
        "--drift-report",
        type=Path,
        default=None,
        help="Path to drift_report.json from today's snapshot.",
    )
    parser.add_argument(
        "--receipts-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts" / "promotions",
        help="Directory containing promotion receipts.",
    )
    parser.add_argument(
        "--history-file",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "artifacts" / "ruleset_health_history.jsonl",
        help="JSONL history file for rolling tracking.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write ruleset_health.json sidecar.",
    )
    parser.add_argument(
        "--active-ruleset-id",
        default=None,
        help="Active ruleset ID (overrides manifest auto-detection if given).",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "production_data" / "decision_rulesets" / "manifest.json",
        help=(
            "Path to decision_rulesets/manifest.json. When --active-ruleset-id "
            "is not given, the entry with status='active' is used as the canonical "
            "ID. Pass an empty path or a non-existent path to disable manifest fallback."
        ),
    )
    args = parser.parse_args(argv)

    if args.as_of_date:
        snap_dir = repo_root / "data" / "snapshots" / args.as_of_date
        if args.drift_report is None:
            args.drift_report = snap_dir / "drift_report.json"
        if args.output_dir is None:
            args.output_dir = snap_dir

    result = run_health_check(
        drift_report_path=args.drift_report,
        receipts_dir=args.receipts_dir,
        history_path=args.history_file,
        output_dir=args.output_dir,
        active_ruleset_id=args.active_ruleset_id,
        manifest_path=args.manifest_path,
    )

    print(json.dumps(result, indent=2))
    status = result.get("status", "OK")
    recommend_rollback = bool(result.get("recommend_rollback"))
    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        exec_id = log_agent_run(
            "ruleset_health_monitor",
            f"Ruleset health for {result.get('today', {}).get('date', args.as_of_date or 'unknown')}",
            inputs={"as_of_date": args.as_of_date, "drift_report": str(args.drift_report or "")},
            outputs={"status": status, "recommend_rollback": recommend_rollback},
            success=status in ("OK", "PASS"),
            error=None if status in ("OK", "PASS") else status,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if exec_id:
            attach_outcome_verdict(
                exec_id,
                was_correct=status in ("OK", "PASS") and not recommend_rollback,
                evidence=f"status={status} rollback={recommend_rollback}",
            )
    except Exception:
        pass
    if recommend_rollback:
        print(f"\nWARNING: Rollback recommended ({result['consecutive_warn_days']} consecutive WARN days)")
        return 2
    if status == "WARN":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
