#!/usr/bin/env python3
"""Build AACT trial delta features for within-top-30 ranking.

For each ticker, computes execution signals from trial timeline changes:
  1. pcd_shift_days     — net primary completion date shift (positive = delayed)
  2. enrollment_delta   — enrollment change vs prior snapshot
  3. results_posted     — new results posted since prior snapshot
  4. status_upgrades    — trials moving to active/recruiting/completed
  5. status_downgrades  — trials moving to terminated/withdrawn/suspended
  6. execution_score    — composite: on-time + growing enrollment + results = good

These features vary meaningfully inside the DEM top-30 because trial
execution is independent of DEM's scoring (which uses catalyst proximity
+ financial health, not trial execution quality).

Usage:
    python tools/build_aact_trial_deltas.py --as-of-date 2026-04-02
    python tools/build_aact_trial_deltas.py --as-of-date 2026-04-02 --prior 2026-04-01
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
AACT_DIR = REPO_ROOT / "data" / "aact" / "snapshots"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "aact_deltas"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("aact_deltas")

SCHEMA = "aact_trial_deltas.v1"

# Status classifications
ACTIVE_STATUSES = frozenset(
    {
        "RECRUITING",
        "ACTIVE_NOT_RECRUITING",
        "ENROLLING_BY_INVITATION",
        "AVAILABLE",
        "ACTIVE, NOT YET RECRUITING",
    }
)
COMPLETED_STATUSES = frozenset({"COMPLETED"})
NEGATIVE_STATUSES = frozenset(
    {
        "TERMINATED",
        "WITHDRAWN",
        "SUSPENDED",
        "NO_LONGER_AVAILABLE",
    }
)

# Phase rankings (higher = later stage)
PHASE_RANK = {
    "Phase 1": 1,
    "Phase 1/Phase 2": 1.5,
    "PHASE1": 1,
    "Phase 2": 2,
    "Phase 2/Phase 3": 2.5,
    "PHASE2": 2,
    "Phase 3": 3,
    "PHASE3": 3,
    "Phase 4": 4,
    "PHASE4": 4,
}


def load_trial_master(snapshot_date: str) -> dict[str, list[dict]]:
    """Load trial master and index by mapped_ticker."""
    path = AACT_DIR / snapshot_date / "trial_master.json"
    if not path.exists():
        log.warning("No trial master for %s", snapshot_date)
        return {}
    data = json.loads(path.read_text())
    by_ticker: dict[str, list[dict]] = defaultdict(list)
    for trial in data.get("trials", []):
        ticker = trial.get("mapped_ticker")
        if ticker and trial.get("mapping_confidence") in ("high", "medium"):
            by_ticker[ticker].append(trial)
    return dict(by_ticker)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except (ValueError, TypeError):
        return None


def _phase_rank(trial: dict) -> float:
    return PHASE_RANK.get(trial.get("phase", ""), 0)


def compute_ticker_deltas(
    ticker: str,
    current_trials: list[dict],
    prior_trials: list[dict],
    as_of: date,
) -> dict[str, Any]:
    """Compute delta features for a single ticker between two snapshots."""
    prior_by_nct = {t["nct_id"]: t for t in prior_trials}
    current_by_nct = {t["nct_id"]: t for t in current_trials}

    # --- PCD shifts ---
    pcd_shifts = []
    for nct_id, cur in current_by_nct.items():
        prior = prior_by_nct.get(nct_id)
        if not prior:
            continue
        cur_pcd = _parse_date(cur.get("primary_completion_date"))
        prior_pcd = _parse_date(prior.get("primary_completion_date"))
        if cur_pcd and prior_pcd and cur_pcd != prior_pcd:
            shift = (cur_pcd - prior_pcd).days
            pcd_shifts.append(
                {
                    "nct_id": nct_id,
                    "shift_days": shift,
                    "phase": cur.get("phase", ""),
                }
            )

    # --- Enrollment changes ---
    enrollment_deltas = []
    for nct_id, cur in current_by_nct.items():
        prior = prior_by_nct.get(nct_id)
        if not prior:
            continue
        cur_enroll = cur.get("enrollment") or 0
        prior_enroll = prior.get("enrollment") or 0
        if cur_enroll > 0 and prior_enroll > 0 and cur_enroll != prior_enroll:
            pct_change = (cur_enroll - prior_enroll) / prior_enroll
            if abs(pct_change) >= 0.05:  # 5% threshold
                enrollment_deltas.append(
                    {
                        "nct_id": nct_id,
                        "old_enrollment": prior_enroll,
                        "new_enrollment": cur_enroll,
                        "pct_change": round(pct_change, 4),
                    }
                )

    # --- Results posted ---
    newly_posted = []
    for nct_id, cur in current_by_nct.items():
        prior = prior_by_nct.get(nct_id)
        cur_has = cur.get("has_results", False)
        prior_has = prior.get("has_results", False) if prior else False
        if cur_has and not prior_has:
            newly_posted.append(
                {
                    "nct_id": nct_id,
                    "results_first_posted": cur.get("results_first_posted_date"),
                    "phase": cur.get("phase", ""),
                }
            )

    # --- Status transitions ---
    upgrades = []
    downgrades = []
    for nct_id, cur in current_by_nct.items():
        prior = prior_by_nct.get(nct_id)
        if not prior:
            continue
        cur_status = (cur.get("overall_status") or "").upper()
        prior_status = (prior.get("overall_status") or "").upper()
        if cur_status == prior_status:
            continue
        if cur_status in ACTIVE_STATUSES or cur_status in COMPLETED_STATUSES:
            if prior_status not in ACTIVE_STATUSES and prior_status not in COMPLETED_STATUSES:
                upgrades.append(
                    {
                        "nct_id": nct_id,
                        "old_status": prior_status,
                        "new_status": cur_status,
                    }
                )
        if cur_status in NEGATIVE_STATUSES:
            if prior_status not in NEGATIVE_STATUSES:
                downgrades.append(
                    {
                        "nct_id": nct_id,
                        "old_status": prior_status,
                        "new_status": cur_status,
                    }
                )

    # --- New trials ---
    new_trials = [nct_id for nct_id in current_by_nct if nct_id not in prior_by_nct]

    # --- Pipeline depth ---
    active_trials = [t for t in current_trials if (t.get("overall_status") or "").upper() in ACTIVE_STATUSES]
    late_stage = [t for t in active_trials if _phase_rank(t) >= 2.5]

    # --- Aggregate PCD shift ---
    total_pcd_shift = sum(s["shift_days"] for s in pcd_shifts)
    n_delayed = sum(1 for s in pcd_shifts if s["shift_days"] > 0)
    n_accelerated = sum(1 for s in pcd_shifts if s["shift_days"] < 0)

    # --- Execution score ---
    # Positive: results posted, status upgrades, enrollment growth, PCD acceleration
    # Negative: status downgrades, PCD delays, enrollment shrinkage
    score = 0.0
    score += len(newly_posted) * 2.0  # results are the strongest signal
    score += len(upgrades) * 1.0
    score -= len(downgrades) * 1.5
    score -= n_delayed * 0.5
    score += n_accelerated * 0.5
    # Enrollment growth
    for ed in enrollment_deltas:
        if ed["pct_change"] > 0:
            score += 0.3
        else:
            score -= 0.3
    # Late-stage depth bonus
    score += min(len(late_stage), 3) * 0.2

    return {
        "ticker": ticker,
        "n_trials_current": len(current_trials),
        "n_trials_prior": len(prior_trials),
        "n_active": len(active_trials),
        "n_late_stage_active": len(late_stage),
        "n_new_trials": len(new_trials),
        # PCD
        "n_pcd_shifts": len(pcd_shifts),
        "n_pcd_delayed": n_delayed,
        "n_pcd_accelerated": n_accelerated,
        "total_pcd_shift_days": total_pcd_shift,
        "pcd_shifts": pcd_shifts,
        # Enrollment
        "n_enrollment_changes": len(enrollment_deltas),
        "enrollment_deltas": enrollment_deltas,
        # Results
        "n_results_posted": len(newly_posted),
        "results_posted": newly_posted,
        # Status
        "n_status_upgrades": len(upgrades),
        "n_status_downgrades": len(downgrades),
        "status_upgrades": upgrades,
        "status_downgrades": downgrades,
        # Composite
        "execution_score": round(score, 2),
    }


def build_deltas(as_of_date: str, prior_date: str | None = None) -> dict:
    """Build delta features for all mapped tickers."""
    # Find prior snapshot
    if not prior_date:
        available = sorted(
            d.name
            for d in AACT_DIR.iterdir()
            if d.is_dir() and d.name < as_of_date and (d / "trial_master.json").exists()
        )
        if not available:
            return {"error": "No prior AACT snapshot found"}
        prior_date = available[-1]

    log.info("Computing deltas: %s vs %s", as_of_date, prior_date)

    current = load_trial_master(as_of_date)
    prior = load_trial_master(prior_date)

    if not current:
        return {"error": f"No current trial master for {as_of_date}"}

    all_tickers = sorted(set(current.keys()) | set(prior.keys()))

    results = []
    for ticker in all_tickers:
        cur_trials = current.get(ticker, [])
        prior_trials = prior.get(ticker, [])
        if not cur_trials and not prior_trials:
            continue
        delta = compute_ticker_deltas(
            ticker,
            cur_trials,
            prior_trials,
            date.fromisoformat(as_of_date),
        )
        results.append(delta)

    # Sort by execution_score descending
    results.sort(key=lambda x: x["execution_score"], reverse=True)

    # Summary
    n_with_activity = sum(1 for r in results if r["execution_score"] != 0)
    n_results_posted = sum(r["n_results_posted"] for r in results)
    n_pcd_shifts = sum(r["n_pcd_shifts"] for r in results)
    n_downgrades = sum(r["n_status_downgrades"] for r in results)

    return {
        "schema": SCHEMA,
        "as_of_date": as_of_date,
        "prior_date": prior_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_tickers": len(results),
        "n_with_activity": n_with_activity,
        "n_results_posted_total": n_results_posted,
        "n_pcd_shifts_total": n_pcd_shifts,
        "n_downgrades_total": n_downgrades,
        "tickers": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Build AACT trial delta features")
    parser.add_argument("--as-of-date", default=date.today().isoformat())
    parser.add_argument("--prior", default=None, help="Prior snapshot date (auto-detected if omitted)")
    args = parser.parse_args()
    started = time.perf_counter()

    result = build_deltas(args.as_of_date, args.prior)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        try:
            from tools.agent_skill_telemetry import log_agent_run

            log_agent_run(
                "build_aact_trial_deltas",
                f"AACT deltas for {args.as_of_date}",
                inputs={"as_of_date": args.as_of_date},
                outputs={"error": result["error"]},
                success=False,
                error=result["error"],
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception:
            pass
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"aact_deltas_{args.as_of_date}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))

    print(f"AACT TRIAL DELTAS — {args.as_of_date} vs {result['prior_date']}")
    print(f"  Tickers: {result['n_tickers']}")
    print(f"  With activity: {result['n_with_activity']}")
    print(f"  Results posted: {result['n_results_posted_total']}")
    print(f"  PCD shifts: {result['n_pcd_shifts_total']}")
    print(f"  Status downgrades: {result['n_downgrades_total']}")

    # Show top movers
    active = [t for t in result["tickers"] if t["execution_score"] != 0]
    if active:
        print("\n  Top execution movers:")
        for t in active[:10]:
            print(
                f"    {t['ticker']:6s}  score={t['execution_score']:+.1f}  "
                f"results={t['n_results_posted']}  pcd={t['n_pcd_shifts']}  "
                f"up={t['n_status_upgrades']}  down={t['n_status_downgrades']}"
            )

    print(f"\n  Saved: {out_path}")

    try:
        from tools.agent_skill_telemetry import log_agent_run
        from tools.record_skill_feedback import attach_outcome_verdict

        n_tickers = result.get("n_tickers", 0)
        exec_id = log_agent_run(
            "build_aact_trial_deltas",
            f"AACT deltas for {args.as_of_date}",
            inputs={"as_of_date": args.as_of_date, "prior_date": result.get("prior_date")},
            outputs={
                "n_tickers": n_tickers,
                "n_with_activity": result.get("n_with_activity"),
            },
            success=True,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if exec_id:
            attach_outcome_verdict(
                exec_id,
                was_correct=n_tickers > 0,
                evidence=f"n_tickers={n_tickers} n_with_activity={result.get('n_with_activity', 0)}",
            )
    except Exception:
        pass


if __name__ == "__main__":
    main()
