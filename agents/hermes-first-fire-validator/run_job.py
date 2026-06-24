#!/usr/bin/env python3
"""
Hermes First-Fire Validator Job

Reads the knowledge layer first-fire ledger and routes validation results to Town operator inbox.
Phase B entry point for Town-Hermes bridge (Spec 090).

Usage:
    python3 agents/hermes-first-fire-validator/run_job.py
"""

import json
import logging
import sys
from pathlib import Path

# Setup
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from common.operator_delivery import send_operator_event  # noqa: E402


def _is_failure_status(status: str) -> bool:
    normalized = status.upper()
    return normalized in {"FAIL", "FAILED"} or normalized.startswith("FAIL_")


def main():
    """Main job: read first-fire ledger, validate, route results to Town."""
    logger.info("Starting hermes-first-fire-validator job")

    # Read the first-fire ledger from knowledge layer
    ledger_path = REPO_ROOT / "artifacts/ops/first_fire_ledger/latest.json"

    if not ledger_path.exists():
        logger.error(f"First-fire ledger not found: {ledger_path}")
        send_operator_event(
            channel="town",
            severity="FAIL",
            event_type="first_fire_fail",
            title="First-fire validation: job FAILED — ledger file missing",
            summary=f"Expected {ledger_path} not found. Knowledge layer may not have run.",
            next_operator_action="investigate"
        )
        return 1

    try:
        with open(ledger_path) as f:
            ledger = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load ledger: {e}")
        send_operator_event(
            channel="town",
            severity="FAIL",
            event_type="first_fire_fail",
            title="First-fire validation: job FAILED — parse error",
            summary=f"Error reading {ledger_path}: {str(e)[:100]}",
            next_operator_action="investigate"
        )
        return 1

    # Extract and classify results
    jobs = ledger.get("jobs", [])
    passes = []
    failures = []

    for job in jobs:
        job_name = job.get("job", "unknown")
        eval_status = str(job.get("eval", job.get("status", "UNKNOWN")))

        if eval_status == "PASS":
            passes.append(job)
            logger.info(f"First-fire PASS: {job_name}")
        elif _is_failure_status(eval_status):
            failures.append(job)
            logger.warning(f"First-fire FAIL: {job_name}")
        else:
            # PENDING, PENDING_NOT_YET_DUE, etc. — don't route (not yet evaluated)
            logger.info(f"First-fire {eval_status}: {job_name} — not yet evaluated")

    # Route PASS events (if any)
    if passes:
        job_names = ", ".join(j.get("job") for j in passes)
        try:
            success = send_operator_event(
                channel="town",
                severity="INFO",
                event_type="first_fire_pass",
                title=f"First-fire validation PASS: {job_names}",
                summary=(
                    f"{len(passes)} job(s) passed first-fire validation. "
                    f"Ready for promotion to production."
                ),
                artifact=str(ledger_path.relative_to(REPO_ROOT)),
                next_operator_action="approve",
                extra={
                    "passed_jobs": [j.get("job") for j in passes],
                    "pass_count": len(passes),
                }
            )
            if success:
                logger.info(f"PASS event routed to Town for {len(passes)} job(s)")
            else:
                logger.error("Failed to route PASS event to Town")
        except Exception as e:
            logger.error(f"Exception during PASS event delivery: {e}", exc_info=True)

    # Route FAIL events (if any)
    if failures:
        job_names = ", ".join(j.get("job") for j in failures)
        try:
            success = send_operator_event(
                channel="town",
                severity="FAIL",
                event_type="first_fire_fail",
                title=f"First-fire validation FAIL: {job_names}",
                summary=(
                    f"{len(failures)} job(s) failed first-fire validation. "
                    f"See ledger for details; correct issues before promotion."
                ),
                artifact=str(ledger_path.relative_to(REPO_ROOT)),
                next_operator_action="investigate",
                extra={
                    "failed_jobs": [j.get("job") for j in failures],
                    "fail_count": len(failures),
                }
            )
            if success:
                logger.info(f"FAIL event routed to Town for {len(failures)} job(s)")
                return 1
            else:
                logger.error("Failed to route FAIL event to Town")
                return 1
        except Exception as e:
            logger.error(f"Exception during FAIL event delivery: {e}", exc_info=True)
            return 1

    # No PASS or FAIL items to route — all pending or awaiting evaluation
    if not passes and not failures:
        logger.info("No first-fire items ready for validation (all PENDING or not yet due)")
        # Route INFO event to indicate no-op
        try:
            send_operator_event(
                channel="town",
                severity="INFO",
                event_type="first_fire_pass",
                title="First-fire validation: no items to evaluate",
                summary="All first-fire items are PENDING or not yet due for evaluation.",
                artifact=str(ledger_path.relative_to(REPO_ROOT)),
                next_operator_action="none",
            )
        except Exception as e:
            logger.warning(f"Failed to route no-op event: {e}")
        return 0

    return 0


if __name__ == "__main__":
    import time

    _started = time.perf_counter()
    _rc = main()
    try:
        from tools.agent_skill_telemetry import log_hermes_job_exit

        log_hermes_job_exit("hermes-first-fire-validator", _rc, _started)
    except Exception:
        pass
    sys.exit(_rc)
