#!/usr/bin/env python3
"""
Hermes Held-Spec Ledger Job

Reads the knowledge layer held specs ledger and routes to Town operator inbox.
Phase B entry point for Town-Hermes bridge (Spec 090).

Usage:
    python3 agents/hermes-held-spec-ledger/run_job.py
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


APPROVED_STATUSES = {"APPROVED", "CLEARED", "RELEASED"}
BLOCKED_STATUSES = {"HELD", "HELD_SUPPRESSED", "SPEC_REQUIRED", "BLOCKED"}
WAITING_STATUSES = {"AWAITING_FIRST_FIRE", "WAITING_CLEARANCE", "NEEDS_OPERATOR_DECISION"}


def _status_for(spec: dict) -> str:
    return str(spec.get("state") or spec.get("status") or "").upper()


def main():
    """Main job: read held specs, route to Town."""
    logger.info("Starting hermes-held-spec-ledger job")

    # Read the held spec ledger from knowledge layer
    ledger_path = REPO_ROOT / "artifacts/ops/held_spec_ledger/latest.json"

    if not ledger_path.exists():
        logger.error(f"Held spec ledger not found: {ledger_path}")
        send_operator_event(
            channel="town",
            severity="FAIL",
            event_type="held_spec_ledger",
            title="Held spec ledger: job FAILED — ledger file missing",
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
            event_type="held_spec_ledger",
            title="Held spec ledger: job FAILED — parse error",
            summary=f"Error reading {ledger_path}: {str(e)[:100]}",
            next_operator_action="investigate"
        )
        return 1

    # The knowledge-layer builder writes `items`; older ledgers used `held_specs`.
    held_specs = ledger.get("items", ledger.get("held_specs", []))
    approved = [s for s in held_specs if _status_for(s) in APPROVED_STATUSES]
    blocked = [s for s in held_specs if _status_for(s) in BLOCKED_STATUSES]
    waiting = [s for s in held_specs if _status_for(s) in WAITING_STATUSES]

    logger.info(f"Held specs: {len(held_specs)} total ({len(approved)} approved, {len(blocked)} blocked, {len(waiting)} waiting)")

    # Route to Town
    try:
        success = send_operator_event(
            channel="town",
            severity="INFO",
            event_type="held_spec_ledger",
            title=f"Held specs ledger: {len(held_specs)} specs ({len(waiting)} awaiting clearance)",
            summary=(
                f"{len(approved)} approved, {len(blocked)} blocked, {len(waiting)} awaiting clearance. "
                f"See artifact for full ledger."
            ),
            artifact=str(ledger_path.relative_to(REPO_ROOT)),
            next_operator_action="review",
            extra={
                "total_specs": len(held_specs),
                "approved": len(approved),
                "blocked": len(blocked),
                "waiting_clearance": len(waiting),
            }
        )

        if success:
            logger.info("Event routed to Town successfully")
            return 0
        else:
            logger.error("Failed to route event to Town")
            return 1

    except Exception as e:
        logger.error(f"Exception during event delivery: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    import time

    _started = time.perf_counter()
    _rc = main()
    try:
        from tools.agent_skill_telemetry import log_hermes_job_exit

        log_hermes_job_exit("hermes-held-spec-ledger", _rc, _started)
    except Exception:
        pass
    sys.exit(_rc)
