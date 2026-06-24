#!/usr/bin/env python3
"""
Hermes Contradiction Detector — Town bridge (Spec 090 Phase B).

Reads knowledge-layer warnings and routes HARD_CONTRADICTION items to Town.
Typically invoked after tools/build_hermes_knowledge_layer.py.

Usage:
    python3 agents/hermes-contradiction-detector/run_job.py
    python3 agents/hermes-contradiction-detector/run_job.py --from-build
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

STATE_PATH = REPO_ROOT / "artifacts" / "ops" / "knowledge_layer" / "latest_state.json"


def _warnings_from_state() -> list[dict]:
    if not STATE_PATH.exists():
        logger.error("Knowledge layer state not found: %s", STATE_PATH)
        return []
    try:
        state = json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse %s: %s", STATE_PATH, exc)
        return []
    return list(state.get("warnings") or [])


def main() -> int:
    parser = argparse.ArgumentParser(description="Route hard contradictions to Town")
    parser.add_argument(
        "--from-build",
        action="store_true",
        help="No-op flag for cron wrappers; reads latest_state.json only.",
    )
    parser.parse_args()

    from common.town_bridge_events import notify_hard_contradictions

    warnings = _warnings_from_state()
    hard = [w for w in warnings if w.get("severity") == "HARD_CONTRADICTION"]
    if not hard:
        logger.info("No hard contradictions in knowledge layer state")
        return 0

    logger.info("Found %d hard contradiction(s); routing to Town", len(hard))
    ok = notify_hard_contradictions(warnings)
    return 0 if ok else 1


if __name__ == "__main__":
    import time

    _started = time.perf_counter()
    _rc = main()
    try:
        from tools.agent_skill_telemetry import log_hermes_job_exit

        log_hermes_job_exit("hermes-contradiction-detector", _rc, _started)
    except Exception:
        pass
    raise SystemExit(_rc)
