#!/usr/bin/env python3
"""
Hermes Ruleset Integrity Job

Validates that CLAUDE.md declarations match actual runtime configuration.
Routes validation results to Town operator inbox (Spec 090 Phase B).

Usage:
    python3 agents/hermes-ruleset-integrity/run_job.py
"""

import json
import logging
import re
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

from common.operator_delivery import send_operator_event


def extract_declared_ruleset_from_claude():
    """Extract active ruleset ID from CLAUDE.md."""
    claude_path = REPO_ROOT / "CLAUDE.md"

    if not claude_path.exists():
        return None, f"CLAUDE.md not found at {claude_path}"

    try:
        with open(claude_path) as f:
            content = f.read()

        # Look for: Current: `8887576e` (v1.14.0)
        match = re.search(r"Current:\s+`([a-f0-9]{8})`", content)
        if match:
            return match.group(1), None

        return None, "No 'Current:' declaration found in CLAUDE.md"

    except Exception as e:
        return None, f"Failed to read CLAUDE.md: {str(e)[:100]}"


def extract_pinned_ruleset_from_file(filepath):
    """Extract pinned ruleset ID from Python source file."""
    if not filepath.exists():
        return None

    try:
        with open(filepath) as f:
            content = f.read()

        # Look for patterns like: ACTIVE_RULESET_ID = "8887576e"
        match = re.search(r"ACTIVE_RULESET_ID\s*=\s*[\"']([a-f0-9]{8})[\"']", content)
        if match:
            return match.group(1)

        # Also look for: RULESET_ID = "8887576e"
        match = re.search(r"RULESET_ID\s*=\s*[\"']([a-f0-9]{8})[\"']", content)
        if match:
            return match.group(1)

        return None

    except Exception as e:
        logger.warning(f"Failed to read {filepath}: {e}")
        return None


def check_ruleset_file_exists(ruleset_id):
    """Check if ruleset JSON file exists."""
    ruleset_dir = REPO_ROOT / "production_data" / "decision_rulesets"

    if not ruleset_dir.exists():
        return False, f"Ruleset directory not found: {ruleset_dir}"

    # Expected file pattern: v1.14.0_coinvest_only_selector.json (with ID in manifest)
    ruleset_file = ruleset_dir / "manifest.json"

    if not ruleset_file.exists():
        return False, "Ruleset manifest.json not found"

    try:
        with open(ruleset_file) as f:
            manifest = json.load(f)

        # Check if declared ID exists in manifest
        for entry in manifest.get("rulesets", []):
            if entry.get("id") == ruleset_id:
                return True, None

        return False, f"Ruleset ID {ruleset_id} not found in manifest.json"

    except Exception as e:
        return False, f"Failed to read manifest.json: {str(e)[:100]}"


def main():
    """Main job: validate ruleset integrity."""
    logger.info("Starting hermes-ruleset-integrity job")

    checks = {
        "claude_declared": None,
        "run_screen_pinned": None,
        "phase2_pinned": None,
        "manifest_has_id": None,
        "all_match": False,
    }

    # Check 1: Extract declared ruleset from CLAUDE.md
    declared_id, error = extract_declared_ruleset_from_claude()
    if error:
        logger.error(f"Check 1 FAILED: {error}")
        checks["claude_declared"] = ("FAIL", error)
    else:
        logger.info(f"Check 1 PASS: CLAUDE.md declares {declared_id}")
        checks["claude_declared"] = ("PASS", declared_id)

    # If CLAUDE.md check fails, route FAIL event immediately
    if checks["claude_declared"][0] == "FAIL":
        send_operator_event(
            channel="town",
            severity="FAIL",
            event_type="ruleset_mismatch_fail",
            title="Ruleset integrity: CLAUDE.md check FAILED",
            summary=checks["claude_declared"][1],
            next_operator_action="investigate"
        )
        return 1

    declared_id = checks["claude_declared"][1]

    # Check 2: Extract pinned ruleset from run_screen.py
    run_screen_id = extract_pinned_ruleset_from_file(REPO_ROOT / "run_screen.py")
    if run_screen_id and run_screen_id == declared_id:
        logger.info(f"Check 2 PASS: run_screen.py pins {run_screen_id}")
        checks["run_screen_pinned"] = ("PASS", run_screen_id)
    else:
        logger.warning(f"Check 2 WARN: run_screen.py pins {run_screen_id} (expected {declared_id})")
        checks["run_screen_pinned"] = ("WARN", run_screen_id)

    # Check 3: Extract pinned ruleset from run_phase2_snapshot_delta.py
    phase2_id = extract_pinned_ruleset_from_file(REPO_ROOT / "run_phase2_snapshot_delta.py")
    if phase2_id and phase2_id == declared_id:
        logger.info(f"Check 3 PASS: run_phase2_snapshot_delta.py pins {phase2_id}")
        checks["phase2_pinned"] = ("PASS", phase2_id)
    else:
        logger.warning(f"Check 3 WARN: run_phase2_snapshot_delta.py pins {phase2_id} (expected {declared_id})")
        checks["phase2_pinned"] = ("WARN", phase2_id)

    # Check 4: Verify ruleset exists in manifest
    exists, error = check_ruleset_file_exists(declared_id)
    if exists:
        logger.info(f"Check 4 PASS: Ruleset {declared_id} exists in manifest")
        checks["manifest_has_id"] = ("PASS", declared_id)
    else:
        logger.error(f"Check 4 FAILED: {error}")
        checks["manifest_has_id"] = ("FAIL", error)

    # Determine overall result
    failures = [v for v in checks.values() if isinstance(v, tuple) and v[0] == "FAIL"]
    warnings = [v for v in checks.values() if isinstance(v, tuple) and v[0] == "WARN"]

    if not failures:
        # PASS (even if warnings)
        checks["all_match"] = True
        summary = f"Ruleset {declared_id} validation PASS. "
        if warnings:
            summary += f"({len(warnings)} warnings; see artifact for details)"

        try:
            send_operator_event(
                channel="town",
                severity="INFO",
                event_type="ruleset_mismatch_pass",
                title=f"Ruleset integrity PASS: {declared_id}",
                summary=summary,
                next_operator_action="none",
                extra={
                    "declared_id": declared_id,
                    "checks_passed": sum(1 for v in checks.values() if isinstance(v, tuple) and v[0] == "PASS"),
                    "checks_warned": len(warnings),
                    "checks_failed": len(failures),
                }
            )
            logger.info("PASS event routed to Town")
            return 0
        except Exception as e:
            logger.error(f"Failed to route PASS event: {e}", exc_info=True)
            return 1

    else:
        # FAIL
        summary = f"Ruleset {declared_id} validation FAILED. "
        summary += f"{len(failures)} check(s) failed. See artifact for details."

        try:
            send_operator_event(
                channel="town",
                severity="FAIL",
                event_type="ruleset_mismatch_fail",
                title=f"Ruleset integrity FAILED: {declared_id}",
                summary=summary,
                next_operator_action="investigate",
                extra={
                    "declared_id": declared_id,
                    "checks_passed": sum(1 for v in checks.values() if isinstance(v, tuple) and v[0] == "PASS"),
                    "checks_warned": len(warnings),
                    "checks_failed": len(failures),
                }
            )
            logger.error("FAIL event routed to Town")
            return 1
        except Exception as e:
            logger.error(f"Failed to route FAIL event: {e}", exc_info=True)
            return 1


if __name__ == "__main__":
    import time

    _started = time.perf_counter()
    _rc = main()
    try:
        from tools.agent_skill_telemetry import log_hermes_job_exit

        log_hermes_job_exit("hermes-ruleset-integrity", _rc, _started)
    except Exception:
        pass
    sys.exit(_rc)
