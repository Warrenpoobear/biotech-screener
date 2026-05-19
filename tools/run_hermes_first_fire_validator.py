#!/usr/bin/env python3
"""Thin wrapper: read first-fire ledger + knowledge layer, inline into message, invoke LLM agent.

Read-only. Never modifies the ledger. Use --dry-run to suppress memory writes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"[WARN] Cannot read {path}: {e}", file=sys.stderr)
        return None


def check_artifact(path: Path) -> dict:
    """Check if an artifact file exists and its basic metadata."""
    if not path.exists():
        return {"path": str(path), "status": "MISSING"}
    stat = path.stat()
    return {
        "path": str(path),
        "status": "EXISTS",
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "empty": stat.st_size == 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run hermes-first-fire-validator agent with inline ledger + artifact data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Dry-run mode: prepend [DRY RUN] to message, no memory write",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        help="Model for LLM call (default Llama via Together, claude-* via Anthropic)",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        default=False,
        help="Skip preflight governance check",
    )
    parser.add_argument(
        "--check-artifacts",
        action="store_true",
        default=True,
        help="Check referenced artifact paths and include in message (default: True)",
    )
    args = parser.parse_args()

    # Load input files
    first_fire_path = PROJECT_ROOT / "artifacts" / "ops" / "first_fire_ledger" / "latest.json"
    knowledge_path = PROJECT_ROOT / "artifacts" / "ops" / "knowledge_layer" / "latest_state.json"

    first_fire = load_json(first_fire_path)
    knowledge = load_json(knowledge_path)

    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Build the message with inline data
    msg_parts = []
    msg_parts.append(f"as_of_time: {now}")

    if args.dry_run:
        msg_parts.append("[DRY RUN]")

    # First-fire ledger
    if first_fire:
        msg_parts.append(f"\n--- First-Fire Ledger ({first_fire_path}) ---\n{json.dumps(first_fire, indent=2)}")
    else:
        msg_parts.append(f"\n--- First-Fire Ledger ({first_fire_path}) ---\nMISSING or UNREADABLE")

    # Knowledge layer
    if knowledge:
        msg_parts.append(f"\n--- Knowledge Layer State ({knowledge_path}) ---\n{json.dumps(knowledge, indent=2)}")
    else:
        msg_parts.append(f"\n--- Knowledge Layer State ({knowledge_path}) ---\nMISSING or UNREADABLE")

    # Check referenced artifacts (lightweight)
    if args.check_artifacts and first_fire:
        jobs = first_fire.get("jobs", [])
        artifact_checks = []
        for job in jobs:
            for art_path in job.get("expected_artifacts", []):
                full_path = PROJECT_ROOT / art_path
                artifact_checks.append(check_artifact(full_path))
            log_path = job.get("expected_log")
            if log_path:
                full_log = PROJECT_ROOT / log_path
                artifact_checks.append(check_artifact(full_log))

        if artifact_checks:
            msg_parts.append(f"\n--- Referenced Artifact Status ---\n{json.dumps(artifact_checks, indent=2)}")

    # Final instruction
    if args.dry_run:
        msg_parts.append("\n[DRY RUN] Do NOT write to memory even if --write-memory is set.")
    else:
        msg_parts.append("\nRun full validation and write result to memory.")

    message = "\n\n".join(msg_parts)

    # Build subprocess args
    run_args = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "run_agent_direct.py"),
        "--agent",
        "hermes-first-fire-validator",
        "--message",
        message,
    ]
    if args.model:
        run_args.extend(["--model", args.model])
    if args.dry_run:
        run_args.append("--skip-preflight")
    if not args.dry_run:
        run_args.append("--write-memory")

    # Run the agent
    print("[hermes-first-fire-validator-wrapper] Invoking agent...", file=sys.stderr)
    result = subprocess.run(run_args, cwd=PROJECT_ROOT)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
