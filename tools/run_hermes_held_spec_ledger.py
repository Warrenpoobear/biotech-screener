#!/usr/bin/env python3
"""Thin wrapper: read held-spec ledger + knowledge layer, inline into message, invoke LLM agent.

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


def main():
    parser = argparse.ArgumentParser(description="Run hermes-held-spec-ledger agent with inline ledger data")
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
    args = parser.parse_args()

    # Load input files
    ledger_path = PROJECT_ROOT / "artifacts" / "ops" / "held_spec_ledger" / "latest.json"
    knowledge_path = PROJECT_ROOT / "artifacts" / "ops" / "knowledge_layer" / "latest_state.json"

    ledger = load_json(ledger_path)
    knowledge = load_json(knowledge_path)

    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Build the message with inline data
    msg_parts = []
    msg_parts.append(f"as_of_time: {now}")

    if args.dry_run:
        msg_parts.append("[DRY RUN]")

    if ledger:
        msg_parts.append(f"\n--- Held Spec Ledger ({ledger_path}) ---\n{json.dumps(ledger, indent=2)}")
    else:
        msg_parts.append(f"\n--- Held Spec Ledger ({ledger_path}) ---\nMISSING or UNREADABLE")

    if knowledge:
        msg_parts.append(f"\n--- Knowledge Layer State ({knowledge_path}) ---\n{json.dumps(knowledge, indent=2)}")
    else:
        msg_parts.append(f"\n--- Knowledge Layer State ({knowledge_path}) ---\nMISSING or UNREADABLE")

    if args.dry_run:
        msg_parts.append("\n[DRY RUN] Do NOT write to memory even if --write-memory is set.")
    else:
        msg_parts.append("\nRun full analysis and write result to memory.")

    message = "\n\n".join(msg_parts)

    # Build subprocess args
    run_args = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "run_agent_direct.py"),
        "--agent",
        "hermes-held-spec-ledger",
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
    print("[hermes-held-spec-ledger-wrapper] Invoking agent...", file=sys.stderr)
    result = subprocess.run(run_args, cwd=PROJECT_ROOT)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
