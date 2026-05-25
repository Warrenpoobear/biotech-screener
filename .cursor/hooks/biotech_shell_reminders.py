#!/usr/bin/env python3
"""Non-blocking beforeShellExecution reminders for biotech-screener."""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        data = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        print('{"permission": "allow"}')
        return 0

    cmd = data.get("command", "") or ""
    messages: list[str] = []

    if "run_screen" in cmd:
        if "--phase2" in cmd and "--decision-mode" not in cmd:
            messages.append("biotech-screener run_screen: use --decision-mode phase2, not --phase2.")
        if "results.json" in cmd and "results_agent" not in cmd:
            messages.append(
                "biotech-screener run_screen on Windows: atomic rename may PermissionError "
                "on locked results.json; use sandbox all permissions or a fresh output path."
            )

    if "compare_module5_versions" in cmd:
        messages.append(
            "biotech-screener backtest: compare_module5_versions needs "
            "from datetime import timedelta (not time.timedelta)."
        )

    if messages:
        print(
            json.dumps(
                {
                    "permission": "allow",
                    "agent_message": " ".join(messages),
                }
            )
        )
    else:
        print('{"permission": "allow"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
