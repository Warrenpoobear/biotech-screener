#!/usr/bin/env python3
"""
Path C Window Close Decision Automation (2026-06-03+)

Implements the hard decision logic at window close:
1. Check IC observability
2. If observable: evaluate against floor (0.0200)
3. If unobservable: operator chooses extend or revert
4. Execute decision and document outcome

Usage:
    python3 tools/path_c_window_close_decision.py
    python3 tools/path_c_window_close_decision.py --output-json
    python3 tools/path_c_window_close_decision.py --write --as-of-date 2026-06-24
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GOV_DIR = REPO_ROOT / "artifacts" / "governance"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.forward_evidence_package import (  # noqa: E402
    IC_FLOOR,
    PATH_C_WINDOW_END,
    path_c_close_decision,
)


def decision_tree(*, window_end: str = PATH_C_WINDOW_END, floor: float = IC_FLOOR) -> dict:
    """Execute decision tree at window close (stdout + structured outcome)."""
    print("\n" + "=" * 80)
    print(f"PATH C WINDOW CLOSE DECISION — through {window_end}")
    print("=" * 80 + "\n")

    print("[STEP 1] Checking IC observability...")
    outcome_block = path_c_close_decision(window_end=window_end)
    ic_status = outcome_block["ic_status"]
    print(f"  Status: {ic_status['status']}")
    if ic_status["latest_date"]:
        print(f"  Latest observation date: {ic_status['latest_date']}")
    if ic_status["latest_ic"] is not None:
        print(f"  Latest mean_ic: {ic_status['latest_ic']:.4f}")
    if ic_status["reason"]:
        print(f"  Reason: {ic_status['reason']}")
    print()

    decision = outcome_block["decision"]
    action = outcome_block["action"]

    if ic_status["observable"]:
        print("[STEP 2] IC IS OBSERVABLE — Evaluate against floor")
        latest_ic = ic_status["latest_ic"]
        gap = (latest_ic - floor) if latest_ic is not None else None
        verdict = "✓ PASS" if decision == "PATH_C_VALID" else "✗ FAIL"
        print(f"  Floor threshold: {floor:.4f}")
        if gap is not None:
            print(f"  Gap: {gap:+.4f}")
        print(f"  Verdict: {verdict}")
        print(f"  Action: {action}")
        print()
        next_steps = (
            "Continue monitoring; proceed with next governance cycle; begin Path A design post-freeze"
            if decision == "PATH_C_VALID"
            else "Escalate to governance review; document evidence; begin Path A design work"
        )
    else:
        print("[STEP 2] IC IS UNOBSERVABLE — Operator decision required")
        print(f"  Reason: {ic_status['reason']}")
        print()
        print("  OPTION A: Extend observation window (recommended if conviction remains)")
        print("    - Extend Path C until first observable IC print")
        print("    - Evaluate against floor at that date")
        print()
        print("  OPTION B: Revert to HOLD (conservative if uncertain)")
        print("    - Revert to HOLD pending Path A design")
        print("    - Triggers Path A portfolio timing gate design (post-freeze)")
        print()
        verdict = "DECISION_REQUIRED"
        next_steps = "Document decision and rationale in governance ledger with timestamp"
        print(f"  Verdict: {verdict}")
        print(f"  Required action: {action}")
        print()

    print("[STEP 3] Document decision")
    print(f"  Path: artifacts/governance/path_c_window_close_{window_end}.json")
    print("  Or run: FREEZE_LIFT_ACK=1 python3 tools/forward_evidence_package.py --write")
    print()

    outcome = {
        "date": datetime.now().isoformat(),
        "window_close_date": window_end,
        "ic_status": ic_status["status"],
        "ic_observable": ic_status["observable"],
        "ic_value": ic_status["latest_ic"],
        "ic_floor": floor,
        "decision": decision,
        "action": action,
        "next_steps": next_steps,
        "overdue_note": outcome_block.get("overdue_note"),
    }

    print("=" * 80)
    print(f"SUMMARY: {decision}")
    print("=" * 80)
    print()
    return outcome


def main() -> int:
    parser = argparse.ArgumentParser(description="Path C window close decision automation")
    parser.add_argument("--window-end", default=PATH_C_WINDOW_END, help="Window end YYYY-MM-DD")
    parser.add_argument("--floor", type=float, default=IC_FLOOR, help="IC floor threshold")
    parser.add_argument("--as-of-date", help="Write artifact date stamp (default: window-end)")
    parser.add_argument("--write", action="store_true", help="Write JSON to artifacts/governance/")
    parser.add_argument("--output-json", action="store_true", help="Output decision as JSON")
    args = parser.parse_args()

    outcome = decision_tree(window_end=args.window_end, floor=args.floor)

    if args.write:
        GOV_DIR.mkdir(parents=True, exist_ok=True)
        stamp = args.as_of_date or args.window_end
        out_path = GOV_DIR / f"path_c_window_close_{stamp}.json"
        payload = {
            "path_c_close": path_c_close_decision(window_end=args.window_end),
            "cli_outcome": outcome,
        }
        out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")

    if args.output_json:
        print(json.dumps(outcome, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
