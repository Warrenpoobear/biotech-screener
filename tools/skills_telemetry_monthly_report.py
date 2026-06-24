#!/usr/bin/env python3
"""Generate monthly skills telemetry report (advisory-only).

Thin CLI over hermes_skills_learning_loop_v2. Reads environment-tagged JSONL
from artifacts/skills_learning/ and writes a monthly markdown report.

Usage:
    python3 tools/skills_telemetry_monthly_report.py
    python3 tools/skills_telemetry_monthly_report.py --month 2026-06 --env prod
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.hermes_skills_learning_loop_v2 import generate_monthly_report


def main() -> int:
    ap = argparse.ArgumentParser(description="Monthly skills telemetry report (advisory-only)")
    ap.add_argument("--month", help="YYYY-MM (default: current UTC month)")
    ap.add_argument("--env", default="prod", choices=("prod", "test"), help="Log environment tag")
    args = ap.parse_args()

    report_path = generate_monthly_report(month_str=args.month, environment=args.env)
    print(f"Wrote monthly report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
