#!/usr/bin/env python3
"""Recompute golden output fingerprint and update the pinned constant.

Reads EXPECTED_FINGERPRINT from tests/test_decision_engine_contract.py,
recomputes the fingerprint via _compute_golden_output_fingerprint(), and
(unless --dry-run) writes the new value back into the test file.

Also emits a diff summary showing tier/band/eligible for each golden
fixture so the developer can assess whether the change is material.

CLI:
  python scripts/refresh_goldens.py
  python scripts/refresh_goldens.py --dry-run
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_decision_engine_contract import (
    ALL_GOLDEN,
    _compute_golden,
    _compute_golden_output_fingerprint,
)

# Pattern matching `EXPECTED_FINGERPRINT = "..."` inside the test file.
_FP_PATTERN = re.compile(r'(EXPECTED_FINGERPRINT\s*=\s*)"([a-f0-9]+)"')

TEST_FILE = PROJECT_ROOT / "tests" / "test_decision_engine_contract.py"


def _golden_summary() -> str:
    """Return a table showing tier/band/eligible for each golden fixture."""
    lines = ["  Ticker         Tier  Band  Eligible"]
    lines.append("  " + "-" * 38)
    for g in ALL_GOLDEN:
        fields = _compute_golden(g)
        ticker = g["ticker"]
        tier = fields.get("tier_dev", "-")
        band = fields.get("size_band", "-")
        elig = fields.get("eligible", "-")
        lines.append(f"  {ticker:<14s} {tier or '-':<5s} {band or '-':<5s} {elig}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute golden output fingerprint and update test constant."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing.",
    )
    args = parser.parse_args()

    # Read current constant from test file
    source = TEST_FILE.read_text(encoding="utf-8")
    m = _FP_PATTERN.search(source)
    if not m:
        print("ERROR: Could not find EXPECTED_FINGERPRINT in test file.", file=sys.stderr)
        return 1

    old_fp = m.group(2)
    new_fp = _compute_golden_output_fingerprint()

    if old_fp == new_fp:
        print(f"Fingerprint unchanged: {new_fp}")
        return 0

    # Changed — show diff summary
    print(f"Fingerprint changed: {old_fp} -> {new_fp}")
    print()
    print("Golden fixture summary:")
    print(_golden_summary())
    print()

    if args.dry_run:
        print("[dry-run] Would update EXPECTED_FINGERPRINT in:")
        print(f"  {TEST_FILE}")
        return 0

    # Write updated constant
    new_source = source[:m.start(2)] + new_fp + source[m.end(2):]
    TEST_FILE.write_text(new_source, encoding="utf-8")
    print(f"Updated EXPECTED_FINGERPRINT in {TEST_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
